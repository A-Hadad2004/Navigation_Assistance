# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,
    initialize_weights,
    model_info,
    profile,
    scale_img,
    select_device,
    time_sync,
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        #פונקצית מעבר
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        #פונקצית מעבר היא כוללת אפשרויות לפרופילינג- מדידת ביצועים ולוויזואליזציה- תיאור גרפי של הנתונים| של מאפיינים
        y, dt = [], []  # outputs
        #y: רשימה לשמירת הפלטים מכל שכבה.
        #dt: רשימה לשמירת נתוני הפרופילינג (אם אפשרות הפרופילינג מופעלת).
        for m in self.model:
            #לולאה זו עוברת על כל שכבה במודל.
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                #אם השכבה הנוכחית לא מקבלת את הקלט ישירות מהשכבה הקודמת, נקבל את הקלט מהשכבה שצוינה ב-m.f.
            if profile:
                self._profile_one_layer(m, x, dt)
                #אם פרופילינג מופעל, נקרא לפונקציה
                #_profile_one_layer
                #שתבצע פרופילינג על השכבה הנוכחית.
                #פרופילינג= תהליך של מדידת ובדיקת הביצועים של קוד או מודל
            x = m(x)  # run
            #הרצת הקלט דרך השכבה הנוכחית.
            y.append(x if m.i in self.save else None)  # save output
            #שמירת הפלט של השכבה אם היא נדרשת לשמירה (אם m.i נמצאת ברשימת השכבות שיש לשמור).
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
                #אם אפשרות הוויזואליזציה מופעלת, נקרא לפונקציה
                #feature_visualization
                #שתבצע וויזואליזציה על הפלט של השכבה הנוכחית.
                # הוויזואליזציה= להצגה גרפית של הנתונים, התהליכים או התוצאות המתקבלים מהמודל
        return x
    #הפונקציה מחזירה את הפלט הסופי לאחר המעבר דרך כל השכבות במודל.

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        #מבצעת פרופילינג על שכבה בודדת במודל YOLOv5, על מנת להעריך את ביצועי השכבה מבחינת
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        #אם זו השיכבה האחרונה, מעתיקים את הקלט כדי למנוע שינויים במקום  שעשויים לשנות את הקלט המקורי.
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        #משתמש בספריית
        #thop
        #כדי לחשב את מספר הפעולות הנקודתיות
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        #מודד את זמן הביצוע של השכבה על ידי הרצתה 10 פעמים ברציפות. הזמן נמדד בשניות ומומר למילישניות ומוסף לרשימת ה-dt.
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
            #אם השיכבה ראשונה רושמים את:הפרמטרים, הזמן,כותרת
            #אם השכבה אחרונה רושמים את: סך כל הזמנים שלקח לכל השכבות לבצע

    def fuse(self):
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        #מבצעת פעולה של מיזוג שכבות
        LOGGER.info("Fusing layers... ")
        #מדפיס הודעה ללוג המציינת שהתחיל תהליך המיזוג של השכבות.
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                # עובר על כל השכבות במודל ובודק אם השכבה היא מסוג
                # Conv או DWConv
                # ויש לה מאפיין
                # bn=(BatchNorm)
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                # אם השכבה עומדת בתנאים, היא עוברת מיזוג עם שכבת ה
                # -BatchNorm
                # :שלה באמצעות הפונקציה
                # fuse_conv_and_bn
                delattr(m, "bn")  # remove batchnorm
                # השכבה bn נמחקת מהמודל.
                m.forward = m.forward_fuse  # update forward
                # הפונקציה
                # forward
                # -של השכבה מעודכנת ל
                # forward_fuse,
                # שהיא פונקציה מותאמת לביצוע המיזוג.
        self.info()
        return self

    def info(self, verbose=False, img_size=640):
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        #מיועדת להדפיס מידע על המודל, כגון מבנה השכבות, מספר הפרמטרים, ונתונים אחרים שעשויים להיות רלוונטיים למשתמש.
        model_info(self, verbose, img_size)
        #משמשת להדפסת מידע על המודל בצורה מאורגנת

    def _apply(self, fn):
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        #משמשת ליישום טרנספורמציות על הטנזורים של המודל, כמו המרה ל
        # CPU, ל-GPU
        self = super()._apply(fn)
        #הפונקציה קוראת לפונקציה
        # _apply של המחלקה העליונה
        # (nn.Module של PyTorch)
        # כדי ליישם את הטרנספורמציה על כל הפרמטרים והבאפרים של המודל.
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            #בודקת אם זה זיהוי או סגמנטציה אם כן מעבירה את כול הפמטרים שינוי של הפונקציה שקיבלנו
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

#ההערות שלי אחרי השורה לא לפני
class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        #המודל קורא מקובץ ה-יאלם מציב באופן ברירת מחדל שהקלט יהיה 3
        super().__init__()
        #שולח למחלקת האב שלו
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        #בודק האם המשתנה cfg הוא דיקשנרי אם כן פשוט מציב אותו
        else:  # is *.yaml
            import yaml  # for torch hub
        #אחרת הוא מייבא את הספרייה yalm שמאפשרת קריאה של קיבצי יאלם

            self.yaml_file = Path(cfg).name
            #בגלל שכרגע המשתנה הוא רק נתיב
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict
                #פותחת את הנתיב וטוענת את הקובץ למצב דיקשנרי

        # Define model
        #מגדירה את המודל
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        #מגדירה את הקלט מהקובץ
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
            #אם קיים כבר מספר מחלקות זאת אומרת שזה לא הפעם הראשונה ומספר המחלקות שונה מהקובץ הוא דורס את התוכן
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
            #אם קיים כבר עוגנים כלומר נשלחו למחלקה הזו אז הוא דורס את מה שיש במשתנה שמייצג את הקובץ
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        #השורה מעל בעצם מגדירה את השכבות של הבלוקים במודל לאחר שבשורות למעלה חילצנו את התכונות מסוימות עכשיו נחלץ את מבנה הבלוקים
        #אנחנו נשלח העתק עמוק של הדיקשנרי יאלם לפוקצייה
        #parse_model
        #שהיא בעצם מחלצת את כול מבנה הבלוקים ומחזירה מבנה שכבות רציף
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        #מקבל את שמות המחלקות מהקובץ
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        #השיכבה האחרונה היא של החיזוי
        if isinstance(m, (Detect, Segment)):
            #אם המודל הוא מסוג זיהוי או סגמנטציה
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            #שולח למעבר רגיל בגלל שזה מודל זיהוי
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            #יוצר את החלון הזזה על ידי שליחה לפונקציית המעבר טנזור אפסים שמייצג צורה של תמונה ומחלק את ה-משתנה שמכיל 256 חלקי כול האיברים שמחזיר הטנזור
            check_anchor_order(m)
            #פונקצייה מטורצ' שבודקת את תקינות העוגנים
            m.anchors /= m.stride.view(-1, 1, 1)
            #מחלקת את העוגנים לסטרייד על מנת להביא את העוגנים הסופיים
            self.stride = m.stride
            self._initialize_biases()  # only run once
            #לבדוק

        # Init weights, biases
        initialize_weights(self)
        #מגדירה את המשקלים למשקלים התחלתיים
        self.info()
        LOGGER.info("")
        #הינפו יהיה ריק

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        #פוקציית מעבר רגילה
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        #לשפר את הדיוק של המודל על ידי הפקת תחזיות ממספר גרסאות של אותה תמונה ולאחר מכן שילובן לתוצאה סופית אחת.
        #מתבצע בתהליך הזיהוי או המבחן
        img_size = x.shape[-2:]  # height, width
        #מחלץ את גובה ורוחב התמונה המקורית.
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        #s הוא רשימה של קני מידה שונים לביצוע האינפרנס.
        #f הוא רשימה של סוגי פליפ שונים (כמו פליפ אופקי או פליפ אנכי)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            #xi הוא התמונה המוגדלת ומתהפכת בהתאם ל-scale ול-flip.
            yi = self._forward_once(xi)[0]  # forward
            #yi הוא הפלט של המודל לתמונה המוגדלת.
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            # מתקן את ההערכות חזרה לגודל המקורי של התמונה.
            y.append(yi)
            #התוצאות  של כל שלב נשמרות ברשימה.
        y = self._clip_augmented(y)  # clip augmented tails
        #קורא לפונקציה _clip_augmented כדי לקצץ את התוצאות המוגדלות.
        return torch.cat(y, 1), None  # augmented inference, train
         #משלב את כל התוצאות (detections) למימד אחד ומחזיר אותן יחדיו.

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        # מיועדת להתאים את תחזיות האובייקטים שנמצאו במהלך אינפרנס מוגדל  לגודל המקורי של התמונה, ולהתאים אותן למצבי הפליפים שבוצעו
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        #משמש במהלך תהליך האינפרנס, מעבד את התוצאות שהוגדלו ומקצץ חלקים מהן כדי לשמור על התאמה נכונה למודל.
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        #מיועדת לאתחול הטיות
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        #cf הוא אופציונלי ומייצג את התדירות של המחלקות
        m = self.model[-1]  # Detect() module
        #מחלץ את המודול האחרון במודל, שהוא מודול הזיהוי (Detect).
        for mi, s in zip(m.m, m.stride):  # from
            #הלולאה עוברת על כל השכבות במודול הזיהוי ועל ה-strides שלהן.
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            #משנה את הצורה של ההטיה של השכבה הנוכחית כך שתתאים למספר העוגנים
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            #אתחול ההטיות עבור תחזיות האובייקטים
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            #אם cf אינו מסופק, הוא משתמש בערך ברירת מחדל המחושב על ידי הלוג של יחס 0.6 למספר המחלקות (עם תיקון קטן למניעת חלוקה באפס).
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            #מעדכן את ההטיה של השכבה הנוכחית ומגדיר שהיא ניתנת ללמידה


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


def parse_model(d, ch):
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    #הפונקציה  מבצעת פעולת פירוק של מודל  מתוך מבנה הנתונים שמוגדר בתוך המילון d.
    #היא מאפשרת להגדיר את שכבות המודל על פי מבנה השכבות שהוגדר בצורה דינמית, תוך התאמה למבנה הנתונים שמועברים אליה.
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:
        ch_mul = 8
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
