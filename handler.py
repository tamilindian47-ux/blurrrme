import base64, io, os, subprocess, tempfile, traceback
import numpy as np
import cv2
import imageio_ffmpeg
import runpod
import mediapipe as mp
from PIL import Image, ImageSequence, ImageOps
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = "/models/face_landmarker.task"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
MAX_OUT_BYTES = 6_800_000

GROUPS = {
    "EYES": [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398,
             33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173],
    "BROWS": [276, 283, 282, 295, 285, 300, 293, 334, 296, 336,
              46, 53, 52, 65, 55, 70, 63, 105, 66, 107],
    "NOSE": [6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 97, 326, 327, 129, 358, 49, 279, 48, 278, 64, 294],
    "LIPS": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "OVAL": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
             377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
}
REGIONS = {
    "eyes": ["EYES"],
    "eyes_brows": ["EYES", "BROWS"],
    "eyes_nose": ["EYES", "NOSE"],
    "nose": ["NOSE"],
    "mouth": ["LIPS"],
    "nose_mouth": ["NOSE", "LIPS"],
    "eyes_nose_mouth": ["EYES", "BROWS", "NOSE", "LIPS"],
    "whole_face": ["OVAL"],
}


def dedupe(faces):
    kept = []
    for f in sorted(faces, key=lambda f: -float(np.ptp(f[:, 0]))):
        c = f.mean(axis=0)
        s = float(np.ptp(f[:, 0]))
        if all(np.linalg.norm(c - k[1]) > 0.4 * max(s, k[2]) for k in kept):
            kept.append((f, c, s))
    return [k[0] for k in kept]


class Detector:
    def __init__(self, video, max_faces, small):
        self.small = small
        self.video = video and not small
        mode = vision.RunningMode.VIDEO if self.video else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU),
            running_mode=mode, num_faces=max_faces,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)
        self.max_faces = max_faces

    def _faces(self, img, ox, oy, ts, scale=1.0):
        h, w = img.shape[:2]
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
        if self.video:
            res = self.lm.detect_for_video(mp_img, ts)
        else:
            res = self.lm.detect(mp_img)
        out = []
        for face in res.face_landmarks:
            out.append(np.array([[l.x * w / scale + ox, l.y * h / scale + oy] for l in face], dtype=np.float32))
        return out

    def detect(self, rgb, ts):
        H, W = rgb.shape[:2]
        faces = self._faces(rgb, 0, 0, ts)
        if self.small:
            tw, th = int(W * 0.6), int(H * 0.6)
            scale = max(1.0, min(3.0, 1024.0 / max(tw, th)))
            for ox in (0, W - tw):
                for oy in (0, H - th):
                    crop = rgb[oy:oy + th, ox:ox + tw]
                    if scale > 1.0:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                    faces += self._faces(crop, ox, oy, ts, scale)
        return dedupe(faces)[: self.max_faces]


def build_mask(pts, W, H, shape, pad, flat):
    mask = np.zeros((H, W), np.uint8)
    if shape == "tight":
        hull = cv2.convexHull(pts).reshape(-1, 2)
        c = hull.mean(axis=0)
        hull = (hull - c) * (1 + pad) + c
        cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 255)
        return mask
    (cx, cy), (w, h), ang = cv2.minAreaRect(pts)
    if w < h:
        w, h, ang = h, w, ang + 90
    w *= 1 + pad
    h *= 1 + pad
    if flat:
        h = max(h, 0.34 * w)
    if shape == "ellipse":
        cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                    (max(1, int(w / 2 * 1.2)), max(1, int(h / 2 * 1.2))), ang, 0, 360, 255, -1)
    else:
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        cv2.fillConvexPoly(mask, np.round(box).astype(np.int32), 255)
    return mask


def apply_effect(frame, mask, effect, intensity):
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    size = max(x1 - x0, y1 - y0)
    t = intensity / 100.0
    if effect == "bar":
        frame[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0] = 0
        return
    if effect == "blur":
        sigma = max(1.0, (0.03 + 0.37 * t) * size)
        mg = int(sigma * 2)
        a0, a1 = max(0, y0 - mg), min(H, y1 + mg)
        b0, b1 = max(0, x0 - mg), min(W, x1 + mg)
        crop = frame[a0:a1, b0:b1]
        bl = cv2.GaussianBlur(crop, (0, 0), sigma)
        m = mask[a0:a1, b0:b1] > 0
        crop[m] = bl[m]
        return
    cells = max(3, int(round(40 - 37 * t)))
    block = max(2, int(np.ceil(size / cells)))
    crop = frame[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    small = cv2.resize(crop, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_AREA)
    px = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    m = mask[y0:y1, x0:x1] > 0
    crop[m] = px[m]


class Censor:
    def __init__(self, o, video):
        self.o = o
        self.idx = [i for g in REGIONS[o["region"]] for i in GROUPS[g]]
        self.flat = o["region"] == "eyes"
        self.det = Detector(video, o["max_faces"], o["small_faces"])
        self.video = video
        self.prev, self.miss, self.hold = [], 0, 5
        self.total = 0
        self.frames_with_faces = 0

    def process(self, rgb, ts):
        H, W = rgb.shape[:2]
        faces = self.det.detect(rgb, ts)
        if faces:
            self.prev, self.miss = faces, 0
        elif self.video and self.prev and self.miss < self.hold:
            faces = self.prev
            self.miss += 1
        if faces:
            self.frames_with_faces += 1
            self.total = max(self.total, len(faces))
        for f in faces:
            pts = f[self.idx].astype(np.float32)
            mask = build_mask(pts, W, H, self.o["shape"], self.o["pad"], self.flat)
            apply_effect(rgb, mask, self.o["effect"], self.o["intensity"])
        return rgb


def process_image(raw, o):
    im = Image.open(io.BytesIO(raw))
    fmt = im.format
    im = ImageOps.exif_transpose(im)
    cen = Censor(o, False)
    out = Image.fromarray(cen.process(np.array(im.convert("RGB")), 0))
    buf = io.BytesIO()
    if fmt == "JPEG":
        out.save(buf, "JPEG", quality=95)
        mime = "image/jpeg"
    else:
        out.save(buf, "PNG")
        mime = "image/png"
        if buf.tell() > MAX_OUT_BYTES:
            buf = io.BytesIO()
            out.save(buf, "JPEG", quality=92)
            mime = "image/jpeg"
    return {"mediatype": "image", "mime": mime, "base64": base64.b64encode(buf.getvalue()).decode(),
            "faces": cen.total}


def process_gif(raw, o):
    im = Image.open(io.BytesIO(raw))
    loop = im.info.get("loop", 0)
    frames, durs = [], []
    for fr in ImageSequence.Iterator(im):
        durs.append(max(20, int(fr.info.get("duration", 100))))
        frames.append(np.array(fr.convert("RGB")))
    cen = Censor(o, True)
    outs, ts = [], 0
    for arr, d in zip(frames, durs):
        outs.append(Image.fromarray(cen.process(arr, ts)))
        ts += d
    scale = 1.0
    cur = outs
    while True:
        buf = io.BytesIO()
        cur[0].save(buf, "GIF", save_all=True, append_images=cur[1:], duration=durs,
                    loop=loop, disposal=2, optimize=False)
        if buf.tell() <= MAX_OUT_BYTES or scale < 0.3:
            break
        scale *= 0.75
        size = (max(16, int(outs[0].width * scale)), max(16, int(outs[0].height * scale)))
        cur = [f.resize(size, Image.LANCZOS) for f in outs]
    return {"mediatype": "image", "mime": "image/gif", "base64": base64.b64encode(buf.getvalue()).decode(),
            "faces": cen.total}


def process_video(raw, o):
    with tempfile.TemporaryDirectory() as d:
        src, dst, log = f"{d}/in.mp4", f"{d}/out.mp4", f"{d}/ff.log"
        with open(src, "wb") as f:
            f.write(raw)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise ValueError("Could not open the video file.")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 120:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, fr = cap.read()
        if not ok:
            raise ValueError("The video has no readable frames.")
        h, w = fr.shape[:2]
        dur = total / fps if total > 0 else 30.0
        vbits = int(max(300_000, MAX_OUT_BYTES * 8 / max(dur, 1.0) - 128_000))
        cmd = [FFMPEG, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "pipe:0",
               "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
               "-maxrate", str(vbits), "-bufsize", str(vbits * 2), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", dst]
        cen = Censor(o, True)
        with open(log, "wb") as lf:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=lf)
            i = 0
            try:
                while ok:
                    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    out = cen.process(rgb, int(i * 1000 / fps))
                    proc.stdin.write(out.tobytes())
                    i += 1
                    ok, fr = cap.read()
                proc.stdin.close()
            except BrokenPipeError:
                pass
            rc = proc.wait()
        cap.release()
        if rc != 0 or not os.path.exists(dst):
            msg = open(log, "rb").read().decode(errors="ignore")[-400:]
            raise RuntimeError("ffmpeg failed: " + msg)
        with open(dst, "rb") as f:
            data = f.read()
    return {"mediatype": "video", "mime": "video/mp4", "base64": base64.b64encode(data).decode(),
            "faces": cen.total, "frames": i, "frames_with_faces": cen.frames_with_faces}


def parse(inp):
    region = inp.get("region", "eyes")
    effect = inp.get("effect", "blur")
    shape = inp.get("shape", "box")
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'.")
    if effect not in ("blur", "pixelate", "bar"):
        raise ValueError(f"Unknown effect '{effect}'.")
    if shape not in ("box", "ellipse", "tight"):
        raise ValueError(f"Unknown shape '{shape}'.")
    return {
        "region": region, "effect": effect, "shape": shape,
        "intensity": max(1, min(100, int(inp.get("intensity", 60)))),
        "pad": max(0, min(100, float(inp.get("padding", 20)))) / 100.0,
        "max_faces": max(1, min(20, int(inp.get("max_faces", 10)))),
        "small_faces": bool(inp.get("small_faces", False)),
    }


def handler(job):
    try:
        inp = job.get("input", {})
        b64 = inp.get("file")
        if not b64:
            return {"error": "Missing 'file' (base64)."}
        if "," in b64[:100]:
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        o = parse(inp)
        kind = inp.get("media", "image")
        print(f"Job {kind} {len(raw)//1024}KB {o}", flush=True)
        if kind == "video":
            return process_video(raw, o)
        if kind == "gif":
            return process_gif(raw, o)
        return process_image(raw, o)
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    try:
        Detector(False, 1, False)
        print("Face landmarker ready", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    runpod.serverless.start({"handler": handler})
