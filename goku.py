"""Local teleport effect for macOS, Windows and Linux."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import ssl
import platform
import time
import urllib.request

import cv2
import certifi
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "selfie_segmenter_landscape.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
}
TITLE = "Goku Teleport"
DURATION = 0.45


def download_models():
    MODELS.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        path = MODELS / name
        if path.is_file() and path.stat().st_size > 100_000:
            continue
        print(f"Downloading {name}…", flush=True)
        temporary = path.with_suffix(".part")
        try:
            with urllib.request.urlopen(url, timeout=60,
                    context=ssl.create_default_context(cafile=certifi.where())) as source:
                temporary.write_bytes(source.read())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def create_tasks(stack):
    base = python.BaseOptions
    video = vision.RunningMode.VIDEO
    segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=base(model_asset_path=str(MODELS / "selfie_segmenter_landscape.tflite")),
        running_mode=video, output_confidence_masks=True,
    ))
    stack.callback(segmenter.close)
    hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=base(model_asset_path=str(MODELS / "hand_landmarker.task")),
        running_mode=video, num_hands=2, min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4, min_tracking_confidence=0.4,
    ))
    stack.callback(hands.close)
    face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
        base_options=base(model_asset_path=str(MODELS / "blaze_face_short_range.tflite")),
        running_mode=video, min_detection_confidence=0.5,
    ))
    stack.callback(face.close)
    return segmenter, hands, face


def forehead_pose(points, box, width, height):
    if box is None or points.shape != (21, 3) or not np.isfinite(points).all():
        return False
    pixels = points[:, :2] * (width, height)
    palm = max(float(np.linalg.norm(pixels[5] - pixels[17])), 1.0)
    for start in (5, 9):
        finger = pixels[start:start + 4]
        length = np.linalg.norm(np.diff(finger, axis=0), axis=1).sum()
        if np.linalg.norm(finger[-1] - finger[0]) < 0.77 * max(length, 1.0):
            return False
        if np.linalg.norm(finger[-1] - pixels[0]) < 1.05 * np.linalg.norm(finger[1] - pixels[0]):
            return False
    if np.linalg.norm(pixels[8] - pixels[12]) > palm * 0.65:
        return False
    for tip, joint in ((16, 14), (20, 18)):
        if np.linalg.norm(pixels[tip] - pixels[0]) > 1.15 * np.linalg.norm(pixels[joint] - pixels[0]):
            return False
    x, y, w, h = box
    target = np.array([x + w * 0.5, y + h * 0.12])
    for tip in (8, 12):
        delta = (pixels[tip] - target) / (max(w * 0.55, 1), max(h * 0.38, 1))
        if float(delta @ delta) > 1:
            return False
    return True


@dataclass
class GestureLatch:
    held_since: float | None = None
    released_since: float | None = None
    armed: bool = True
    last_sample: float | None = None

    def update(self, pose, now):
        if self.last_sample is not None and now - self.last_sample > 0.35:
            self.held_since = None
        self.last_sample = now
        if not pose:
            self.held_since = None
            if self.released_since is None:
                self.released_since = now
            if now - self.released_since >= 0.3:
                self.armed = True
            return False
        self.released_since = None
        if not self.armed:
            return False
        if self.held_since is None:
            self.held_since = now
        if now - self.held_since < 0.6:
            return False
        self.armed = False
        self.held_since = None
        return True


def silhouette(segmenter, image, timestamp, size):
    result = segmenter.segment_for_video(image, timestamp)
    if not result.confidence_masks:
        raise RuntimeError("Could not detect your silhouette. Try improving the lighting.")
    raw = np.squeeze(result.confidence_masks[0].numpy_view())
    mask = cv2.resize(raw, size)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    return np.clip((mask - 0.12) / 0.68, 0, 1)


def render_teleport(person, background, mask, progress, returning=False):
    progress = float(np.clip(progress, 0, 1))
    visible = progress if returning else 1 - progress
    alpha = mask[:, :, None] * visible
    output = person.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    strength = float(np.sin(np.pi * progress))
    if strength > 0.001:
        h, w = mask.shape
        y, x = np.mgrid[:h, :w]
        columns = (x % 17 < 2) | (x % 31 == 0)
        streaks = ((y + int(progress * h * 3) + (x // 17) * 29) % 113 < 75)
        ink = (columns & streaks) * mask * strength
        output *= 1 - ink[:, :, None] * 0.9
        light = ((x + 7) % 23 == 0) * mask * strength
        output += light[:, :, None] * 110
    return np.clip(output, 0, 255).astype(np.uint8)


def make_sound():
    path = ROOT / "assets" / "goku-teleport-sound.wav"
    if not path.is_file():
        raise RuntimeError(f"Missing sound: {path}")
    return path


def camera_backends(system=None):
    system = system or platform.system()
    return {"Darwin": (cv2.CAP_AVFOUNDATION,),
            "Windows": (cv2.CAP_DSHOW, cv2.CAP_MSMF),
            "Linux": (cv2.CAP_V4L2,)}.get(system, (cv2.CAP_ANY,))


def open_camera(index):
    for backend in camera_backends():
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            return capture
        capture.release()
    raise RuntimeError("Cannot open webcam. Check camera permissions or try --camera 1.")


def virtual_backend(system=None):
    return "v4l2loopback" if (system or platform.system()) == "Linux" else "obs"


def run(args):
    if args.list_audio:
        from goku_audio import list_inputs
        list_inputs()
        return
    if args.audio_test:
        from goku_audio import create_bridge
        bridge = create_bridge(make_sound(), args.mic).start()
        try:
            print("Five-second audio test: speak; the effect will play after one second.", flush=True)
            time.sleep(1)
            bridge.play()
            time.sleep(4)
            bridge.check()
        finally:
            bridge.close()
        return
    download_models()
    with ExitStack() as stack:
        segmenter, hands, face = create_tasks(stack)
        if args.self_test:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            hands.detect_for_video(image, 1)
            face.detect_for_video(image, 1)
            silhouette(segmenter, image, 1, (640, 360))
            make_sound()
            print("Self-test passed: face, hand and segmentation models, and sound.")
            return
        virtual = None
        if not args.preview:
            import pyvirtualcam
            try:
                virtual = pyvirtualcam.Camera(width=960, height=540, fps=30,
                    fmt=pyvirtualcam.PixelFormat.BGR, backend=virtual_backend(),
                    device=args.virtual_device)
            except RuntimeError as error:
                raise RuntimeError("Virtual camera unavailable. See README for OBS (Mac/Windows) or v4l2loopback (Linux).\n" + str(error)) from error
            stack.callback(virtual.close)
        capture = open_camera(args.camera)
        stack.callback(capture.release)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        capture.set(cv2.CAP_PROP_FPS, 30)
        stack.callback(cv2.destroyAllWindows)
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, 960, 540)
        print("Leave the frame empty for 4 seconds. SPACE recaptures the background. Q quits.", flush=True)
        print("Then hold two fingers together against your forehead for 0.6 seconds. Lower your hand before repeating.", flush=True)
        print("Meet / Zoom: select Goku on Linux or OBS Virtual Camera on Mac/Windows. Disable meeting backgrounds and effects.", flush=True)
        sound = make_sound()
        audio_bridge = None
        if args.virtual_audio:
            from goku_audio import create_bridge
            audio_bridge = create_bridge(sound, args.mic).start()
            stack.callback(audio_bridge.close)
        from goku_audio import LocalSound
        local_sound = None
        if not args.mute and not args.no_local_sound:
            local_sound = LocalSound(sound).start()
            stack.callback(local_sound.close)
        calibrated_at = time.monotonic() + 4
        background = None
        samples = []
        latch = GestureLatch()
        hidden = False
        transition = None
        last_detection = -1.0
        started = time.monotonic()
        while True:
            if audio_bridge is not None:
                audio_bridge.check()
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The webcam is not returning frames. Check camera permissions and reopen the launcher.")
            frame = cv2.resize(frame, (960, 540))
            now = time.monotonic()
            timestamp = int((now - started) * 1000)
            if background is None:
                if now >= calibrated_at - 0.6:
                    samples.append(frame.copy())
                if now >= calibrated_at:
                    background = np.median(samples[-18:], axis=0).astype(np.uint8)
                    samples.clear()
                    print("Background ready. You can return to the frame.", flush=True)
                display = frame
            else:
                if transition is None and now - last_detection >= 0.09:
                    last_detection = now
                    image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_result = face.detect_for_video(image, timestamp)
                    boxes = [(d.bounding_box.origin_x, d.bounding_box.origin_y,
                              d.bounding_box.width, d.bounding_box.height)
                             for d in face_result.detections]
                    box = max(boxes, key=lambda b: b[2] * b[3]) if boxes else None
                    hand_result = hands.detect_for_video(image, timestamp)
                    pose = any(forehead_pose(np.array([(p.x, p.y, p.z) for p in hand]),
                               box, 960, 540) for hand in hand_result.hand_landmarks)
                    if latch.update(pose, now):
                        mask = silhouette(segmenter, image, timestamp, (960, 540))
                        if np.count_nonzero(mask > 0.5) > 500:
                            transition = (time.monotonic(), frame.copy(), mask)
                            if not args.mute:
                                if audio_bridge is not None:
                                    audio_bridge.play()
                                if local_sound is not None:
                                    local_sound.play()
                            print("Reappearing…" if hidden else "Teleporting…", flush=True)
                if transition is not None:
                    at, person, mask = transition
                    progress = min((time.monotonic() - at) / DURATION, 1.0)
                    display = render_teleport(person, background, mask, progress, returning=hidden)
                    if progress >= 1:
                        hidden = not hidden
                        transition = None
                else:
                    display = background.copy() if hidden else frame
            if virtual is not None:
                virtual.send(np.ascontiguousarray(display))
            cv2.imshow(TITLE, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                background = None
                samples.clear()
                calibrated_at = time.monotonic() + 4
                hidden = False
                transition = None
                latch = GestureLatch()
                print("Leave the frame empty for 4 seconds to capture the background.", flush=True)
            if virtual is not None:
                virtual.sleep_until_next_frame()


def main():
    parser = argparse.ArgumentParser(description="Goku Teleport")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--preview", action="store_true", help="Local preview without a virtual camera")
    parser.add_argument("--mute", action="store_true", help="Mute the effect while keeping your voice in the virtual microphone")
    parser.add_argument("--virtual-audio", "--blackhole", dest="virtual_audio",
                        action=argparse.BooleanOptionalAction, default=False,
                        help="Mix your microphone and the effect into a virtual microphone")
    parser.add_argument("--no-local-sound", action="store_true",
                        help="Only send the effect to the call")
    parser.add_argument("--virtual-device", help="Virtual camera device, e.g. /dev/video10")
    parser.add_argument("--mic", help="Physical microphone name or index")
    parser.add_argument("--list-audio", action="store_true", help="List audio input devices")
    parser.add_argument("--audio-test", action="store_true", help="Test five seconds of audio without opening the webcam")
    parser.add_argument("--self-test", action="store_true")
    try:
        run(parser.parse_args())
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"\nError: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
