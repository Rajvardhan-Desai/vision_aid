import os, sys, time, logging, threading, collections
from typing import Dict, Any
import cv2

from .config import parse_config
from .gpio_shim import GPIO, is_simulated
from .queues import PriorityMsgQueue
from .audio import (
    SpeechEngine,
    audio_thread_func,
    queue_audio_message,
    set_audio_enabled,
)
from .camera import Camera
from .utils import Throttle, parse_wh, resize_with_ratio, calculate_adaptive_inference_size
from .threads import ThreadManager
from .detection import (
    Detector,
    process_currency_detections,
    update_object_history,
    detect_with_mode,
)
from .gps_email import get_gps_location, send_email
from .faces import FaceDB
from .motor import VibrationController

log = logging.getLogger("vision_aid")

TRIG_PIN = 23
ECHO_PIN = 24
MOTOR_PIN = 25

FACE_RECOGNITION_INTERVAL = 5
GRID_SIZE = 50
REQUIRED_FRAMES = 3
HARMFUL_OBJECTS = {"knife", "scissors", "fire", "gun"}
CURRENCY_MODE_TIMEOUT = 60  # seconds


def safe_setup_gpio() -> bool:
    try:
        for pin in (TRIG_PIN, ECHO_PIN, MOTOR_PIN):
            if not isinstance(pin, int) or pin < 0:
                raise ValueError(f"Invalid GPIO pin: {pin}")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(TRIG_PIN, GPIO.OUT)
        GPIO.setup(ECHO_PIN, GPIO.IN)
        GPIO.output(TRIG_PIN, False)
        GPIO.setup(MOTOR_PIN, GPIO.OUT)
        GPIO.output(MOTOR_PIN, False)
        log.info("GPIO initialized (%s)", "Sim" if is_simulated() else "RPi.GPIO")
        return True
    except Exception as e:
        log.error("GPIO init failed: %s", e)
        return False


def load_yolo_model(path: str):
    from ultralytics import YOLO
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    log.info("Loading YOLO model: %s", path)
    return YOLO(path, task='detect')


def run_single_image(model, image_path: str, thresh: float):
    from .detection import Detector
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError("Failed to read image")
    det = Detector(model)
    dets = det.detect_all(img, conf_thres=thresh)
    return dets


def distance_cm() -> float:
    try:
        GPIO.output(TRIG_PIN, True); time.sleep(0.00001); GPIO.output(TRIG_PIN, False)
        timeout = time.time() + 0.02
        while GPIO.input(ECHO_PIN) == 0 and time.time() < timeout: pass
        start = time.time()
        timeout = time.time() + 0.02
        while GPIO.input(ECHO_PIN) == 1 and time.time() < timeout: pass
        end = time.time()
        dur = end - start
        return max(0.0, min((dur * 34300.0) / 2.0, 500.0))
    except Exception:
        return -1.0


def obstacle_priority(dist_cm: float) -> int:
    if dist_cm < 0: return 0
    if dist_cm < 40: return 4
    if dist_cm < 70: return 3
    if dist_cm < 110: return 2
    return 1


def motor_pattern_for_distance(vib: VibrationController, dist_cm: float, stop_evt):
    if dist_cm < 0: return
    if dist_cm < 40: vib.pattern_urgent(stop_evt)
    elif dist_cm < 70: vib.pattern_warning(stop_evt)
    elif dist_cm < 110: vib.pattern_gentle(stop_evt)
    # else do nothing


def registration_flow(args) -> int:
    """
    Handle --register-face and --list-known-faces flows.
    --register-face accepts either:
      NAME:path/to/image.jpg    (register from file)
      NAME                      (capture one frame from camera)
    """
    db = FaceDB()

    if args.list_known_faces:
        names = db.list_names()
        if not names:
            print("No known faces.")
        else:
            print("\n".join(names))
        return 0

    if args.register_face:
        # parse "NAME:path" or just "NAME"
        token = args.register_face
        if ":" in token:
            name, img_path = token.split(":", 1)
            img = cv2.imread(img_path)
            if img is None:
                log.error("Failed to read image for registration: %s", img_path)
                return 2
            db.register_from_image(name.strip(), img)
            print(f"Registered '{name.strip()}' from file.")
            return 0
        else:
            # capture one frame from camera
            cam = Camera(args.source, resolution=parse_wh(args.resolution))
            frame = cam.read(); cam.close()
            if frame is None:
                log.error("Camera returned empty frame for registration capture")
                return 2
            db.register_from_image(token.strip(), frame)
            print(f"Registered '{token.strip()}' from camera.")
            return 0

    return -1  # no registration-related action


def live_loop(stop_evt, args, audio_q: PriorityMsgQueue, speech: SpeechEngine, shared):
    res = parse_wh(args.resolution)
    infer_wh = parse_wh(args.inference_size) if args.inference_size else (640, 640)
    cam = Camera(args.source, resolution=res if res != (0,0) else None)
    model = load_yolo_model(args.model)
    det = Detector(model)
    currency_det = None
    faces = FaceDB() if args.face_recognition else None
    vib = VibrationController(MOTOR_PIN)

    speak_throttle = Throttle(2.5)
    log_throttle = Throttle(2.0)

    # distance smoothing (median over last N)
    window = collections.deque(maxlen=7)
    fps_window = collections.deque(maxlen=5)
    obj_history: Dict[Any, int] = {}
    frame_idx = 0

    while not stop_evt.is_set():
        t_start = time.time()
        frame = cam.read()
        if frame is None:
            if log_throttle.allow("cam_none"):
                log.warning("Camera returned empty frame")
            time.sleep(0.05)
            continue
        view = frame; fx = fy = 1.0
        if infer_wh:
            view, (fx, fy), _, _ = resize_with_ratio(frame, infer_wh)

        # Currency mode: load currency model lazily and run the correct detector
        currency_active = time.time() < shared.currency_mode_until
        if currency_active and currency_det is None:
            try:
                if not args.currency_model:
                    raise FileNotFoundError("No currency model path provided")
                currency_det = Detector(load_yolo_model(args.currency_model))
            except Exception as e:
                if log_throttle.allow("currency_load_fail"):
                    log.error("Currency model load failed: %s", e)
                shared.currency_mode_until = 0.0
                currency_active = False

        dets = detect_with_mode(
            view,
            det,
            currency_det,
            currency_active,
            obj_thresh=args.thresh,
            curr_thresh=max(0.85, args.thresh),
        )

        if currency_active:
            currency = process_currency_detections(
                dets, class_whitelist=None, min_conf=max(0.85, args.thresh), use_nms=True
            )
            shared.last_objects = [c["class_name"] for c in currency]
            if currency and speak_throttle.allow("currency"):
                msg = ", ".join(f"{c['class_name']} {c['conf']:.0%}" for c in currency[:3])
                queue_audio_message(audio_q, "currency", f"Currency: {msg}", priority=2)
        else:
            stable = update_object_history(
                obj_history, dets, grid_size=GRID_SIZE, required_frames=REQUIRED_FRAMES
            )
            shared.last_objects = [d["class_name"] for d in stable]
            for d in stable:
                if args.alert_all or d["class_name"] in HARMFUL_OBJECTS:
                    if speak_throttle.allow(d["class_name"]):
                        queue_audio_message(audio_q, d["class_name"], d["class_name"], priority=1)

            # Faces (dynamic scaling already inside FaceDB)
            if faces and frame_idx % FACE_RECOGNITION_INTERVAL == 0:
                fr = faces.recognize(view, fx=fx, fy=fy)
                # announce known ones with higher priority
                known = [f for f in fr if f["name"] != "Unknown" and f["conf_like"] > 0.55]
                names = sorted({f['name'] for f in known}) if known else []
                shared.last_faces = names
                if names and speak_throttle.allow("faces"):
                    queue_audio_message(
                        audio_q,
                        "faces",
                        " ".join(names),
                        priority=3,
                        is_familiar_face=True,
                    )

        # Distance + vibration + prioritized speech
        d = distance_cm()
        if d >= 0:
            window.append(d)
            med = sorted(window)[len(window)//2] if window else d
            p = obstacle_priority(med)
            if p >= 2 and speak_throttle.allow("distance"):
                queue_audio_message(audio_q, "distance", f"Obstacle at {int(med)} centimeters", priority=p)
            # vibration feedback
            motor_pattern_for_distance(vib, med, stop_evt)

        # Update FPS and adapt inference size
        frame_idx += 1
        dt = time.time() - t_start
        if dt > 0:
            fps_window.append(1.0 / dt)
        if fps_window:
            avg_fps = sum(fps_window) / len(fps_window)
            infer_wh = calculate_adaptive_inference_size(avg_fps, infer_wh)

        # Draw preview
        if not args.headless:
            for d0 in dets:
                bbox = d0.get("bbox_xyxy", [])
                if len(bbox) != 4:
                    log.warning("Invalid bbox for drawing: %s", bbox)
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                if infer_wh:
                    x1 = int(x1 / fx); y1 = int(y1 / fy); x2 = int(x2 / fx); y2 = int(y2 / fy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{d0['class_name']} {d0['conf']:.2f}", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.imshow("VisionAid", frame)
            key = cv2.waitKey(max(1, args.delay)) & 0xFF
            if key == ord('q'):
                stop_evt.set()
            elif key == ord('c'):
                shared.currency_mode_until = time.time() + CURRENCY_MODE_TIMEOUT
                queue_audio_message(audio_q, "vc_curr", "Currency mode", priority=2)

    # Cleanup
    try: cam.close()
    except Exception: pass
    try: cv2.destroyAllWindows()
    except Exception: pass


def main():
    args, logger = parse_config()
    for name in ("ultralytics", "numexpr", "PIL", "matplotlib"):
        logging.getLogger(name).setLevel(logging.WARNING)

    # Handle face registration/listing quick actions and exit
    reg_rc = registration_flow(args)
    if reg_rc >= 0:  # 0 or 2 are terminal for registration flows
        sys.exit(reg_rc)

    # Single-image path (prints all detections)
    if args.image:
        try:
            model = load_yolo_model(args.model)
            dets = run_single_image(model, args.image, args.thresh)
            if not dets:
                print("No detections above threshold.")
            else:
                for d in dets:
                    print(f"{d['class_name']}: {d['conf']:.2f}  bbox={d['bbox_xyxy']}")
            return
        except Exception as e:
            logger.error("Image inference error: %s", e)
            sys.exit(2)

    if not safe_setup_gpio():
        sys.exit(1)

    class SharedState:
        def __init__(self):
            self.last_faces = []
            self.last_objects = []
            self.currency_mode_until = 0.0

    shared = SharedState()

    VOICE_COMMANDS = [
        "help", "mute", "speak", "all", "less", "faces",
        "scan", "distance", "save face", "stop", "currency", "emergency",
    ]

    # Audio (with graceful shutdown/drain)
    audio_q = PriorityMsgQueue()
    speech = SpeechEngine(volume=args.audio_volume)
    stop_evt = threading.Event()
    t_audio = threading.Thread(target=audio_thread_func, args=(stop_evt, audio_q, speech), daemon=True)
    t_audio.start()

    # Thread orchestration
    from .threads import ThreadManager
    tm = ThreadManager()
    tm.spawn("live", live_loop, args, audio_q, speech, shared)

    main_stop = threading.Event()

    def handle_voice_command(cmd: str):
        logger.info("Voice command: %s", cmd)
        if cmd == "help":
            cmds = ", ".join(VOICE_COMMANDS)
            queue_audio_message(audio_q, "vc_help", f"Commands: {cmds}", force=True)
        elif cmd == "mute":
            set_audio_enabled(False)
            queue_audio_message(audio_q, "vc_mute", "Muted", force=True)
        elif cmd == "speak":
            set_audio_enabled(True)
            queue_audio_message(audio_q, "vc_speak", "Audio on", force=True)
        elif cmd == "all":
            args.alert_all = True
            queue_audio_message(audio_q, "vc_all", "Announcing all objects", force=True)
        elif cmd == "less":
            args.alert_all = False
            queue_audio_message(audio_q, "vc_less", "Announcing fewer objects", force=True)
        elif cmd == "faces":
            names = shared.last_faces
            msg = "No faces" if not names else ", ".join(names)
            queue_audio_message(audio_q, "vc_faces", msg, force=True)
        elif cmd == "scan":
            objs = shared.last_objects
            msg = "No objects" if not objs else ", ".join(objs[:5])
            queue_audio_message(audio_q, "vc_scan", msg, force=True)
        elif cmd == "distance":
            d = distance_cm()
            if d >= 0:
                queue_audio_message(audio_q, "vc_distance", f"{int(d)} centimeters", force=True)
        elif cmd == "save_face":
            queue_audio_message(audio_q, "vc_save", "Face saving not implemented", force=True)
        elif cmd == "currency":
            shared.currency_mode_until = time.time() + CURRENCY_MODE_TIMEOUT
            queue_audio_message(audio_q, "vc_curr", "Currency mode", force=True)
        elif cmd == "stop":
            queue_audio_message(audio_q, "vc_stop", "Shutting down", priority=4, force=True)
            main_stop.set()
        elif cmd == "emergency":
            queue_audio_message(audio_q, "vc_emerg", "Emergency triggered", priority=4, force=True)
            try:
                loc = get_gps_location()
                msg = f"Location: {loc[0]}, {loc[1]}" if loc else "Location unknown"
                send_email("VisionAid Emergency", msg)
            except Exception as e:
                logger.error("Emergency email failed: %s", e)

    if args.voice_commands:
        from .voice import voice_command_loop
        from .audio import set_audio_enabled
        tm.spawn("voice", voice_command_loop, args, audio_q, handle_voice_command)

    # Optional GPS/email thread (still stubbed)
    def gps_mail_loop(stop_evt):
        while not stop_evt.is_set():
            time.sleep(60)
            loc = get_gps_location()
            if loc:
                send_email("VisionAid GPS", f"Location: {loc[0]}, {loc[1]}")

    tm.spawn("gps_mail", gps_mail_loop)

    try:
        while not main_stop.is_set():
            time.sleep(0.3)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user.")
    finally:
        # stop worker threads
        tm.stop_all()
        tm.join_all(timeout_per=3.0)
        # stop audio (then it drains)
        stop_evt.set()
        try:
            # leave the queue to drain in audio thread for a bit
            time.sleep(0.8)
        except Exception:
            pass
        # GPIO cleanup
        try:
            GPIO.output(MOTOR_PIN, False)
            GPIO.cleanup()
        except Exception:
            pass


if __name__ == "__main__":
    main()
