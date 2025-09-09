# vision_aid/config.py
import argparse
import logging
from . import settings

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("VisionAid")

    p.add_argument('--model', required=True, help='Path to YOLO model file')
    p.add_argument('--source', default='usb0', help='Camera index (usb0, usb1) or path/RTSP')
    p.add_argument('--thresh', type=float, default=0.5)
    p.add_argument('--resolution', required=True, help='WxH, e.g., 640x480')
    p.add_argument('--inference_size', default=None, help='WxH for speed-up')
    p.add_argument('--delay', type=int, default=1)
    p.add_argument('--everyday-only', action='store_true', default=True)
    p.add_argument('--distance-interval', type=float, default=0.1)
    p.add_argument('--audio', action='store_true', default=True)
    p.add_argument('--audio-volume', type=int, default=80)
    p.add_argument('--headless', action='store_true', default=False)
    p.add_argument('--alert-all', action='store_true', default=False)
    p.add_argument('--frames-required', type=int, default=2)
    p.add_argument('--log-level', default='INFO')
    p.add_argument('--face-recognition', action='store_true', default=True)
    p.add_argument('--register-face', default=None)
    p.add_argument('--list-known-faces', action='store_true', default=False)

    # Voice (unchanged)
    p.add_argument('--voice-commands', action='store_true', default=False)
    p.add_argument('--vosk-model', default='model')
    p.add_argument('--audio-device', type=int, default=5)
    p.add_argument('--bluetooth-card', type=int, default=None)

    # Email/GPS from .env (with CLI override if provided)
    p.add_argument('--smtp-server', default=settings.smtp_server())
    p.add_argument('--smtp-port', type=int, default=settings.smtp_port())
    p.add_argument('--sender-email', default=settings.sender_email())
    p.add_argument('--sender-password', default=settings.sender_password())
    p.add_argument('--emergency-contact', default=settings.emergency_contact())

    # Currency model path from .env
    p.add_argument('--currency-model', default=settings.currency_model())

    # NEW: single-image inference
    p.add_argument('--image', default=None, help='Run prediction on an image and exit')
    return p

def parse_config():
    p = build_parser()
    args = p.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("vision_aid")
    return args, logger
