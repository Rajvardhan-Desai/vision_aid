import json
import queue
import logging
from typing import Callable, Optional

try:
    import sounddevice as sd
    from vosk import Model, KaldiRecognizer
except Exception:  # pragma: no cover - optional dependency
    sd = None
    Model = KaldiRecognizer = None

from .audio import queue_audio_message

logger = logging.getLogger("vision_aid")

COMMAND_ALIASES = {
    "help": "help",
    "mute": "mute",
    "speak": "speak",
    "all": "all",
    "less": "less",
    "faces": "faces",
    "scan": "scan",
    "distance": "distance",
    "save face": "save_face",
    "stop": "stop",
    "currency": "currency",
    "emergency": "emergency",
}

CRITICAL_COMMANDS = {"stop", "emergency"}


def _extract_command(text: str) -> Optional[str]:
    for phrase, cmd in COMMAND_ALIASES.items():
        if phrase in text:
            return cmd
    return None


def voice_command_loop(stop_evt, args, audio_q, cmd_handler: Callable[[str], None]):
    if sd is None or Model is None:
        logger.error("sounddevice or vosk not available; voice commands disabled")
        return
    try:
        model = Model(args.vosk_model)
    except Exception as e:  # pragma: no cover - model load failure
        logger.error("Failed to load Vosk model: %s", e)
        return

    recognizer = KaldiRecognizer(model, 16000)
    audio_queue = queue.Queue()

    def callback(indata, frames, time_, status):  # pragma: no cover - hardware interaction
        if status:
            logger.debug("sounddevice status: %s", status)
        audio_queue.put(bytes(indata))

    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, device=args.audio_device,
                               dtype="int16", channels=1, callback=callback):
            listening = False
            awaiting: Optional[str] = None
            while not stop_evt.is_set():
                try:
                    data = audio_queue.get(timeout=0.3)
                except queue.Empty:
                    continue
                if recognizer.AcceptWaveform(data):
                    res = json.loads(recognizer.Result())
                    text = res.get("text", "").strip().lower()
                    if not text:
                        continue
                    if awaiting:
                        if text in {"yes", "yeah", "yep"}:
                            cmd_handler(awaiting)
                        else:
                            queue_audio_message(audio_q, "voice", "Canceled", force=True)
                        awaiting = None
                        listening = False
                        continue
                    if not listening:
                        if "assistant" in text:
                            listening = True
                            queue_audio_message(audio_q, "voice", "Yes?", force=True)
                        continue
                    cmd = _extract_command(text)
                    if not cmd:
                        queue_audio_message(audio_q, "voice", "Command not found", force=True)
                        listening = False
                        continue
                    if cmd in CRITICAL_COMMANDS:
                        awaiting = cmd
                        queue_audio_message(audio_q, "voice", f"Confirm {cmd}?", force=True)
                    else:
                        cmd_handler(cmd)
                        listening = False
                # partial results ignored
    except Exception as e:  # pragma: no cover - hardware interaction
        logger.error("Voice loop error: %s", e)

