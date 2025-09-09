import subprocess, threading, time, logging
from typing import Dict
from .queues import PriorityMsgQueue, MAX_AUDIO_QUEUE_SIZE

logger = logging.getLogger("vision_aid")

AUDIO_COOLDOWN = 3.0
FAMILIAR_FACE_COOLDOWN = 10.0
MAX_CONCURRENT_SPEECH = 3

class SpeechEngine:
    def __init__(self, volume: int = 80):
        self.volume = max(0, min(100, volume))
        self._procs = []
        self._timers: Dict[int, threading.Timer] = {}
        self._lock = threading.Lock()

    def _cleanup_procs(self):
        with self._lock:
            alive = []
            for p in self._procs:
                if p.poll() is None:
                    alive.append(p)
                else:
                    tid = id(p)
                    t = self._timers.pop(tid, None)
                    if t:
                        try: t.cancel()
                        except Exception: pass
            self._procs = alive

    def speak(self, message: str):
        self._cleanup_procs()
        with self._lock:
            if len(self._procs) >= MAX_CONCURRENT_SPEECH:
                logger.debug("Too many TTS processes; drop: %s", message)
                return
            try:
                p = subprocess.Popen(
                    ['espeak', '-a', str(self.volume), '-s', '150', f'"{message}"'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                self._procs.append(p)

                def kill_if_running():
                    if p.poll() is None:
                        try: p.kill()
                        except Exception: pass

                timer = threading.Timer(10.0, kill_if_running)
                timer.daemon = True
                timer.start()
                self._timers[id(p)] = timer

                def waiter():
                    p.wait()
                    with self._lock:
                        t = self._timers.pop(id(p), None)
                        if t:
                            try: t.cancel()
                            except Exception: pass
                threading.Thread(target=waiter, daemon=True).start()
            except Exception as e:
                logger.error("TTS error: %s", e)

_last_audio_time: Dict[str, float] = {}

def queue_audio_message(q: PriorityMsgQueue, class_key, message: str,
                        priority: int = 1, is_familiar_face: bool = False) -> bool:
    import time
    global _last_audio_time
    now = time.time()
    msg_id = f"face_{class_key}" if is_familiar_face else str(class_key)
    cooldown = FAMILIAR_FACE_COOLDOWN if is_familiar_face else AUDIO_COOLDOWN
    last = _last_audio_time.get(msg_id, 0)
    if now - last < cooldown:
        return False
    if q.qsize() >= MAX_AUDIO_QUEUE_SIZE and priority < 2 and not is_familiar_face:
        return False
    try:
        q.put(message, priority=priority)
        _last_audio_time[msg_id] = now
        return True
    except Exception as e:
        logger.error("Audio queue put failed: %s", e)
        return False

def audio_thread_func(stop_evt, q: PriorityMsgQueue, speech: SpeechEngine):
    """
    Reads from the priority queue until stop is set; then drains remaining items and exits.
    """
    while not stop_evt.is_set():
        try:
            msg = q.get(timeout=0.3)
        except Exception:
            continue
        try:
            speech.speak(msg)
            time.sleep(min(2.0, 0.08 * max(1, len(msg))))
        finally:
            q.task_done()

    # Drain gracefully
    try:
        while not q.empty():
            msg = q.get(timeout=0.1)
            try:
                speech.speak(msg)
                time.sleep(min(1.5, 0.06 * max(1, len(msg))))
            finally:
                q.task_done()
    except Exception:
        pass
