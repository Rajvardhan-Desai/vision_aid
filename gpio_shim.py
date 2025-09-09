import time
import logging

logger = logging.getLogger("vision_aid")

class _SimGPIO:
    BCM = "BCM"
    OUT = "OUT"
    IN  = "IN"
    LOW = 0
    HIGH = 1

    def __init__(self):
        self._pins = {}
        logger.warning("Using Simulated GPIO (non-RPi environment)")

    def setmode(self, *args, **kwargs): pass
    def setwarnings(self, flag): pass
    def setup(self, pin, mode): self._pins[pin] = self.LOW
    def output(self, pin, val): self._pins[pin] = val
    def input(self, pin): return self._pins.get(pin, self.LOW)
    def cleanup(self): self._pins.clear()

try:
    import RPi.GPIO as _GPIO
    GPIO = _GPIO
except Exception:
    GPIO = _SimGPIO()

def is_simulated() -> bool:
    return isinstance(GPIO, _SimGPIO)

# Tiny utility for timing in distance functions, safe on both envs
def sleep(sec: float):
    time.sleep(sec)
