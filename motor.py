import time, logging
from .gpio_shim import GPIO

log = logging.getLogger("vision_aid")

class VibrationController:
    def __init__(self, motor_pin: int):
        self.pin = motor_pin
        GPIO.setup(self.pin, GPIO.OUT)
        GPIO.output(self.pin, False)

    def _pulse(self, on_ms: int, off_ms: int, cycles: int = 1, stop_evt=None):
        for _ in range(cycles):
            if stop_evt and stop_evt.is_set(): break
            GPIO.output(self.pin, True)
            time.sleep(on_ms/1000.0)
            GPIO.output(self.pin, False)
            time.sleep(off_ms/1000.0)

    def pattern_gentle(self, stop_evt=None):
        self._pulse(60, 200, cycles=2, stop_evt=stop_evt)

    def pattern_warning(self, stop_evt=None):
        self._pulse(120, 150, cycles=3, stop_evt=stop_evt)

    def pattern_urgent(self, stop_evt=None):
        self._pulse(200, 100, cycles=4, stop_evt=stop_evt)
