import threading
import logging
import time
from typing import Callable, List

log = logging.getLogger("vision_aid")

class StoppableThread(threading.Thread):
    def __init__(self, target: Callable, name: str, args=(), kwargs=None, daemon=True):
        super().__init__(target=self._run_wrapper, name=name, daemon=daemon)
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self._stop = threading.Event()
        self._started_evt = threading.Event()

    def _run_wrapper(self):
        self._started_evt.set()
        try:
            self._target(self._stop, *self._args, **self._kwargs)
        except Exception as e:
            log.exception("Thread %s crashed: %s", self.name, e)

    def stop(self):
        self._stop.set()

    def started(self, timeout=2.0) -> bool:
        return self._started_evt.wait(timeout=timeout)


class ThreadManager:
    def __init__(self):
        self._threads: List[StoppableThread] = []

    def spawn(self, name: str, target: Callable, *args, **kwargs) -> StoppableThread:
        t = StoppableThread(target=target, name=name, args=args, kwargs=kwargs)
        t.start()
        if not t.started(timeout=3.0):
            log.warning("Thread %s did not report started within timeout", name)
        self._threads.append(t)
        return t

    def stop_all(self):
        for t in self._threads:
            try: t.stop()
            except Exception: pass

    def join_all(self, timeout_per=3.0):
        for t in self._threads:
            try:
                t.join(timeout=timeout_per)
            except Exception:
                pass
