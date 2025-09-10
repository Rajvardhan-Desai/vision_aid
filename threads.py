import threading
import logging
from typing import Callable, List

log = logging.getLogger("vision_aid")


class StoppableThread(threading.Thread):
    """Thread wrapper that injects a stop event as the first arg."""

    def __init__(self, target: Callable, name: str, args=(), kwargs=None, daemon: bool = True):
        self._target_fn = target
        self._target_args = args
        self._target_kwargs = kwargs or {}
        self._stop_evt = threading.Event()
        self._started_evt = threading.Event()
        super().__init__(target=self._run_wrapper, name=name, daemon=daemon)

    def _run_wrapper(self):
        self._started_evt.set()
        try:
            self._target_fn(self._stop_evt, *self._target_args, **self._target_kwargs)
        except Exception as e:
            log.exception("Thread %s crashed: %s", self.name, e)

    def stop(self):
        self._stop_evt.set()

    def started(self, timeout: float = 2.0) -> bool:
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
            try:
                t.stop()
            except Exception:
                pass

    def join_all(self, timeout_per: float = 3.0):
        for t in self._threads:
            try:
                t.join(timeout=timeout_per)
            except Exception:
                pass
