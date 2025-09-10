import threading
import logging
from typing import Callable, List, Tuple, Optional, Dict, Any

log = logging.getLogger("vision_aid")


class StoppableThread(threading.Thread):
    """
    Thread wrapper that injects a threading.Event ('stop_event') as the first argument
    to the target function.

    Target signature must be:  target(stop_event: threading.Event, *args, **kwargs)
    """

    def __init__(
        self,
        target: Callable,
        name: str,
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        daemon: bool = True,
    ):
        self._target_fn = target
        self._target_args = args
        self._target_kwargs = kwargs or {}
        self._stop_event = threading.Event()
        self._started_evt = threading.Event()
        super().__init__(target=self._run_wrapper, name=name, daemon=daemon)

    def _run_wrapper(self) -> None:
        # Signal that the thread object has begun running user code
        self._started_evt.set()
        try:
            self._target_fn(self._stop_event, *self._target_args, **self._target_kwargs)
        except Exception as e:
            # Never let exceptions kill the whole process silently
            log.exception("Thread %s crashed: %s", self.name, e)

    def stop(self) -> None:
        """Request the thread to stop; the target should return soon after."""
        self._stop_event.set()

    def started(self, timeout: float = 2.0) -> bool:
        """Wait until the internal run wrapper starts (not the target's internal readiness)."""
        return self._started_evt.wait(timeout=timeout)

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event


class ThreadManager:
    """Helper to spawn, stop, and join multiple StoppableThread instances."""

    def __init__(self):
        self._threads: List[StoppableThread] = []

    def spawn(self, name: str, target: Callable, *args, **kwargs) -> StoppableThread:
        t = StoppableThread(target=target, name=name, args=args, kwargs=kwargs)
        t.start()
        if not t.started(timeout=3.0):
            log.warning("Thread %s did not report started within timeout", name)
        self._threads.append(t)
        return t

    def stop_all(self) -> None:
        for t in self._threads:
            try:
                t.stop()
            except Exception as e:
                log.exception("Failed to stop thread %s: %s", t.name, e)

    def join_all(self, timeout_per: float = 3.0) -> None:
        for t in self._threads:
            try:
                t.join(timeout=timeout_per)
                if t.is_alive():
                    log.warning("Thread %s did not join within %.1fs", t.name, timeout_per)
            except Exception as e:
                log.exception("Failed to join thread %s: %s", t.name, e)
