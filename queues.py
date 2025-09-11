import threading
from queue import PriorityQueue, Full, Empty

MAX_AUDIO_QUEUE_SIZE = 20

class PriorityMsgQueue:
    def __init__(self, maxsize=MAX_AUDIO_QUEUE_SIZE):
        self._q = PriorityQueue(maxsize=maxsize)
        self._counter = 0
        self._lock = threading.Lock()

    def put(self, message: str, priority: int):
        # lower numeric value => higher priority; invert if you prefer
        with self._lock:
            item = (-(priority or 0), self._counter, message)
            self._counter += 1
        self._q.put(item, block=False)

    def get(self, timeout=None):
        try:
            _, _, msg = self._q.get(timeout=timeout)
            return msg
        except Empty:
            raise

    def task_done(self):
        self._q.task_done()

    def qsize(self) -> int:
        return self._q.qsize()

    def empty(self) -> bool:
        return self._q.empty()

    def try_put(self, message: str, priority: int) -> bool:
        try:
            self.put(message, priority)
            return True
        except Full:
            return False
