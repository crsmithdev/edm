# Design: Fix Cache Race Condition

## Approach

**Option 1 (Recommended)**: Add thread locks to global cache
**Option 2**: Use process-local caches (no shared state)

Choosing Option 1 for simplicity and backward compatibility.

## Implementation

### File: `src/edm/io/audio.py`

**Before:**
```python
class AudioCache:
    def __init__(self, max_size: int = 10):
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._access_order: list[str] = []
        self.max_size = max_size

    def get(self, key: str) -> tuple[np.ndarray, int] | None:
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, audio: np.ndarray, sr: int) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = (audio.copy(), sr)
        self._access_order.append(key)
```

**After:**
```python
import threading

class AudioCache:
    def __init__(self, max_size: int = 10):
        self._cache: dict[str, tuple[np.ndarray, int]] = {}
        self._access_order: list[str] = []
        self.max_size = max_size
        self._lock = threading.Lock()  # ADD LOCK

    def get(self, key: str) -> tuple[np.ndarray, int] | None:
        with self._lock:  # CRITICAL SECTION
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None

    def put(self, key: str, audio: np.ndarray, sr: int) -> None:
        with self._lock:  # CRITICAL SECTION
            if key in self._cache:
                self._access_order.remove(key)
            elif len(self._cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]

            self._cache[key] = (audio.copy(), sr)
            self._access_order.append(key)
```

## Testing

### Concurrency Test

Create `tests/unit/test_audio_cache_threadsafe.py`:

```python
import threading
import numpy as np
from edm.io.audio import AudioCache

def test_concurrent_cache_access():
    """Test that cache is thread-safe under concurrent access."""
    cache = AudioCache(max_size=10)

    def worker(thread_id: int):
        for i in range(100):
            key = f"audio_{i % 5}"
            audio = np.random.randn(44100)
            cache.put(key, audio, 44100)
            result = cache.get(key)
            assert result is not None

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cache should be consistent
    stats = cache.stats()
    assert stats["size"] <= cache.max_size
```

## Risks

- **Performance**: Minimal overhead (microseconds per lock acquisition)
- **Deadlock**: Not possible with single lock
- **Alternative**: Process-local caches if performance critical
