"""Thread safety tests for AudioCache."""

import threading
from pathlib import Path

import numpy as np

from edm.io.audio import AudioCache


def test_concurrent_cache_put():
    """Test that concurrent put operations don't corrupt cache."""
    cache = AudioCache(max_size=20)
    errors = []

    def worker(thread_id: int):
        try:
            for i in range(50):
                key_path = Path(f"/fake/audio_{thread_id}_{i}.wav")
                audio = np.random.randn(44100)
                cache.put(key_path, 44100, (audio, 44100))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should complete without errors
    assert len(errors) == 0

    # Cache should be consistent
    stats = cache.stats()
    assert stats["size"] <= cache.max_size


def test_concurrent_cache_get():
    """Test that concurrent get operations don't corrupt cache."""
    cache = AudioCache(max_size=10)

    # Pre-populate cache
    for i in range(10):
        key_path = Path(f"/fake/audio_{i}.wav")
        audio = np.random.randn(44100)
        cache.put(key_path, 44100, (audio, 44100))

    errors = []
    results = []

    def worker(thread_id: int):
        try:
            for i in range(100):
                key_path = Path(f"/fake/audio_{i % 10}.wav")
                result = cache.get(key_path, 44100)
                results.append(result is not None)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should complete without errors
    assert len(errors) == 0

    # All gets should have succeeded
    assert all(results)


def test_concurrent_mixed_operations():
    """Test concurrent mix of get/put operations."""
    cache = AudioCache(max_size=10)
    errors = []

    def reader(thread_id: int):
        try:
            for i in range(50):
                key_path = Path(f"/fake/audio_{i % 5}.wav")
                cache.get(key_path, 44100)
        except Exception as e:
            errors.append(e)

    def writer(thread_id: int):
        try:
            for i in range(50):
                key_path = Path(f"/fake/audio_{i % 5}.wav")
                audio = np.random.randn(44100)
                cache.put(key_path, 44100, (audio, 44100))
        except Exception as e:
            errors.append(e)

    threads = []
    for i in range(5):
        threads.append(threading.Thread(target=reader, args=(i,)))
        threads.append(threading.Thread(target=writer, args=(i,)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should complete without errors
    assert len(errors) == 0

    # Cache should be consistent
    stats = cache.stats()
    assert stats["size"] <= cache.max_size


def test_cache_size_consistency_under_load():
    """Test that cache never exceeds max_size under concurrent load."""
    cache = AudioCache(max_size=5)
    errors = []

    def worker(thread_id: int):
        try:
            for i in range(20):
                key_path = Path(f"/fake/audio_{thread_id}_{i}.wav")
                audio = np.random.randn(44100)
                cache.put(key_path, 44100, (audio, 44100))

                # Check size never exceeds max
                stats = cache.stats()
                if stats["size"] > cache.max_size:
                    errors.append(f"Cache size {stats['size']} exceeded max {cache.max_size}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Should have no size violations
    assert len(errors) == 0

    # Final size should be at most max_size
    stats = cache.stats()
    assert stats["size"] <= cache.max_size
