"""Tests for audio caching functionality."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from edm.io.audio import (
    AudioCache,
    clear_audio_cache,
    get_audio_cache,
    load_audio,
    set_cache_size,
)


class TestAudioCache:
    """Tests for AudioCache class."""

    def test_cache_init(self):
        """Test cache initialization."""
        cache = AudioCache(max_size=5)
        assert cache.max_size == 5
        assert cache._hits == 0
        assert cache._misses == 0

    def test_cache_put_and_get(self):
        """Test basic put and get operations."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        # Put data
        cache.put(filepath, None, audio_data)

        # Get data
        result = cache.get(filepath, None)
        assert result is not None
        np.testing.assert_array_equal(result[0], audio_data[0])
        assert result[1] == audio_data[1]

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")

        result = cache.get(filepath, None)
        assert result is None
        assert cache._misses == 1

    def test_cache_hit_increments_counter(self):
        """Test cache hit increments hit counter."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        cache.put(filepath, None, audio_data)
        cache.get(filepath, None)

        assert cache._hits == 1

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = AudioCache(max_size=2)

        # Add two items
        path1 = Path("/test/audio1.mp3")
        path2 = Path("/test/audio2.mp3")
        path3 = Path("/test/audio3.mp3")

        cache.put(path1, None, (np.array([1.0]), 44100))
        cache.put(path2, None, (np.array([2.0]), 44100))

        # Access path1 to make it recently used
        cache.get(path1, None)

        # Add path3 - should evict path2 (least recently used)
        cache.put(path3, None, (np.array([3.0]), 44100))

        # path1 should still be cached
        assert cache.get(path1, None) is not None
        # path2 should be evicted
        assert cache.get(path2, None) is None
        # path3 should be cached
        assert cache.get(path3, None) is not None

    def test_cache_disabled_when_size_zero(self):
        """Test cache is disabled when max_size is 0."""
        cache = AudioCache(max_size=0)
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        # Put should do nothing
        cache.put(filepath, None, audio_data)

        # Get should return None
        result = cache.get(filepath, None)
        assert result is None

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        cache.put(filepath, None, audio_data)
        assert cache.get(filepath, None) is not None

        cache.clear()
        assert cache.get(filepath, None) is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        # Miss
        cache.get(filepath, None)
        # Put and hit
        cache.put(filepath, None, audio_data)
        cache.get(filepath, None)
        cache.get(filepath, None)

        stats = cache.stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["size"] == 1
        assert stats["max_size"] == 5
        assert stats["hit_rate"] == 0.67  # 2 / 3

    def test_cache_key_includes_sample_rate(self):
        """Test that cache key includes sample rate."""
        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")
        audio_data_44k = (np.array([1.0, 2.0]), 44100)
        audio_data_22k = (np.array([1.0]), 22050)

        # Store with different sample rates
        cache.put(filepath, 44100, audio_data_44k)
        cache.put(filepath, 22050, audio_data_22k)

        # Retrieve with specific sample rates
        result_44k = cache.get(filepath, 44100)
        result_22k = cache.get(filepath, 22050)

        assert result_44k is not None
        assert result_22k is not None
        assert len(result_44k[0]) == 2
        assert len(result_22k[0]) == 1


class TestLoadAudio:
    """Tests for load_audio function."""

    @patch("edm.io.audio.librosa.load")
    def test_load_audio_caches_result(self, mock_load):
        """Test that load_audio caches the result."""
        mock_load.return_value = (np.array([1.0, 2.0, 3.0]), 44100)

        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")

        # First load
        result1 = load_audio(filepath, cache=cache)
        assert mock_load.call_count == 1

        # Second load should use cache
        result2 = load_audio(filepath, cache=cache)
        assert mock_load.call_count == 1  # Not called again

        np.testing.assert_array_equal(result1[0], result2[0])

    @patch("edm.io.audio.librosa.load")
    def test_load_audio_with_sample_rate(self, mock_load):
        """Test load_audio with specific sample rate."""
        mock_load.return_value = (np.array([1.0, 2.0, 3.0]), 22050)

        cache = AudioCache(max_size=5)
        filepath = Path("/test/audio.mp3")

        result = load_audio(filepath, sr=22050, cache=cache)

        mock_load.assert_called_once_with(str(filepath), sr=22050)
        assert result[1] == 22050


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_set_cache_size(self):
        """Test setting cache size creates new cache."""
        set_cache_size(20)
        cache = get_audio_cache()
        assert cache.max_size == 20

    def test_clear_audio_cache(self):
        """Test clearing global cache."""
        set_cache_size(5)
        cache = get_audio_cache()
        filepath = Path("/test/audio.mp3")
        audio_data = (np.array([1.0, 2.0, 3.0]), 44100)

        cache.put(filepath, None, audio_data)
        assert cache.get(filepath, None) is not None

        clear_audio_cache()
        assert cache.get(filepath, None) is None
