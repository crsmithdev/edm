"""Audio file loading with LRU caching."""

from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import structlog

logger = structlog.get_logger(__name__)

# Type alias for audio data
AudioData = tuple[np.ndarray, int]


class AudioCache:
    """LRU cache for loaded audio data.

    Caches decoded audio to avoid redundant file I/O and decoding operations
    during analysis. Uses an ordered dict for LRU eviction.

    Attributes:
        max_size: Maximum number of audio files to cache.
    """

    def __init__(self, max_size: int = 10):
        """Initialize the audio cache.

        Args:
            max_size: Maximum number of audio files to cache. Set to 0 to disable caching.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, AudioData] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, filepath: Path, sr: int | None = None) -> AudioData | None:
        """Get audio data from cache.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate (used as part of cache key).

        Returns:
            Cached audio data as (y, sr) tuple, or None if not cached.
        """
        if self.max_size == 0:
            return None

        key = self._make_key(filepath, sr)
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            logger.debug("audio cache hit", filepath=str(filepath), sr=sr)
            return self._cache[key]

        self._misses += 1
        return None

    def put(self, filepath: Path, sr: int | None, audio_data: AudioData) -> None:
        """Store audio data in cache.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate (used as part of cache key).
            audio_data: Audio data as (y, sr) tuple.
        """
        if self.max_size == 0:
            return

        key = self._make_key(filepath, sr)

        # Evict LRU entry if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            evicted_key, _ = self._cache.popitem(last=False)
            logger.debug("audio cache eviction", evicted_key=evicted_key)

        self._cache[key] = audio_data
        self._cache.move_to_end(key)
        logger.debug("audio cached", filepath=str(filepath), sr=sr)

    def clear(self) -> None:
        """Clear all cached audio data."""
        count = len(self._cache)
        self._cache.clear()
        logger.debug("audio cache cleared", entries_cleared=count)

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, size, and hit rate.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 2),
        }

    def _make_key(self, filepath: Path, sr: int | None) -> str:
        """Create cache key from filepath and sample rate."""
        return f"{filepath.resolve()}:{sr}"


# Global cache instance
_audio_cache: AudioCache | None = None


def get_audio_cache(max_size: int = 10) -> AudioCache:
    """Get or create the global audio cache.

    Args:
        max_size: Maximum cache size (only used on first call).

    Returns:
        Global AudioCache instance.
    """
    global _audio_cache
    if _audio_cache is None:
        _audio_cache = AudioCache(max_size=max_size)
    return _audio_cache


def set_cache_size(max_size: int) -> None:
    """Set the cache size, creating a new cache if needed.

    Args:
        max_size: Maximum number of audio files to cache.
    """
    global _audio_cache
    _audio_cache = AudioCache(max_size=max_size)


def clear_audio_cache() -> None:
    """Clear the global audio cache."""
    global _audio_cache
    if _audio_cache is not None:
        _audio_cache.clear()


def load_audio(
    filepath: Path,
    sr: int | None = None,
    cache: AudioCache | None = None,
) -> AudioData:
    """Load audio file with optional caching.

    Args:
        filepath: Path to the audio file.
        sr: Target sample rate. If None, uses native sample rate.
        cache: AudioCache instance. If None, uses global cache.

    Returns:
        Tuple of (audio samples, sample rate).

    Raises:
        AudioFileError: If the file cannot be loaded.
    """
    if cache is None:
        cache = get_audio_cache()

    # Try cache first
    cached = cache.get(filepath, sr)
    if cached is not None:
        return cached

    # Load from disk
    logger.debug("loading audio from disk", filepath=str(filepath), sr=sr)
    try:
        y, actual_sr = librosa.load(str(filepath), sr=sr)
        audio_data = (y, actual_sr)

        # Cache the result
        cache.put(filepath, sr, audio_data)

        return audio_data

    except Exception as e:
        logger.error("failed to load audio", filepath=str(filepath), error=str(e))
        from edm.exceptions import AudioFileError

        raise AudioFileError(f"Failed to load audio file: {filepath}") from e
