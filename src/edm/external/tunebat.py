"""TuneBat API/scraper client."""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TuneBatTrackInfo:
    """Track information from TuneBat.

    Attributes:
        title: Track title.
        artist: Artist name.
        bpm: BPM value.
        key: Musical key.
    """

    title: str
    artist: str
    bpm: float | None = None
    key: str | None = None


class TuneBatClient:
    """Client for TuneBat data.

    **NOTE:** This is a placeholder implementation. The search_track() method
    always returns None and does not actually query TuneBat.
    """

    def __init__(self):
        """Initialize TuneBat client."""
        logger.debug("tunebat client initialized")

    def search_track(self, artist: str, title: str) -> TuneBatTrackInfo | None:
        """Search for a track on TuneBat.

        **NOTE:** Not yet implemented. This method always returns None.

        Args:
            artist: Artist name.
            title: Track title.

        Returns:
            None (not yet implemented).
        """
        logger.debug("searching tunebat", artist=artist, title=title)

        # TODO: Implement TuneBat API or web scraper
        # Placeholder implementation - always returns None
        return None
