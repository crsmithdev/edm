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
    """Client for TuneBat data."""

    def __init__(self):
        """Initialize TuneBat client."""
        logger.info("tunebat client initialized")

    def search_track(self, artist: str, title: str) -> TuneBatTrackInfo | None:
        """Search for a track on TuneBat.

        Args:
            artist: Artist name.
            title: Track title.

        Returns:
            Track information if found, None otherwise.

        Raises:
            ExternalServiceError: If the request fails.
        """
        logger.info("searching tunebat", artist=artist, title=title)

        # TODO: Implement TuneBat API or web scraper
        # Placeholder implementation
        return None
