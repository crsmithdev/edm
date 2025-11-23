"""External API clients for music services."""

from edm.external.beatport import BeatportClient
from edm.external.spotify import SpotifyClient
from edm.external.tunebat import TuneBatClient

__all__ = ["SpotifyClient", "BeatportClient", "TuneBatClient"]
