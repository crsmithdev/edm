"""Spotify API integration for track metadata and BPM lookup."""

import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SpotifyTrackInfo:
    """Track information from Spotify.

    Attributes:
        id: Spotify track ID.
        title: Track title.
        artist: Artist name.
        album: Album name.
        bpm: BPM from audio features.
        key: Musical key (0-11, Pitch Class notation).
        mode: Mode (0 = minor, 1 = major).
        energy: Energy value (0-1).
        danceability: Danceability value (0-1).
    """

    id: str
    title: str
    artist: str
    album: str
    bpm: float | None = None
    key: int | None = None
    mode: int | None = None
    energy: float | None = None
    danceability: float | None = None


class SpotifyClient:
    """Client for Spotify API with caching."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: Path | None = None,
    ):
        """Initialize Spotify client.

        Args:
            client_id: Spotify API client ID. If None, reads from SPOTIFY_CLIENT_ID env var.
            client_secret: Spotify API client secret. If None, reads from SPOTIFY_CLIENT_SECRET env var.
            cache_dir: Directory for caching API responses. If None, uses ~/.cache/edm/spotify.
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Spotify credentials not found. Set SPOTIFY_CLIENT_ID and "
                "SPOTIFY_CLIENT_SECRET environment variables."
            )

        self.cache_dir = cache_dir or Path.home() / ".cache" / "edm" / "spotify"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._token: str | None = None
        self._token_expires_at: float = 0.0
        self._sp_client = None

    def _get_client(self):
        """Get or create spotipy client with authentication.

        Returns:
            Authenticated Spotify client.

        Raises:
            ImportError: If spotipy is not installed.
            ValueError: If credentials are not configured.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not configured. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
            )

        # Check if token is still valid
        if self._sp_client and time.time() < self._token_expires_at:
            return self._sp_client

        # Create new authenticated client
        try:
            import spotipy
            from spotipy.oauth2 import SpotifyClientCredentials

            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret,
                cache_handler=spotipy.cache_handler.CacheFileHandler(
                    cache_path=str(self.cache_dir / ".spotify_cache")
                ),
            )

            self._sp_client = spotipy.Spotify(auth_manager=auth_manager)

            # Token typically expires in 3600 seconds, but refresh a bit early
            self._token_expires_at = time.time() + 3500

            logger.debug("successfully authenticated with spotify api")
            return self._sp_client

        except ImportError:
            raise ImportError(
                "spotipy is required for Spotify integration. Install with: pip install spotipy"
            )

    @lru_cache(maxsize=512)
    def search_track(self, artist: str, title: str) -> SpotifyTrackInfo | None:
        """Search for a track on Spotify.

        Args:
            artist: Artist name.
            title: Track title.

        Returns:
            Track information if found, None otherwise.
        """
        logger.debug("searching spotify", artist=artist, title=title)

        try:
            sp = self._get_client()

            # Build search query
            query = f"artist:{artist} track:{title}"
            logger.debug("spotify search query", query=query)

            # Search for track
            results = sp.search(q=query, type="track", limit=1)

            if not results or not results.get("tracks", {}).get("items"):
                logger.debug("no spotify results", artist=artist, title=title)
                return None

            # Get first result
            track_data = results["tracks"]["items"][0]
            track_id = track_data["id"]

            # Build base track info
            track = SpotifyTrackInfo(
                id=track_id,
                title=track_data["name"],
                artist=track_data["artists"][0]["name"] if track_data["artists"] else "",
                album=track_data["album"]["name"] if track_data.get("album") else "",
            )

            # Get audio features
            features = self.get_audio_features(track_id)
            if features:
                track.bpm = features.get("tempo")
                track.key = features.get("key")
                track.mode = features.get("mode")
                track.energy = features.get("energy")
                track.danceability = features.get("danceability")

            logger.debug("found on spotify", artist=track.artist, title=track.title)
            return track

        except ValueError as e:
            logger.error("spotify client error", error=str(e))
            return None
        except Exception as e:
            logger.error("spotify search failed", error=str(e))
            return None

    @lru_cache(maxsize=512)
    def get_audio_features(self, track_id: str) -> dict | None:
        """Get audio features for a track including BPM.

        Args:
            track_id: Spotify track ID.

        Returns:
            Audio features dictionary with keys like 'tempo', 'key', 'mode', etc.
            Returns None if features cannot be retrieved.
        """
        try:
            sp = self._get_client()

            logger.debug("fetching audio features", track_id=track_id)
            features = sp.audio_features(track_id)

            if features and len(features) > 0:
                return features[0]
            else:
                logger.debug("no audio features found", track_id=track_id)
                return None

        except ValueError as e:
            logger.error("spotify client error", error=str(e))
            return None
        except Exception as e:
            logger.error("failed to get audio features", error=str(e))
            return None
