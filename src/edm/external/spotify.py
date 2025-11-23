"""Spotify API integration for track metadata and BPM lookup."""

import logging
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpotifyTrackInfo:
    """Track information from Spotify.

    Attributes
    ----------
    id : str
        Spotify track ID.
    title : str
        Track title.
    artist : str
        Artist name.
    album : str
        Album name.
    bpm : Optional[float]
        BPM from audio features.
    key : Optional[int]
        Musical key (0-11, Pitch Class notation).
    mode : Optional[int]
        Mode (0 = minor, 1 = major).
    energy : Optional[float]
        Energy value (0-1).
    danceability : Optional[float]
        Danceability value (0-1).
    """
    id: str
    title: str
    artist: str
    album: str
    bpm: Optional[float] = None
    key: Optional[int] = None
    mode: Optional[int] = None
    energy: Optional[float] = None
    danceability: Optional[float] = None


class SpotifyClient:
    """Client for Spotify API with caching."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        """Initialize Spotify client.

        Parameters
        ----------
        client_id : Optional[str]
            Spotify API client ID. If None, reads from SPOTIFY_CLIENT_ID env var.
        client_secret : Optional[str]
            Spotify API client secret. If None, reads from SPOTIFY_CLIENT_SECRET env var.
        cache_dir : Optional[Path]
            Directory for caching API responses. If None, uses ~/.cache/edm/spotify.
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

        self._token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._sp_client = None

    def _get_client(self):
        """Get or create spotipy client with authentication.

        Returns
        -------
        spotipy.Spotify
            Authenticated Spotify client.

        Raises
        ------
        ImportError
            If spotipy is not installed.
        ValueError
            If credentials are not configured.
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
                )
            )

            self._sp_client = spotipy.Spotify(auth_manager=auth_manager)

            # Token typically expires in 3600 seconds, but refresh a bit early
            self._token_expires_at = time.time() + 3500

            logger.debug("Successfully authenticated with Spotify API")
            return self._sp_client

        except ImportError:
            raise ImportError(
                "spotipy is required for Spotify integration. "
                "Install with: pip install spotipy"
            )

    @lru_cache(maxsize=512)
    def search_track(self, artist: str, title: str) -> Optional[SpotifyTrackInfo]:
        """Search for a track on Spotify.

        Parameters
        ----------
        artist : str
            Artist name.
        title : str
            Track title.

        Returns
        -------
        SpotifyTrackInfo or None
            Track information if found, None otherwise.

        Raises
        ------
        ExternalServiceError
            If the API request fails.
        """
        logger.info(f"Searching Spotify for: {artist} - {title}")

        try:
            sp = self._get_client()

            # Build search query
            query = f"artist:{artist} track:{title}"
            logger.debug(f"Spotify search query: {query}")

            # Search for track
            results = sp.search(q=query, type="track", limit=1)

            if not results or not results.get("tracks", {}).get("items"):
                logger.debug(f"No Spotify results for: {artist} - {title}")
                return None

            # Get first result
            track_data = results["tracks"]["items"][0]
            track_id = track_data["id"]

            # Build base track info
            track = SpotifyTrackInfo(
                id=track_id,
                title=track_data["name"],
                artist=track_data["artists"][0]["name"] if track_data["artists"] else "",
                album=track_data["album"]["name"] if track_data.get("album") else ""
            )

            # Get audio features
            features = self.get_audio_features(track_id)
            if features:
                track.bpm = features.get("tempo")
                track.key = features.get("key")
                track.mode = features.get("mode")
                track.energy = features.get("energy")
                track.danceability = features.get("danceability")

            logger.info(f"Found on Spotify: {track.artist} - {track.title}")
            return track

        except ValueError as e:
            logger.error(f"Spotify client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Spotify search failed: {e}")
            return None

    @lru_cache(maxsize=512)
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """Get audio features for a track including BPM.

        Parameters
        ----------
        track_id : str
            Spotify track ID.

        Returns
        -------
        Optional[Dict]
            Audio features dictionary with keys like 'tempo', 'key', 'mode', etc.
            Returns None if features cannot be retrieved.
        """
        try:
            sp = self._get_client()

            logger.debug(f"Fetching audio features for track ID: {track_id}")
            features = sp.audio_features(track_id)

            if features and len(features) > 0:
                return features[0]
            else:
                logger.debug(f"No audio features found for track ID: {track_id}")
                return None

        except ValueError as e:
            logger.error(f"Spotify client error: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to get audio features: {e}")
            return None

    def clear_cache(self):
        """Clear the LRU cache and delete cached files."""
        self.search_track.cache_clear()
        self.get_audio_features.cache_clear()

        # Remove cache files
        cache_file = self.cache_dir / ".spotify_cache"
        if cache_file.exists():
            cache_file.unlink()
            logger.info("Cleared Spotify cache")


def get_bpm_from_spotify(artist: str, title: str) -> Optional[float]:
    """Convenience function to get BPM from Spotify.

    Parameters
    ----------
    artist : str
        Artist name.
    title : str
        Track title.

    Returns
    -------
    Optional[float]
        BPM if found, None otherwise.
    """
    client = SpotifyClient()
    track = client.search_track(artist, title)
    return track.bpm if track else None

