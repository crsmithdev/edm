"""Security tests for XML External Entity (XXE) prevention."""

import tempfile
from pathlib import Path

import pytest
from defusedxml.ElementTree import ParseError

from edm.data.rekordbox import parse_rekordbox_xml


def test_xxe_attack_file_inclusion():
    """Test that XXE attacks attempting file inclusion are prevented."""
    malicious_xml = """<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="&xxe;" Artist="Attacker" TotalTime="180"
           AverageBpm="128.00" SampleRate="44100" Location="file://test.mp3">
    </TRACK>
  </COLLECTION>
</DJ_PLAYLISTS>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(malicious_xml)
        xml_path = Path(f.name)

    try:
        # defusedxml should reject this XML with external entities
        with pytest.raises((ParseError, ValueError)):
            parse_rekordbox_xml(xml_path)
    finally:
        xml_path.unlink()


def test_xxe_attack_billion_laughs():
    """Test that billion laughs DoS attack is prevented."""
    malicious_xml = """<?xml version="1.0"?>
<!DOCTYPE lolz [
  <!ENTITY lol "lol">
  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">
  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">
]>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="&lol3;" Artist="Attacker" TotalTime="180"
           AverageBpm="128.00" SampleRate="44100" Location="file://test.mp3">
    </TRACK>
  </COLLECTION>
</DJ_PLAYLISTS>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(malicious_xml)
        xml_path = Path(f.name)

    try:
        # defusedxml should reject this XML with entity expansion
        with pytest.raises((ParseError, ValueError)):
            parse_rekordbox_xml(xml_path)
    finally:
        xml_path.unlink()


def test_valid_rekordbox_xml_still_works():
    """Test that valid Rekordbox XML still parses correctly."""
    valid_xml = """<?xml version="1.0" encoding="UTF-8"?>
<DJ_PLAYLISTS Version="1.0.0">
  <PRODUCT Name="rekordbox" Version="6.0" Company="Pioneer DJ"/>
  <COLLECTION Entries="1">
    <TRACK TrackID="1" Name="Test Track" Artist="Test Artist"
           TotalTime="180" AverageBpm="128.00" SampleRate="44100"
           Location="file://localhost/Music/test.mp3">
      <POSITION_MARK Name="Cue" Type="0" Start="10.0" Num="0"/>
      <TEMPO Inizio="0.000" Bpm="128.00" Metro="4/4" Battito="1"/>
    </TRACK>
  </COLLECTION>
</DJ_PLAYLISTS>
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
        f.write(valid_xml)
        xml_path = Path(f.name)

    try:
        tracks = parse_rekordbox_xml(xml_path)
        assert len(tracks) == 1
        assert tracks[0].name == "Test Track"
        assert tracks[0].artist == "Test Artist"
        assert tracks[0].bpm == 128.0
    finally:
        xml_path.unlink()
