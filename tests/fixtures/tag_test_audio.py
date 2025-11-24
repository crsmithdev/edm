"""Generate test audio files with metadata tags."""

import mutagen
from mutagen.id3 import TBPM, TPE1, TIT2, TALB
from mutagen.flac import FLAC
from pathlib import Path


def create_mp3_with_tags(output_path, bpm=128, artist="Test Artist", title="Test Title", album="Test Album"):
    """Create an MP3 file with ID3 tags."""
    # Copy one of our existing test files
    source = Path(__file__).parent / "click_128bpm.wav"
    
    # Convert to MP3 format for testing (we'll use WAV as base and add tags)
    import soundfile as sf
    import numpy as np
    
    data, sr = sf.read(source)
    # Truncate to 3 seconds for smaller test files
    data = data[:int(3 * sr)]
    sf.write(output_path, data, sr, format='WAV')
    
    # Add ID3 tags
    audio = mutagen.File(output_path, easy=False)
    if audio.tags is None:
        audio.add_tags()
    
    audio.tags.add(TBPM(encoding=3, text=str(bpm)))
    audio.tags.add(TPE1(encoding=3, text=artist))
    audio.tags.add(TIT2(encoding=3, text=title))
    audio.tags.add(TALB(encoding=3, text=album))
    audio.save()


def create_flac_with_tags(output_path, bpm=140, artist="FLAC Artist", title="FLAC Title", album="FLAC Album"):
    """Create a FLAC file with Vorbis comments."""
    source = Path(__file__).parent / "click_140bpm.wav"
    
    import soundfile as sf
    data, sr = sf.read(source)
    data = data[:int(3 * sr)]
    sf.write(output_path, data, sr, format='FLAC')
    
    audio = FLAC(output_path)
    audio['bpm'] = str(bpm)
    audio['artist'] = artist
    audio['title'] = title
    audio['album'] = album
    audio.save()


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    
    print("Creating tagged test audio files...")
    
    # Create MP3 with tags
    mp3_path = fixtures_dir / "tagged_128bpm.wav"
    create_mp3_with_tags(mp3_path, bpm=128, artist="Test Artist", title="Test Track", album="Test Album")
    print(f"✓ Created {mp3_path.name}")
    
    # Create FLAC with tags
    flac_path = fixtures_dir / "tagged_140bpm.flac"
    create_flac_with_tags(flac_path, bpm=140, artist="FLAC Artist", title="FLAC Track", album="FLAC Album")
    print(f"✓ Created {flac_path.name}")
    
    # Create file without BPM tag
    no_bpm_path = fixtures_dir / "no_bpm_tag.wav"
    create_mp3_with_tags(no_bpm_path, bpm=None, artist="No BPM Artist", title="No BPM", album="Test")
    audio = mutagen.File(no_bpm_path)
    if 'TBPM' in audio.tags:
        audio.tags.delall('TBPM')
        audio.save()
    print(f"✓ Created {no_bpm_path.name}")
    
    print("\nDone!")
