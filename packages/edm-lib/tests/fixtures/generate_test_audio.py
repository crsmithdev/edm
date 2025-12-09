"""Generate synthetic test audio files with known BPM for testing."""

from pathlib import Path

import numpy as np
import soundfile as sf


def generate_click_track(bpm, duration=10.0, sample_rate=44100, output_path=None):
    """Generate a simple click track with a known BPM."""
    if output_path is None:
        fixtures_dir = Path(__file__).parent
        output_path = fixtures_dir / f"click_{int(bpm)}bpm.wav"

    beats_per_second = bpm / 60.0
    samples_per_beat = int(sample_rate / beats_per_second)
    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Create click sound
    click_duration = 0.01
    click_samples = int(click_duration * sample_rate)
    t_click = np.linspace(0, click_duration, click_samples)
    click = np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 50)

    # Place clicks at each beat
    for beat in range(int(duration * beats_per_second)):
        start_idx = beat * samples_per_beat
        end_idx = start_idx + len(click)
        if end_idx <= n_samples:
            audio[start_idx:end_idx] += click

    audio = audio / np.abs(audio).max() * 0.8
    stereo_audio = np.stack([audio, audio], axis=1)
    sf.write(output_path, stereo_audio, sample_rate)
    return output_path


def generate_beat_pattern(bpm, duration=10.0, sample_rate=44100, output_path=None):
    """Generate a beat pattern with kick, snare, and hi-hat."""
    if output_path is None:
        fixtures_dir = Path(__file__).parent
        output_path = fixtures_dir / f"beat_{int(bpm)}bpm.wav"

    beats_per_second = bpm / 60.0
    samples_per_beat = int(sample_rate / beats_per_second)
    n_samples = int(duration * sample_rate)
    audio = np.zeros(n_samples, dtype=np.float32)

    # Create drum sounds
    kick_duration, kick_samples = 0.15, int(0.15 * sample_rate)
    t_kick = np.linspace(0, kick_duration, kick_samples)
    kick = np.sin(2 * np.pi * 60 * t_kick) * np.exp(-t_kick * 20)

    snare_duration, snare_samples = 0.1, int(0.1 * sample_rate)
    t_snare = np.linspace(0, snare_duration, snare_samples)
    snare = np.random.randn(snare_samples) * np.exp(-t_snare * 30) * 0.5

    hihat_duration, hihat_samples = 0.05, int(0.05 * sample_rate)
    t_hihat = np.linspace(0, hihat_duration, hihat_samples)
    hihat = np.random.randn(hihat_samples) * np.exp(-t_hihat * 100) * 0.3

    # Place patterns (4/4 time)
    total_beats = int(duration * beats_per_second)
    for beat in range(total_beats):
        beat_in_bar = beat % 4
        start_idx = beat * samples_per_beat

        if beat_in_bar in [0, 2]:  # Kick on 1 and 3
            end_idx = start_idx + len(kick)
            if end_idx <= n_samples:
                audio[start_idx:end_idx] += kick

        if beat_in_bar in [1, 3]:  # Snare on 2 and 4
            end_idx = start_idx + len(snare)
            if end_idx <= n_samples:
                audio[start_idx:end_idx] += snare

        end_idx = start_idx + len(hihat)  # Hi-hat on every beat
        if end_idx <= n_samples:
            audio[start_idx:end_idx] += hihat

    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.8

    stereo_audio = np.stack([audio, audio], axis=1)
    sf.write(output_path, stereo_audio, sample_rate)
    return output_path


if __name__ == "__main__":
    test_bpms = [120, 125, 128, 140, 150, 174]
    print("Generating synthetic test audio files...")
    for bpm in test_bpms:
        click_path = generate_click_track(bpm)
        print(f"✓ Generated {click_path.name}")
        beat_path = generate_beat_pattern(bpm)
        print(f"✓ Generated {beat_path.name}")
    print("\nDone! Test audio files are ready in tests/fixtures/")
