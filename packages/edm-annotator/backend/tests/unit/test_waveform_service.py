"""Tests for WaveformService class."""

import numpy as np
import pytest

from edm_annotator.config import TestingConfig
from edm_annotator.services.audio_service import AudioService
from edm_annotator.services.waveform_service import WaveformService


@pytest.fixture
def waveform_config(temp_audio_dir):
    """Create test configuration for WaveformService."""
    config = {
        "AUDIO_DIR": temp_audio_dir,
        "WAVEFORM_SAMPLE_RATE": TestingConfig.WAVEFORM_SAMPLE_RATE,
        "WAVEFORM_HOP_LENGTH": TestingConfig.WAVEFORM_HOP_LENGTH,
        "WAVEFORM_FRAME_LENGTH": TestingConfig.WAVEFORM_FRAME_LENGTH,
        "BASS_LOW": TestingConfig.BASS_LOW,
        "BASS_HIGH": TestingConfig.BASS_HIGH,
        "MIDS_LOW": TestingConfig.MIDS_LOW,
        "MIDS_HIGH": TestingConfig.MIDS_HIGH,
        "HIGHS_LOW": TestingConfig.HIGHS_LOW,
        "AUDIO_EXTENSIONS": TestingConfig.AUDIO_EXTENSIONS,
    }
    return config


@pytest.fixture
def audio_service_mock(waveform_config):
    """Create AudioService instance for testing."""
    return AudioService(waveform_config)


@pytest.fixture
def waveform_service(waveform_config, audio_service_mock):
    """Create WaveformService instance for testing."""
    return WaveformService(waveform_config, audio_service_mock)


@pytest.fixture
def test_signal():
    """Generate test audio signal with known frequencies."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    # Create signal with components in different bands
    bass = 0.5 * np.sin(2 * np.pi * 100 * t)  # 100 Hz - bass band
    mids = 0.3 * np.sin(2 * np.pi * 1000 * t)  # 1000 Hz - mids band
    highs = 0.2 * np.sin(2 * np.pi * 8000 * t)  # 8000 Hz - highs band

    return (bass + mids + highs).astype(np.float32), sr


class TestGenerateWaveform:
    """Tests for generate_waveform method."""

    def test_generate_returns_complete_structure(self, waveform_service, sample_audio_file):
        """Test generate_waveform returns all required fields."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        # Verify all required keys are present
        assert "waveform_bass" in result
        assert "waveform_mids" in result
        assert "waveform_highs" in result
        assert "waveform_times" in result
        assert "duration" in result
        assert "sample_rate" in result

    def test_generate_output_format(self, waveform_service, sample_audio_file):
        """Test output format validation: lists of correct type."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        # All waveforms should be lists
        assert isinstance(result["waveform_bass"], list)
        assert isinstance(result["waveform_mids"], list)
        assert isinstance(result["waveform_highs"], list)
        assert isinstance(result["waveform_times"], list)

        # Should contain numeric values
        assert all(isinstance(x, int | float) for x in result["waveform_bass"])
        assert all(isinstance(x, int | float) for x in result["waveform_mids"])
        assert all(isinstance(x, int | float) for x in result["waveform_highs"])

    def test_generate_waveform_lengths_match(self, waveform_service, sample_audio_file):
        """Test all waveform arrays have same length."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        bass_len = len(result["waveform_bass"])
        mids_len = len(result["waveform_mids"])
        highs_len = len(result["waveform_highs"])
        times_len = len(result["waveform_times"])

        assert bass_len == mids_len == highs_len == times_len

    def test_generate_duration_accurate(self, waveform_service, sample_audio_file):
        """Test duration matches expected audio length."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        # Sample audio is 1 second
        assert 0.95 <= result["duration"] <= 1.05

    def test_generate_sample_rate_matches_config(self, waveform_service, sample_audio_file):
        """Test sample rate matches configured value."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        assert result["sample_rate"] == TestingConfig.WAVEFORM_SAMPLE_RATE

    def test_generate_times_monotonic_increasing(self, waveform_service, sample_audio_file):
        """Test time values are monotonically increasing."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        times = result["waveform_times"]
        assert all(times[i] < times[i + 1] for i in range(len(times) - 1))

    def test_generate_times_start_near_zero(self, waveform_service, sample_audio_file):
        """Test time axis starts near zero."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        assert result["waveform_times"][0] < 0.1

    def test_generate_times_end_near_duration(self, waveform_service, sample_audio_file):
        """Test time axis ends near total duration."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        last_time = result["waveform_times"][-1]
        duration = result["duration"]

        # Last time should be close to duration
        assert abs(last_time - duration) < 0.1

    def test_generate_non_negative_rms(self, waveform_service, sample_audio_file):
        """Test RMS values are non-negative."""
        result = waveform_service.generate_waveform(sample_audio_file.name)

        assert all(x >= 0 for x in result["waveform_bass"])
        assert all(x >= 0 for x in result["waveform_mids"])
        assert all(x >= 0 for x in result["waveform_highs"])

    def test_generate_handles_silent_audio(self, waveform_service, temp_audio_dir):
        """Test generate_waveform handles silent audio."""
        import soundfile as sf

        # Create silent audio file
        sr = 22050
        duration = 1.0
        silent = np.zeros(int(sr * duration), dtype=np.float32)

        audio_path = temp_audio_dir / "silent.wav"
        sf.write(audio_path, silent, sr)

        result = waveform_service.generate_waveform("silent.wav")

        # All RMS values should be near zero
        assert all(x < 0.01 for x in result["waveform_bass"])
        assert all(x < 0.01 for x in result["waveform_mids"])
        assert all(x < 0.01 for x in result["waveform_highs"])

    def test_generate_complete_dsp_pipeline(self, waveform_service, temp_audio_dir):
        """Test complete DSP pipeline with multi-band signal."""
        import soundfile as sf

        # Create audio with known frequency content
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Signal with bass, mids, and highs
        bass = 0.5 * np.sin(2 * np.pi * 100 * t)
        mids = 0.3 * np.sin(2 * np.pi * 1000 * t)
        highs = 0.2 * np.sin(2 * np.pi * 8000 * t)
        signal = (bass + mids + highs).astype(np.float32)

        audio_path = temp_audio_dir / "multiband.wav"
        sf.write(audio_path, signal, sr)

        result = waveform_service.generate_waveform("multiband.wav")

        # Each band should have energy
        bass_energy = np.mean(result["waveform_bass"])
        mids_energy = np.mean(result["waveform_mids"])
        highs_energy = np.mean(result["waveform_highs"])

        # All bands should have some energy
        assert bass_energy > 0.01
        assert mids_energy > 0.01
        assert highs_energy > 0.01


class TestBandpassFilter:
    """Tests for _apply_bandpass_filter method."""

    def test_bandpass_isolates_bass_frequencies(self, waveform_service):
        """Test bandpass filter correctly isolates bass frequencies (20-250 Hz)."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with bass and high frequency
        bass = np.sin(2 * np.pi * 100 * t)  # 100 Hz - in bass band
        high = np.sin(2 * np.pi * 5000 * t)  # 5000 Hz - outside bass band
        signal = bass + high

        # Apply bass filter
        filtered = waveform_service._apply_bandpass_filter(
            signal, sr, waveform_service.bass_low, waveform_service.bass_high
        )

        # Filtered signal should mainly contain bass
        bass_rms = np.sqrt(np.mean(bass**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        # RMS of filtered should be close to RMS of bass component
        assert abs(filtered_rms - bass_rms) < 0.2

    def test_bandpass_isolates_mids_frequencies(self, waveform_service):
        """Test bandpass filter correctly isolates mids frequencies (250-4000 Hz)."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with mids and bass
        mids = np.sin(2 * np.pi * 1000 * t)  # 1000 Hz - in mids band
        bass = np.sin(2 * np.pi * 50 * t)  # 50 Hz - outside mids band
        signal = mids + bass

        # Apply mids filter
        filtered = waveform_service._apply_bandpass_filter(
            signal, sr, waveform_service.mids_low, waveform_service.mids_high
        )

        # Filtered signal should mainly contain mids
        mids_rms = np.sqrt(np.mean(mids**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        # RMS of filtered should be close to RMS of mids component
        assert abs(filtered_rms - mids_rms) < 0.2

    def test_bandpass_preserves_signal_length(self, waveform_service):
        """Test bandpass filter preserves signal length."""
        sr = 22050
        duration = 1.0
        signal = np.random.randn(int(sr * duration)).astype(np.float32)

        filtered = waveform_service._apply_bandpass_filter(signal, sr, 100, 1000)

        assert len(filtered) == len(signal)

    def test_bandpass_returns_numpy_array(self, waveform_service):
        """Test bandpass filter returns numpy array."""
        sr = 22050
        signal = np.random.randn(sr).astype(np.float32)

        filtered = waveform_service._apply_bandpass_filter(signal, sr, 100, 1000)

        assert isinstance(filtered, np.ndarray)

    def test_bandpass_attenuates_out_of_band(self, waveform_service):
        """Test bandpass filter attenuates frequencies outside the band."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with frequency far outside bass band
        high_freq_signal = np.sin(2 * np.pi * 10000 * t)  # 10 kHz

        # Apply bass filter (20-250 Hz)
        filtered = waveform_service._apply_bandpass_filter(
            high_freq_signal, sr, waveform_service.bass_low, waveform_service.bass_high
        )

        # Filtered signal should be heavily attenuated
        filtered_rms = np.sqrt(np.mean(filtered**2))
        assert filtered_rms < 0.1


class TestHighpassFilter:
    """Tests for _apply_highpass_filter method."""

    def test_highpass_isolates_high_frequencies(self, waveform_service):
        """Test highpass filter correctly isolates high frequencies (4000+ Hz)."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with high and low frequencies
        highs = np.sin(2 * np.pi * 8000 * t)  # 8000 Hz - above cutoff
        lows = np.sin(2 * np.pi * 100 * t)  # 100 Hz - below cutoff
        signal = highs + lows

        # Apply highs filter
        filtered = waveform_service._apply_highpass_filter(signal, sr, waveform_service.highs_low)

        # Filtered signal should mainly contain highs
        highs_rms = np.sqrt(np.mean(highs**2))
        filtered_rms = np.sqrt(np.mean(filtered**2))

        # RMS of filtered should be close to RMS of highs component
        assert abs(filtered_rms - highs_rms) < 0.2

    def test_highpass_attenuates_low_frequencies(self, waveform_service):
        """Test highpass filter attenuates low frequencies."""
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create low frequency signal
        low_freq_signal = np.sin(2 * np.pi * 100 * t)  # 100 Hz

        # Apply highs filter (4000 Hz cutoff)
        filtered = waveform_service._apply_highpass_filter(
            low_freq_signal, sr, waveform_service.highs_low
        )

        # Filtered signal should be heavily attenuated
        filtered_rms = np.sqrt(np.mean(filtered**2))
        assert filtered_rms < 0.1

    def test_highpass_preserves_signal_length(self, waveform_service):
        """Test highpass filter preserves signal length."""
        sr = 22050
        duration = 1.0
        signal = np.random.randn(int(sr * duration)).astype(np.float32)

        filtered = waveform_service._apply_highpass_filter(signal, sr, 4000)

        assert len(filtered) == len(signal)

    def test_highpass_returns_numpy_array(self, waveform_service):
        """Test highpass filter returns numpy array."""
        sr = 22050
        signal = np.random.randn(sr).astype(np.float32)

        filtered = waveform_service._apply_highpass_filter(signal, sr, 4000)

        assert isinstance(filtered, np.ndarray)


class TestComputeRMS:
    """Tests for _compute_rms method."""

    def test_compute_rms_accuracy(self, waveform_service):
        """Test RMS calculation accuracy for known signal."""
        # Create constant amplitude signal
        amplitude = 0.5
        signal = amplitude * np.ones(22050, dtype=np.float32)

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        # RMS of constant signal should be close to amplitude
        # (may have edge effects from windowing)
        mean_rms = np.mean(rms)
        assert abs(mean_rms - amplitude) < 0.05

    def test_compute_rms_sine_wave(self, waveform_service):
        """Test RMS calculation for sine wave (RMS = amplitude / sqrt(2))."""
        sr = 22050
        duration = 1.0
        amplitude = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        signal = amplitude * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        expected_rms = amplitude / np.sqrt(2)  # ~0.707

        # Average RMS should be close to theoretical value
        mean_rms = np.mean(rms)
        assert abs(mean_rms - expected_rms) < 0.1

    def test_compute_rms_zero_signal(self, waveform_service):
        """Test RMS of zero signal is zero."""
        signal = np.zeros(22050, dtype=np.float32)

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        assert all(x < 0.001 for x in rms)

    def test_compute_rms_returns_1d_array(self, waveform_service):
        """Test RMS returns 1D array."""
        signal = np.random.randn(22050).astype(np.float32)

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        assert isinstance(rms, np.ndarray)
        assert rms.ndim == 1

    def test_compute_rms_non_negative(self, waveform_service):
        """Test RMS values are always non-negative."""
        # Create signal with positive and negative values
        signal = np.random.randn(22050).astype(np.float32)

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        assert all(x >= 0 for x in rms)

    def test_compute_rms_frame_parameters(self, waveform_service):
        """Test RMS respects frame_length and hop_length parameters."""
        sr = 22050
        duration = 1.0
        signal = np.random.randn(int(sr * duration)).astype(np.float32)

        # Different hop lengths should produce different number of frames
        rms_512 = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)
        rms_256 = waveform_service._compute_rms(signal, frame_length=1024, hop_length=256)

        # Smaller hop length = more frames
        assert len(rms_256) > len(rms_512)

    def test_compute_rms_dynamic_range(self, waveform_service):
        """Test RMS captures dynamic range changes."""
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create signal with changing amplitude
        # First half: amplitude 0.2, second half: amplitude 0.8
        signal = np.zeros(len(t), dtype=np.float32)
        half = len(t) // 2
        signal[:half] = 0.2 * np.sin(2 * np.pi * 440 * t[:half])
        signal[half:] = 0.8 * np.sin(2 * np.pi * 440 * t[half:])

        rms = waveform_service._compute_rms(signal, frame_length=1024, hop_length=512)

        # Split RMS into two halves
        rms_first_half = rms[: len(rms) // 2]
        rms_second_half = rms[len(rms) // 2 :]

        # Second half should have higher average RMS
        assert np.mean(rms_second_half) > np.mean(rms_first_half)
