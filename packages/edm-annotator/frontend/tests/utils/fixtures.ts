import type { Region, SectionLabel, Boundary } from "@/types/structure";
import type { WaveformData } from "@/types/waveform";
import type { LoadTrackResponse } from "@/types/api";

/**
 * Test fixtures for creating mock data
 */

export interface MockAnnotationOptions {
  /**
   * Number of boundaries to create (default: 4)
   */
  boundaryCount?: number;

  /**
   * Total duration of the track in seconds (default: 180)
   */
  duration?: number;

  /**
   * Beats per minute (default: 128)
   */
  bpm?: number;

  /**
   * Labels to cycle through for regions (default: ["intro", "buildup", "breakdown", "outro"])
   */
  labels?: SectionLabel[];

  /**
   * Annotation tier (1 = reference/hand-tagged, 2 = generated, null = none)
   */
  tier?: number | null;

  /**
   * Downbeat offset in seconds (default: 0)
   */
  downbeat?: number;

  /**
   * Specific boundary times (overrides boundaryCount and duration-based distribution)
   */
  boundaryTimes?: number[];
}

/**
 * Create a mock annotation with evenly-spaced boundaries
 *
 * @example
 * // Create annotation with 4 boundaries over 3 minutes at 128 BPM
 * const annotation = createMockAnnotation();
 *
 * @example
 * // Create annotation with custom parameters
 * const annotation = createMockAnnotation({
 *   boundaryCount: 6,
 *   duration: 240,
 *   bpm: 140,
 *   labels: ["intro", "breakdown"],
 *   tier: 1
 * });
 */
export function createMockAnnotation(options: MockAnnotationOptions = {}) {
  const {
    boundaryCount = 4,
    duration = 180,
    bpm = 128,
    labels = ["intro", "buildup", "breakdown", "outro"],
    tier = null,
    downbeat = 0,
    boundaryTimes,
  } = options;

  // Create boundaries
  let boundaries: number[];
  if (boundaryTimes) {
    boundaries = [...boundaryTimes].sort((a, b) => a - b);
  } else {
    // Evenly distribute boundaries across duration
    boundaries = Array.from({ length: boundaryCount }, (_, i) => {
      return (duration / (boundaryCount - 1)) * i;
    });
  }

  // Create regions from boundaries
  const regions: Region[] = [];
  for (let i = 0; i < boundaries.length - 1; i++) {
    regions.push({
      start: boundaries[i],
      end: boundaries[i + 1],
      label: labels[i % labels.length],
    });
  }

  // Create boundary objects with labels
  const boundaryObjects: Boundary[] = boundaries.slice(0, -1).map((time, i) => ({
    time,
    label: labels[i % labels.length],
  }));

  return {
    boundaries,
    regions,
    boundaryObjects,
    bpm,
    downbeat,
    tier,
    duration,
  };
}

/**
 * Options for creating mock waveform data
 */
export interface MockWaveformOptions {
  /**
   * Total duration in seconds
   */
  duration: number;

  /**
   * Sample rate (samples per second, default: 10)
   */
  sampleRate?: number;

  /**
   * Pattern for waveform data generation
   * - "flat": all zeros
   * - "random": random values between 0 and 1
   * - "sine": sine wave pattern
   * - "peaks": random with periodic peaks
   */
  pattern?: "flat" | "random" | "sine" | "peaks";

  /**
   * Amplitude multiplier for waveform values (default: 1.0)
   */
  amplitude?: number;
}

/**
 * Create mock waveform data for testing
 *
 * @example
 * // Create simple flat waveform
 * const waveform = createMockWaveform(180);
 *
 * @example
 * // Create waveform with random data
 * const waveform = createMockWaveform(120, {
 *   sampleRate: 20,
 *   pattern: "random",
 *   amplitude: 0.8
 * });
 */
export function createMockWaveform(
  duration: number,
  options: Omit<MockWaveformOptions, "duration"> = {}
): WaveformData {
  const {
    sampleRate = 10,
    pattern = "flat",
    amplitude = 1.0,
  } = options;

  const numSamples = Math.floor(duration * sampleRate);
  const waveform_times = Array.from({ length: numSamples }, (_, i) => i / sampleRate);

  // Generate waveform data based on pattern
  const generateSamples = (offset = 0): number[] => {
    return Array.from({ length: numSamples }, (_, i) => {
      switch (pattern) {
        case "flat":
          return 0;
        case "random":
          return Math.random() * amplitude;
        case "sine": {
          const freq = 0.5 + offset * 0.2; // Different frequencies for each band
          return Math.abs(Math.sin(i * freq * 0.1)) * amplitude;
        }
        case "peaks": {
          // Random baseline with periodic peaks
          const baseline = Math.random() * 0.2 * amplitude;
          const isPeak = i % 20 === 0;
          return isPeak ? amplitude : baseline;
        }
        default:
          return 0;
      }
    });
  };

  return {
    waveform_bass: generateSamples(0),
    waveform_mids: generateSamples(1),
    waveform_highs: generateSamples(2),
    waveform_times,
    duration,
    sample_rate: sampleRate,
  };
}

/**
 * Options for creating a complete mock track response
 */
export interface MockTrackResponseOptions extends MockAnnotationOptions {
  /**
   * Filename for the track
   */
  filename?: string;

  /**
   * Waveform pattern (default: "flat")
   */
  waveformPattern?: MockWaveformOptions["pattern"];

  /**
   * Waveform sample rate (default: 10)
   */
  waveformSampleRate?: number;
}

/**
 * Create a complete mock track response including waveform and annotation data
 *
 * @example
 * const track = createMockTrackResponse({
 *   filename: "test-track.mp3",
 *   duration: 180,
 *   bpm: 140,
 *   boundaryCount: 5,
 *   tier: 1
 * });
 */
export function createMockTrackResponse(
  options: MockTrackResponseOptions = {}
): LoadTrackResponse {
  const {
    filename = "test-track.mp3",
    duration = 180,
    waveformPattern = "flat",
    waveformSampleRate = 10,
    ...annotationOptions
  } = options;

  const annotation = createMockAnnotation({
    duration,
    ...annotationOptions,
  });

  const waveformData = createMockWaveform(duration, {
    sampleRate: waveformSampleRate,
    pattern: waveformPattern,
  });

  return {
    filename,
    bpm: annotation.bpm,
    downbeat: annotation.downbeat,
    boundaries: annotation.boundaryObjects,
    annotation_tier: annotation.tier,
    ...waveformData,
  };
}

/**
 * Create a simple mock region with sensible defaults
 */
export function createMockRegion(
  start: number,
  end: number,
  label: SectionLabel = "unlabeled"
): Region {
  return { start, end, label };
}

/**
 * Create multiple mock regions from an array of time points
 *
 * @example
 * const regions = createMockRegions([0, 30, 60, 90], ["intro", "buildup", "breakdown"]);
 */
export function createMockRegions(
  times: number[],
  labels: SectionLabel[] = ["unlabeled"]
): Region[] {
  const regions: Region[] = [];
  for (let i = 0; i < times.length - 1; i++) {
    regions.push({
      start: times[i],
      end: times[i + 1],
      label: labels[i % labels.length],
    });
  }
  return regions;
}

/**
 * Create a simple mock boundary
 */
export function createMockBoundary(
  time: number,
  label: SectionLabel = "unlabeled"
): Boundary {
  return { time, label };
}

/**
 * Create multiple mock boundaries
 */
export function createMockBoundaries(
  times: number[],
  labels: SectionLabel[] = ["unlabeled"]
): Boundary[] {
  return times.map((time, i) => ({
    time,
    label: labels[i % labels.length],
  }));
}
