import { describe, it, expect } from "vitest";
import {
  createMockAnnotation,
  createMockWaveform,
  createMockTrackResponse,
  createMockRegion,
  createMockRegions,
  createMockBoundary,
  createMockBoundaries,
} from "./fixtures";

describe("fixtures", () => {
  describe("createMockAnnotation", () => {
    it("creates annotation with default values", () => {
      const annotation = createMockAnnotation();

      expect(annotation.boundaries).toHaveLength(4);
      expect(annotation.regions).toHaveLength(3); // n-1 regions from n boundaries
      expect(annotation.bpm).toBe(128);
      expect(annotation.downbeat).toBe(0);
      expect(annotation.tier).toBeNull();
      expect(annotation.duration).toBe(180);
    });

    it("creates annotation with custom boundary count", () => {
      const annotation = createMockAnnotation({ boundaryCount: 6 });

      expect(annotation.boundaries).toHaveLength(6);
      expect(annotation.regions).toHaveLength(5);
    });

    it("creates annotation with custom duration", () => {
      const annotation = createMockAnnotation({ duration: 240 });

      expect(annotation.duration).toBe(240);
      // Boundaries should span 0 to 240
      expect(annotation.boundaries[0]).toBe(0);
      expect(annotation.boundaries[annotation.boundaries.length - 1]).toBe(240);
    });

    it("creates annotation with custom BPM", () => {
      const annotation = createMockAnnotation({ bpm: 140 });

      expect(annotation.bpm).toBe(140);
    });

    it("cycles through labels for regions", () => {
      const labels = ["intro", "buildup"];
      const annotation = createMockAnnotation({ boundaryCount: 5, labels });

      // With 5 boundaries, we have 4 regions
      expect(annotation.regions).toHaveLength(4);
      expect(annotation.regions[0].label).toBe("intro");
      expect(annotation.regions[1].label).toBe("buildup");
      expect(annotation.regions[2].label).toBe("intro"); // Cycles back
      expect(annotation.regions[3].label).toBe("buildup");
    });

    it("creates evenly-spaced boundaries", () => {
      const annotation = createMockAnnotation({ boundaryCount: 3, duration: 100 });

      expect(annotation.boundaries).toEqual([0, 50, 100]);
    });

    it("uses custom boundary times when provided", () => {
      const boundaryTimes = [0, 10, 30, 45, 60];
      const annotation = createMockAnnotation({ boundaryTimes });

      expect(annotation.boundaries).toEqual(boundaryTimes);
      expect(annotation.regions).toHaveLength(4);
    });

    it("sorts custom boundary times", () => {
      const boundaryTimes = [30, 0, 60, 10];
      const annotation = createMockAnnotation({ boundaryTimes });

      expect(annotation.boundaries).toEqual([0, 10, 30, 60]);
    });

    it("creates boundary objects with labels", () => {
      const annotation = createMockAnnotation({ boundaryCount: 3 });

      expect(annotation.boundaryObjects).toHaveLength(2); // n-1 boundaries
      expect(annotation.boundaryObjects[0]).toHaveProperty("time");
      expect(annotation.boundaryObjects[0]).toHaveProperty("label");
    });

    it("sets custom tier", () => {
      const annotation1 = createMockAnnotation({ tier: 1 });
      expect(annotation1.tier).toBe(1);

      const annotation2 = createMockAnnotation({ tier: 2 });
      expect(annotation2.tier).toBe(2);

      const annotation3 = createMockAnnotation({ tier: null });
      expect(annotation3.tier).toBeNull();
    });

    it("sets custom downbeat", () => {
      const annotation = createMockAnnotation({ downbeat: 2.5 });

      expect(annotation.downbeat).toBe(2.5);
    });
  });

  describe("createMockWaveform", () => {
    it("creates waveform with default sample rate", () => {
      const waveform = createMockWaveform(10);

      expect(waveform.duration).toBe(10);
      expect(waveform.sample_rate).toBe(10);
      expect(waveform.waveform_times).toHaveLength(100); // 10 seconds * 10 samples/sec
    });

    it("creates waveform with custom sample rate", () => {
      const waveform = createMockWaveform(10, { sampleRate: 20 });

      expect(waveform.sample_rate).toBe(20);
      expect(waveform.waveform_times).toHaveLength(200); // 10 seconds * 20 samples/sec
    });

    it("creates all three frequency bands", () => {
      const waveform = createMockWaveform(10);

      expect(waveform.waveform_bass).toBeDefined();
      expect(waveform.waveform_mids).toBeDefined();
      expect(waveform.waveform_highs).toBeDefined();
      expect(waveform.waveform_bass).toHaveLength(100);
      expect(waveform.waveform_mids).toHaveLength(100);
      expect(waveform.waveform_highs).toHaveLength(100);
    });

    it("creates flat waveform pattern", () => {
      const waveform = createMockWaveform(5, { pattern: "flat" });

      // All values should be 0
      expect(waveform.waveform_bass.every((v) => v === 0)).toBe(true);
      expect(waveform.waveform_mids.every((v) => v === 0)).toBe(true);
      expect(waveform.waveform_highs.every((v) => v === 0)).toBe(true);
    });

    it("creates random waveform pattern", () => {
      const waveform = createMockWaveform(5, { pattern: "random" });

      // Should have some non-zero values
      const hasNonZero = waveform.waveform_bass.some((v) => v > 0);
      expect(hasNonZero).toBe(true);
    });

    it("creates sine waveform pattern", () => {
      const waveform = createMockWaveform(5, { pattern: "sine" });

      // Should have some non-zero values
      const hasNonZero = waveform.waveform_bass.some((v) => v > 0);
      expect(hasNonZero).toBe(true);
    });

    it("creates peaks waveform pattern", () => {
      const waveform = createMockWaveform(5, { pattern: "peaks", sampleRate: 20 });

      // Should have periodic peaks
      const hasNonZero = waveform.waveform_bass.some((v) => v > 0);
      expect(hasNonZero).toBe(true);
    });

    it("applies amplitude multiplier", () => {
      const waveform1 = createMockWaveform(5, { pattern: "sine", amplitude: 1.0 });
      const waveform2 = createMockWaveform(5, { pattern: "sine", amplitude: 0.5 });

      // With lower amplitude, max values should be lower
      const max1 = Math.max(...waveform1.waveform_bass);
      const max2 = Math.max(...waveform2.waveform_bass);
      expect(max2).toBeLessThan(max1);
    });

    it("creates correct time array", () => {
      const waveform = createMockWaveform(10, { sampleRate: 10 });

      expect(waveform.waveform_times[0]).toBe(0);
      expect(waveform.waveform_times[10]).toBe(1); // 1 second
      expect(waveform.waveform_times[50]).toBe(5); // 5 seconds
    });
  });

  describe("createMockTrackResponse", () => {
    it("creates complete track response with defaults", () => {
      const track = createMockTrackResponse();

      expect(track.filename).toBe("test-track.mp3");
      expect(track.bpm).toBe(128);
      expect(track.downbeat).toBe(0);
      expect(track.boundaries).toBeDefined();
      expect(track.annotation_tier).toBeNull();
      expect(track.waveform_bass).toBeDefined();
      expect(track.waveform_mids).toBeDefined();
      expect(track.waveform_highs).toBeDefined();
      expect(track.waveform_times).toBeDefined();
      expect(track.duration).toBe(180);
    });

    it("creates track with custom filename", () => {
      const track = createMockTrackResponse({ filename: "custom-track.wav" });

      expect(track.filename).toBe("custom-track.wav");
    });

    it("creates track with custom annotation parameters", () => {
      const track = createMockTrackResponse({
        duration: 240,
        bpm: 140,
        boundaryCount: 6,
        tier: 1,
      });

      expect(track.duration).toBe(240);
      expect(track.bpm).toBe(140);
      expect(track.boundaries).toHaveLength(5); // n-1 boundaries
      expect(track.annotation_tier).toBe(1);
    });

    it("creates track with custom waveform pattern", () => {
      const track = createMockTrackResponse({ waveformPattern: "random" });

      // Should have some non-zero values
      const hasNonZero = track.waveform_bass.some((v) => v > 0);
      expect(hasNonZero).toBe(true);
    });

    it("creates track with custom waveform sample rate", () => {
      const track = createMockTrackResponse({ waveformSampleRate: 20 });

      expect(track.sample_rate).toBe(20);
    });
  });

  describe("createMockRegion", () => {
    it("creates region with default label", () => {
      const region = createMockRegion(0, 10);

      expect(region.start).toBe(0);
      expect(region.end).toBe(10);
      expect(region.label).toBe("unlabeled");
    });

    it("creates region with custom label", () => {
      const region = createMockRegion(10, 20, "intro");

      expect(region.start).toBe(10);
      expect(region.end).toBe(20);
      expect(region.label).toBe("intro");
    });
  });

  describe("createMockRegions", () => {
    it("creates regions from time array", () => {
      const regions = createMockRegions([0, 30, 60, 90]);

      expect(regions).toHaveLength(3);
      expect(regions[0].start).toBe(0);
      expect(regions[0].end).toBe(30);
      expect(regions[1].start).toBe(30);
      expect(regions[1].end).toBe(60);
      expect(regions[2].start).toBe(60);
      expect(regions[2].end).toBe(90);
    });

    it("creates regions with default labels", () => {
      const regions = createMockRegions([0, 10, 20]);

      expect(regions[0].label).toBe("unlabeled");
      expect(regions[1].label).toBe("unlabeled");
    });

    it("creates regions with custom labels", () => {
      const regions = createMockRegions([0, 10, 20, 30], ["intro", "buildup"]);

      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("buildup");
      expect(regions[2].label).toBe("intro"); // Cycles
    });
  });

  describe("createMockBoundary", () => {
    it("creates boundary with default label", () => {
      const boundary = createMockBoundary(10);

      expect(boundary.time).toBe(10);
      expect(boundary.label).toBe("unlabeled");
    });

    it("creates boundary with custom label", () => {
      const boundary = createMockBoundary(20, "intro");

      expect(boundary.time).toBe(20);
      expect(boundary.label).toBe("intro");
    });
  });

  describe("createMockBoundaries", () => {
    it("creates boundaries from time array", () => {
      const boundaries = createMockBoundaries([0, 30, 60]);

      expect(boundaries).toHaveLength(3);
      expect(boundaries[0].time).toBe(0);
      expect(boundaries[1].time).toBe(30);
      expect(boundaries[2].time).toBe(60);
    });

    it("creates boundaries with default labels", () => {
      const boundaries = createMockBoundaries([0, 10, 20]);

      expect(boundaries[0].label).toBe("unlabeled");
      expect(boundaries[1].label).toBe("unlabeled");
      expect(boundaries[2].label).toBe("unlabeled");
    });

    it("creates boundaries with custom labels", () => {
      const boundaries = createMockBoundaries([0, 10, 20], ["intro", "buildup"]);

      expect(boundaries[0].label).toBe("intro");
      expect(boundaries[1].label).toBe("buildup");
      expect(boundaries[2].label).toBe("intro"); // Cycles
    });
  });

  describe("Integration", () => {
    it("creates consistent annotation and waveform data", () => {
      const duration = 120;
      const annotation = createMockAnnotation({ duration, boundaryCount: 5 });
      const waveform = createMockWaveform(duration);

      expect(annotation.duration).toBe(waveform.duration);
      expect(annotation.boundaries[0]).toBe(0);
      expect(annotation.boundaries[annotation.boundaries.length - 1]).toBe(duration);
    });

    it("creates track response with matching durations", () => {
      const track = createMockTrackResponse({ duration: 150 });

      expect(track.duration).toBe(150);
      expect(track.boundaries![track.boundaries!.length - 1].time).toBeLessThanOrEqual(150);
    });
  });
});
