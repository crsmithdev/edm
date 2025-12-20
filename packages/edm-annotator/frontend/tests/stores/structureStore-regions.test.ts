import { describe, it, expect, beforeEach } from "vitest";
import { useStructureStore } from "@/stores/structureStore";

describe("structureStore - Region Operations", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
  });

  describe("Region Deletion - Merge with Previous", () => {
    it("preserves all non-adjacent region labels when deleting a region", () => {
      const { setBoundaries, removeBoundary, setRegionLabel } = useStructureStore.getState();

      // Set up 5 regions with labels
      setBoundaries([0, 10, 20, 30, 40, 50]);
      setRegionLabel(0, "intro");
      setRegionLabel(1, "verse");
      setRegionLabel(2, "chorus");
      setRegionLabel(3, "bridge");
      setRegionLabel(4, "outro");

      // Delete middle region (index 2, "chorus")
      removeBoundary(20);

      const { regions } = useStructureStore.getState();

      // All other regions should keep their labels
      expect(regions).toHaveLength(4);
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("verse");
      expect(regions[2].label).toBe("bridge"); // chorus merged with verse
      expect(regions[3].label).toBe("outro");
    });

    it("merges first region with next (removes boundary at end)", () => {
      const { setBoundaries, removeBoundary } = useStructureStore.getState();

      // Set up 3 regions: [0-10], [10-20], [20-30]
      setBoundaries([0, 10, 20, 30]);

      // Delete first region (should remove boundary at 10)
      removeBoundary(10);

      const { boundaries, regions } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 20, 30]);
      expect(regions).toHaveLength(2);
      expect(regions[0]).toEqual({ start: 0, end: 20, label: "unlabeled" });
      expect(regions[1]).toEqual({ start: 20, end: 30, label: "unlabeled" });
    });

    it("merges middle region with previous (removes boundary at start)", () => {
      const { setBoundaries, removeBoundary, setRegionLabel } = useStructureStore.getState();

      // Set up 3 regions with labels
      setBoundaries([0, 10, 20, 30]);
      setRegionLabel(0, "intro");
      setRegionLabel(1, "buildup");
      setRegionLabel(2, "breakdown");

      // Delete middle region (index 1)
      // Should remove boundary at 10 (start of region 1), merging with previous
      removeBoundary(10);

      const { boundaries, regions } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 20, 30]);
      expect(regions).toHaveLength(2);
      expect(regions[0]).toEqual({ start: 0, end: 20, label: "intro" });
      expect(regions[1]).toEqual({ start: 20, end: 30, label: "breakdown" });
    });

    it("merges last region with previous (removes boundary at start)", () => {
      const { setBoundaries, removeBoundary, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 10, 20, 30]);
      setRegionLabel(0, "intro");
      setRegionLabel(1, "buildup");
      setRegionLabel(2, "outro");

      // Delete last region (should remove boundary at 20)
      removeBoundary(20);

      const { boundaries, regions } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 10, 30]);
      expect(regions).toHaveLength(2);
      expect(regions[0]).toEqual({ start: 0, end: 10, label: "intro" });
      expect(regions[1]).toEqual({ start: 10, end: 30, label: "buildup" });
    });

    it("cannot delete when only one region remains", () => {
      const { setBoundaries, removeBoundary } = useStructureStore.getState();

      setBoundaries([0, 10]);

      const boundariesBefore = useStructureStore.getState().boundaries;
      removeBoundary(10);
      const boundariesAfter = useStructureStore.getState().boundaries;

      // Should be unchanged
      expect(boundariesAfter).toEqual(boundariesBefore);
    });

    it("preserves label of previous region when merging", () => {
      const { setBoundaries, removeBoundary, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 10, 20]);
      setRegionLabel(0, "intro");
      setRegionLabel(1, "buildup");

      // Delete second region (merges with first)
      removeBoundary(10);

      const { regions } = useStructureStore.getState();

      expect(regions[0].label).toBe("intro"); // Keeps first region's label
    });
  });

  describe("Current Region Highlighting", () => {
    it("identifies current region based on playhead position", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 10, 20, 30]);
      setRegionLabel(0, "intro");
      setRegionLabel(1, "buildup");
      setRegionLabel(2, "breakdown");

      const { regions } = useStructureStore.getState();

      // Test region detection logic
      const currentTime = 15;
      const currentRegion = regions.find(
        (r) => currentTime >= r.start && currentTime < r.end
      );

      expect(currentRegion).toBeDefined();
      expect(currentRegion?.label).toBe("buildup");
    });

    it("handles playhead at boundary edge", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 10, 20, 30]);

      const { regions } = useStructureStore.getState();

      // At exact boundary, should be in the next region
      const currentTime = 10;
      const currentRegion = regions.find(
        (r) => currentTime >= r.start && currentTime < r.end
      );

      expect(currentRegion?.start).toBe(10);
      expect(currentRegion?.end).toBe(20);
    });

    it("handles playhead at track start", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 10, 20, 30]);

      const { regions } = useStructureStore.getState();

      const currentTime = 0;
      const currentRegion = regions.find(
        (r) => currentTime >= r.start && currentTime < r.end
      );

      expect(currentRegion?.start).toBe(0);
      expect(currentRegion?.end).toBe(10);
    });

    it("handles playhead at track end", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 10, 20, 30]);

      const { regions } = useStructureStore.getState();

      const currentTime = 30;
      const currentRegion = regions.find(
        (r) => currentTime >= r.start && currentTime < r.end
      );

      // At track end, no region should match (currentTime < r.end)
      expect(currentRegion).toBeUndefined();
    });
  });

  describe("Boundary Loading with Track Duration", () => {
    it("adds track duration as final boundary when loading", () => {
      const { setBoundaries } = useStructureStore.getState();

      const trackDuration = 180;
      const loadedBoundaries = [0, 10, 20, 30]; // From annotation file

      // Simulate loading logic
      if (loadedBoundaries[loadedBoundaries.length - 1] !== trackDuration) {
        loadedBoundaries.push(trackDuration);
      }

      setBoundaries(loadedBoundaries);

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 10, 20, 30, 180]);
      expect(boundaries[boundaries.length - 1]).toBe(trackDuration);
    });

    it("does not duplicate track duration if already present", () => {
      const { setBoundaries } = useStructureStore.getState();

      const trackDuration = 180;
      const loadedBoundaries = [0, 10, 20, 180]; // Already ends at track duration

      // Simulate loading logic
      if (loadedBoundaries[loadedBoundaries.length - 1] !== trackDuration) {
        loadedBoundaries.push(trackDuration);
      }

      setBoundaries(loadedBoundaries);

      const { boundaries } = useStructureStore.getState();

      expect(boundaries).toEqual([0, 10, 20, 180]);
    });

    it("creates correct regions from boundaries with track duration", () => {
      const { setBoundaries } = useStructureStore.getState();

      const trackDuration = 180;
      const loadedBoundaries = [0, 10, 20, trackDuration];

      setBoundaries(loadedBoundaries);

      const { regions } = useStructureStore.getState();

      expect(regions).toHaveLength(3);
      expect(regions[0]).toEqual({ start: 0, end: 10, label: "unlabeled" });
      expect(regions[1]).toEqual({ start: 10, end: 20, label: "unlabeled" });
      expect(regions[2]).toEqual({ start: 20, end: 180, label: "unlabeled" });
    });
  });

  describe("Annotation Tier Tracking", () => {
    it("initializes with null annotation tier", () => {
      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBeNull();
    });

    it("sets annotation tier to 1 (reference)", () => {
      const { setAnnotationTier } = useStructureStore.getState();

      setAnnotationTier(1);

      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBe(1);
    });

    it("sets annotation tier to 2 (generated)", () => {
      const { setAnnotationTier } = useStructureStore.getState();

      setAnnotationTier(2);

      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBe(2);
    });

    it("sets annotation tier to null", () => {
      const { setAnnotationTier } = useStructureStore.getState();

      setAnnotationTier(2);
      setAnnotationTier(null);

      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBeNull();
    });

    it("resets annotation tier on reset", () => {
      const { setAnnotationTier, reset } = useStructureStore.getState();

      setAnnotationTier(2);
      reset();

      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBeNull();
    });
  });

  describe("Dirty State Tracking", () => {
    it("is not dirty when state matches saved state", () => {
      const { setBoundaries, markAsSaved, isDirty } = useStructureStore.getState();

      setBoundaries([0, 10, 20]);
      markAsSaved();

      expect(isDirty()).toBe(false);
    });

    it("is dirty when boundaries change", () => {
      const { setBoundaries, markAsSaved, addBoundary, isDirty } = useStructureStore.getState();

      setBoundaries([0, 10, 20]);
      markAsSaved();

      addBoundary(15);

      expect(isDirty()).toBe(true);
    });

    it("is dirty when labels change", () => {
      const { setBoundaries, markAsSaved, setRegionLabel, isDirty } = useStructureStore.getState();

      setBoundaries([0, 10, 20]);
      markAsSaved();

      setRegionLabel(0, "intro");

      expect(isDirty()).toBe(true);
    });

    it("is not dirty after saving changes", () => {
      const { setBoundaries, markAsSaved, setRegionLabel, isDirty } = useStructureStore.getState();

      setBoundaries([0, 10, 20]);
      markAsSaved();

      setRegionLabel(0, "intro");
      expect(isDirty()).toBe(true);

      markAsSaved();
      expect(isDirty()).toBe(false);
    });
  });
});
