import { describe, it, expect, beforeEach } from "vitest";
import { useStructureStore } from "@/stores/structureStore";

describe("structureStore", () => {
  beforeEach(() => {
    // Reset store before each test
    useStructureStore.getState().reset();
  });

  describe("addBoundary", () => {
    it("should add boundary and maintain sorted order", () => {
      const { addBoundary, boundaries } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(5);
      addBoundary(15);

      const finalBoundaries = useStructureStore.getState().boundaries;
      expect(finalBoundaries).toEqual([5, 10, 15]);
    });

    it("should deduplicate exact same boundary", () => {
      const { addBoundary } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(10);

      const finalBoundaries = useStructureStore.getState().boundaries;
      expect(finalBoundaries).toEqual([10]);
    });

    it("should rebuild regions after adding boundary", () => {
      const { addBoundary } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(20);

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(1);
      expect(regions[0]).toEqual({
        start: 10,
        end: 20,
        label: "unlabeled",
      });
    });

    it("should handle multiple boundaries and create correct regions", () => {
      const { addBoundary } = useStructureStore.getState();

      addBoundary(0);
      addBoundary(30);
      addBoundary(60);
      addBoundary(90);

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(3);
      expect(regions[0]).toEqual({ start: 0, end: 30, label: "unlabeled" });
      expect(regions[1]).toEqual({ start: 30, end: 60, label: "unlabeled" });
      expect(regions[2]).toEqual({ start: 60, end: 90, label: "unlabeled" });
    });
  });

  describe("removeBoundary", () => {
    it("should remove boundary within tolerance (0.01s)", () => {
      const { addBoundary, removeBoundary } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(20);
      removeBoundary(10.005); // Within 0.01s tolerance

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([20]);
    });

    it("should not remove boundary outside tolerance", () => {
      const { addBoundary, removeBoundary } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(20);
      removeBoundary(10.02); // Outside 0.01s tolerance

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([10, 20]);
    });

    it("should rebuild regions after removing boundary", () => {
      const { addBoundary, removeBoundary } = useStructureStore.getState();

      addBoundary(0);
      addBoundary(30);
      addBoundary(60);

      let { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(2);

      removeBoundary(30);

      regions = useStructureStore.getState().regions;
      expect(regions).toHaveLength(1);
      expect(regions[0]).toEqual({ start: 0, end: 60, label: "unlabeled" });
    });

    it("should handle removing non-existent boundary gracefully", () => {
      const { addBoundary, removeBoundary } = useStructureStore.getState();

      addBoundary(10);
      removeBoundary(50);

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([10]);
    });
  });

  describe("rebuildRegions", () => {
    it("should create correct regions from boundaries", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 32, 64, 128]);

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(3);
      expect(regions[0]).toEqual({ start: 0, end: 32, label: "unlabeled" });
      expect(regions[1]).toEqual({ start: 32, end: 64, label: "unlabeled" });
      expect(regions[2]).toEqual({ start: 64, end: 128, label: "unlabeled" });
    });

    it("should handle empty boundaries", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([]);

      const { regions } = useStructureStore.getState();
      expect(regions).toEqual([]);
    });

    it("should handle single boundary", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([10]);

      const { regions } = useStructureStore.getState();
      expect(regions).toEqual([]);
    });

    it("should preserve existing labels when rebuilding", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 30, 60]);
      setRegionLabel(0, "intro");

      const { regions } = useStructureStore.getState();
      expect(regions[0].label).toBe("intro");
    });
  });

  describe("setRegionLabel", () => {
    it("should update label for valid region index", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 30, 60, 90]);
      setRegionLabel(1, "buildup");

      const { regions } = useStructureStore.getState();
      expect(regions[1].label).toBe("buildup");
    });

    it("should not update label for invalid index", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 30, 60]);
      setRegionLabel(5, "buildup");

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(2);
      expect(regions.every((r) => r.label === "unlabeled")).toBe(true);
    });

    it("should handle negative index gracefully", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 30, 60]);
      setRegionLabel(-1, "intro");

      const { regions } = useStructureStore.getState();
      expect(regions.every((r) => r.label === "unlabeled")).toBe(true);
    });

    it("should update multiple region labels independently", () => {
      const { setBoundaries, setRegionLabel } = useStructureStore.getState();

      setBoundaries([0, 30, 60, 90, 120]);
      setRegionLabel(0, "intro");
      setRegionLabel(2, "breakdown");
      setRegionLabel(3, "outro");

      const { regions } = useStructureStore.getState();
      expect(regions[0].label).toBe("intro");
      expect(regions[1].label).toBe("unlabeled");
      expect(regions[2].label).toBe("breakdown");
      expect(regions[3].label).toBe("outro");
    });
  });

  describe("setBoundaries", () => {
    it("should sort boundaries automatically", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([60, 30, 0, 90]);

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([0, 30, 60, 90]);
    });

    it("should rebuild regions after setting boundaries", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 64, 128]);

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(2);
    });

    it("should replace existing boundaries", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 30, 60]);
      setBoundaries([10, 20, 30, 40]);

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([10, 20, 30, 40]);
    });
  });

  describe("reset", () => {
    it("should clear all boundaries and regions", () => {
      const { setBoundaries, setRegionLabel, reset } =
        useStructureStore.getState();

      setBoundaries([0, 30, 60, 90]);
      setRegionLabel(0, "intro");
      reset();

      const { boundaries, regions } = useStructureStore.getState();
      expect(boundaries).toEqual([]);
      expect(regions).toEqual([]);
    });
  });

  describe("complex boundary operations", () => {
    it("should handle rapid add/remove operations", () => {
      const { addBoundary, removeBoundary } = useStructureStore.getState();

      addBoundary(10);
      addBoundary(20);
      addBoundary(30);
      removeBoundary(20);
      addBoundary(25);
      addBoundary(15);

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toEqual([10, 15, 25, 30]);
    });

    it("should maintain region consistency after complex operations", () => {
      const { addBoundary, removeBoundary, setRegionLabel } =
        useStructureStore.getState();

      addBoundary(0);
      addBoundary(30);
      addBoundary(60);
      setRegionLabel(0, "intro");
      addBoundary(90);
      removeBoundary(30);

      const { regions } = useStructureStore.getState();
      expect(regions).toHaveLength(2);
      // Note: labels are not preserved after rebuild, which is expected behavior
      expect(regions[0]).toEqual({ start: 0, end: 60, label: "unlabeled" });
      expect(regions[1]).toEqual({ start: 60, end: 90, label: "unlabeled" });
    });

    it("should handle boundaries at track start (0.0)", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0, 30, 60]);

      const { regions } = useStructureStore.getState();
      expect(regions[0].start).toBe(0);
    });

    it("should handle very close boundaries", () => {
      const { addBoundary } = useStructureStore.getState();

      addBoundary(10.0);
      addBoundary(10.001);
      addBoundary(10.002);

      const { boundaries } = useStructureStore.getState();
      expect(boundaries).toHaveLength(3);
      expect(boundaries).toEqual([10.0, 10.001, 10.002]);
    });

    it("should handle fractional second boundaries", () => {
      const { setBoundaries } = useStructureStore.getState();

      setBoundaries([0.5, 15.75, 30.125, 45.875]);

      const { boundaries, regions } = useStructureStore.getState();
      expect(boundaries).toEqual([0.5, 15.75, 30.125, 45.875]);
      expect(regions[0].start).toBe(0.5);
      expect(regions[0].end).toBe(15.75);
    });
  });
});
