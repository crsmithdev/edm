import { create } from "zustand";
import type { Region, SectionLabel } from "@/types/structure";

interface StructureState {
  // State
  boundaries: number[];
  regions: Region[];
  savedState: { regions: Region[]; boundaries: number[] } | null;
  annotationTier: number | null; // 1 = reference (hand-tagged), 2 = generated, null = none

  // Actions
  addBoundary: (time: number) => void;
  removeBoundary: (time: number) => void;
  setRegionLabel: (index: number, label: SectionLabel) => void;
  rebuildRegions: () => void;
  setBoundaries: (boundaries: number[]) => void;
  clearBoundaries: () => void;
  setAnnotationTier: (tier: number | null) => void;
  markAsSaved: () => void;
  isDirty: () => boolean;
  reset: () => void;
}

export const useStructureStore = create<StructureState>((set, get) => ({
  // Initial state
  boundaries: [],
  regions: [],
  savedState: null,
  annotationTier: null,

  // Actions
  addBoundary: (time) => {
    const { boundaries } = get();
    // Check if boundary already exists (within very tight tolerance for precision)
    const exists = boundaries.some((b) => Math.abs(b - time) < 0.00001);
    if (exists) {
      return;
    }
    const newBoundaries = [...boundaries, time].sort((a, b) => a - b);
    set({ boundaries: newBoundaries });
    get().rebuildRegions();
  },

  removeBoundary: (time) => {
    const { boundaries } = get();
    // Cannot delete if only 2 boundaries remain (minimum for 1 region)
    if (boundaries.length <= 2) {
      return;
    }
    // Use looser tolerance for removal (user interaction)
    // Keep boundaries that are farther away than the tolerance
    const newBoundaries = boundaries.filter((t) => Math.abs(t - time) >= 0.01);
    set({ boundaries: newBoundaries });
    get().rebuildRegions();
  },

  setRegionLabel: (index, label) => {
    const { regions } = get();
    const newRegions = [...regions];
    if (index >= 0 && index < newRegions.length) {
      newRegions[index] = { ...newRegions[index], label };
      set({ regions: newRegions });
    }
  },

  rebuildRegions: () => {
    const { boundaries, regions } = get();
    if (boundaries.length === 0) {
      set({ regions: [] });
      return;
    }

    const newRegions: Region[] = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const start = boundaries[i];
      const end = boundaries[i + 1];

      // Preserve labels by finding existing region that starts at the same position
      // This maintains labels when boundaries are added/removed
      const existingRegion = regions.find((r) => Math.abs(r.start - start) < 0.00001);

      newRegions.push({
        start,
        end,
        label: existingRegion ? existingRegion.label : "unlabeled",
      });
    }
    set({ regions: newRegions });
  },

  setBoundaries: (boundaries) => {
    set({ boundaries: boundaries.sort((a, b) => a - b) });
    get().rebuildRegions();
  },

  clearBoundaries: () => {
    set({ boundaries: [], regions: [] });
  },

  setAnnotationTier: (tier) => {
    set({ annotationTier: tier });
  },

  markAsSaved: () => {
    const { regions, boundaries } = get();
    set({
      savedState: {
        regions: JSON.parse(JSON.stringify(regions)),
        boundaries: [...boundaries],
      },
    });
  },

  isDirty: () => {
    const { regions, boundaries, savedState } = get();
    if (!savedState) return false;

    // Check boundaries
    if (boundaries.length !== savedState.boundaries.length) return true;
    for (let i = 0; i < boundaries.length; i++) {
      if (Math.abs(boundaries[i] - savedState.boundaries[i]) > 0.00001) return true;
    }

    // Check regions
    if (regions.length !== savedState.regions.length) return true;
    for (let i = 0; i < regions.length; i++) {
      if (regions[i].label !== savedState.regions[i].label) return true;
    }

    return false;
  },

  reset: () =>
    set({
      boundaries: [],
      regions: [],
      savedState: null,
      annotationTier: null,
    }),
}));
