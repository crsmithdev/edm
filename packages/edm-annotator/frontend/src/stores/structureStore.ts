import { create } from "zustand";
import type { Region, SectionLabel } from "@/types/structure";

interface StructureState {
  // State
  boundaries: number[];
  regions: Region[];

  // Actions
  addBoundary: (time: number) => void;
  removeBoundary: (time: number) => void;
  setRegionLabel: (index: number, label: SectionLabel) => void;
  rebuildRegions: () => void;
  setBoundaries: (boundaries: number[]) => void;
  reset: () => void;
}

export const useStructureStore = create<StructureState>((set, get) => ({
  // Initial state
  boundaries: [],
  regions: [],

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
    // Use looser tolerance for removal (user interaction)
    const newBoundaries = boundaries.filter((t) => Math.abs(t - time) > 0.01);
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
    const { boundaries } = get();
    if (boundaries.length === 0) {
      set({ regions: [] });
      return;
    }

    const newRegions: Region[] = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      newRegions.push({
        start: boundaries[i],
        end: boundaries[i + 1],
        label: "unlabeled",
      });
    }
    set({ regions: newRegions });
  },

  setBoundaries: (boundaries) => {
    set({ boundaries: boundaries.sort((a, b) => a - b) });
    get().rebuildRegions();
  },

  reset: () =>
    set({
      boundaries: [],
      regions: [],
    }),
}));
