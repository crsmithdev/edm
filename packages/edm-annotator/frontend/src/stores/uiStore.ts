import { create } from "zustand";

interface UIState {
  // State
  isDragging: boolean;
  dragStartX: number;
  dragStartViewport: number;
  quantizeEnabled: boolean;
  jumpMode: "beats" | "bars";
  statusMessage: string | null;

  // Actions
  setDragging: (dragging: boolean, startX?: number, startViewport?: number) => void;
  toggleQuantize: () => void;
  toggleJumpMode: () => void;
  showStatus: (message: string) => void;
  clearStatus: () => void;
  reset: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  // Initial state
  isDragging: false,
  dragStartX: 0,
  dragStartViewport: 0,
  quantizeEnabled: true,
  jumpMode: "bars",
  statusMessage: null,

  // Actions
  setDragging: (dragging, startX = 0, startViewport = 0) =>
    set({
      isDragging: dragging,
      dragStartX: startX,
      dragStartViewport: startViewport,
    }),

  toggleQuantize: () =>
    set((state) => ({
      quantizeEnabled: !state.quantizeEnabled,
    })),

  toggleJumpMode: () =>
    set((state) => ({
      jumpMode: state.jumpMode === "beats" ? "bars" : "beats",
    })),

  showStatus: (message) => set({ statusMessage: message }),

  clearStatus: () => set({ statusMessage: null }),

  reset: () =>
    set({
      isDragging: false,
      dragStartX: 0,
      dragStartViewport: 0,
      statusMessage: null,
    }),
}));
