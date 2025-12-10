import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EditingControls } from "@/components/Editing/EditingControls";
import { trackService } from "@/services/api";
import { useAudioStore, useStructureStore, useTempoStore, useUIStore, useTrackStore } from "@/stores";

// Mock the API service
vi.mock("@/services/api", () => ({
  trackService: {
    saveAnnotation: vi.fn(),
  },
}));

describe("EditingControls", () => {
  beforeEach(() => {
    // Reset all stores
    useAudioStore.getState().reset();
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useUIStore.getState().reset();
    useTrackStore.getState().reset();

    // Clear all mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Add Boundary Button", () => {
    it("renders add boundary button", () => {
      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Boundary/i });
      expect(addButton).toBeInTheDocument();
    });

    it("calls addBoundary when button is clicked", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 10.5 });

      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Boundary/i });
      addButton.click();

      expect(addBoundarySpy).toHaveBeenCalledWith(10.5);
    });

    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Boundary/i });
      expect(addButton).toBeDisabled();
    });

    it("is enabled when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Boundary/i });
      expect(addButton).toBeEnabled();
    });

    it("shows status message when boundary is added", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 10.5 });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Boundary/i });
      addButton.click();

      const uiState = useUIStore.getState();
      expect(uiState.statusMessage).toContain("Added boundary at 10.50s");
    });
  });

  describe("Set Downbeat Button", () => {
    it("renders set downbeat button", () => {
      render(<EditingControls />);

      const downbeatButton = screen.getByRole("button", { name: /Set Downbeat/i });
      expect(downbeatButton).toBeInTheDocument();
    });

    it("calls setDownbeat when button is clicked", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 5.25 });

      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByRole("button", { name: /Set Downbeat/i });
      downbeatButton.click();

      expect(setDownbeatSpy).toHaveBeenCalledWith(5.25);
    });

    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      const downbeatButton = screen.getByRole("button", { name: /Set Downbeat/i });
      expect(downbeatButton).toBeDisabled();
    });

    it("shows status message when downbeat is set", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 5.25 });

      render(<EditingControls />);

      const downbeatButton = screen.getByRole("button", { name: /Set Downbeat/i });
      downbeatButton.click();

      const uiState = useUIStore.getState();
      expect(uiState.statusMessage).toContain("Downbeat set to 5.25s");
    });
  });

  describe("Quantize Toggle", () => {
    it("renders quantize toggle button", () => {
      render(<EditingControls />);

      const quantizeButton = screen.getByRole("button", { name: /Quantize/i });
      expect(quantizeButton).toBeInTheDocument();
    });

    it("updates state when quantize is toggled", () => {
      useUIStore.setState({ quantizeEnabled: true });

      render(<EditingControls />);

      const quantizeButton = screen.getByRole("button", { name: /Quantize/i });
      quantizeButton.click();

      const uiState = useUIStore.getState();
      expect(uiState.quantizeEnabled).toBe(false);
    });
  });
});
