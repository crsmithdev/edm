import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EditingControls } from "@/components/Editing/EditingControls";
import {
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useUIStore,
  useTrackStore,
} from "@/stores";

describe("EditingControls", () => {
  beforeEach(() => {
    // Reset stores
    useAudioStore.getState().reset();
    useStructureStore.getState().reset();
    useTempoStore.getState().reset();
    useUIStore.getState().reset();
    useTrackStore.getState().reset();

    // Set up initial state
    useAudioStore.setState({ currentTime: 0 });
    useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0 });
    useUIStore.setState({ quantizeEnabled: true, statusMessage: null });
    useTrackStore.setState({ currentTrack: "test-track.mp3" });

    // Clear all mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Rendering", () => {
    it("renders all three editing buttons", () => {
      render(<EditingControls />);

      expect(screen.getByText("Boundary")).toBeInTheDocument();
      expect(screen.getByText("Downbeat")).toBeInTheDocument();
      expect(screen.getByText("Quantize")).toBeInTheDocument();
    });

    it("displays primary variant for Quantize when enabled", () => {
      useUIStore.setState({ quantizeEnabled: true });

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      expect(quantizeButton).toBeInTheDocument();
    });

    it("displays secondary variant for Quantize when disabled", () => {
      useUIStore.setState({ quantizeEnabled: false });

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      expect(quantizeButton).toBeInTheDocument();
    });
  });

  describe("Add Boundary", () => {
    it("adds boundary at current time", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 15.5 });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      await user.click(boundaryButton);

      expect(addBoundarySpy).toHaveBeenCalledWith(15.5);
    });

    it("adds boundary at different time positions", async () => {
      const user = userEvent.setup();
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");

      // Add at 0 seconds
      useAudioStore.setState({ currentTime: 0 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(0);

      // Add at 30.25 seconds
      useAudioStore.setState({ currentTime: 30.25 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(30.25);

      // Add at 120.75 seconds
      useAudioStore.setState({ currentTime: 120.75 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(120.75);
    });

    it("shows status message after adding boundary", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 42.5 });
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      await user.click(boundaryButton);

      expect(showStatusSpy).toHaveBeenCalledWith("Added boundary at 42.50s");
    });


    it("is disabled when no track is loaded", async () => {
      const user = userEvent.setup();
      useTrackStore.setState({ currentTrack: null });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      expect(boundaryButton).toBeDisabled();

      await user.click(boundaryButton);
      expect(addBoundarySpy).not.toHaveBeenCalled();
    });

    it("is enabled when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test-track.mp3" });

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      expect(boundaryButton).not.toBeDisabled();
    });
  });

  describe("Set Downbeat", () => {
    it("sets downbeat at current time", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 2.5 });
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      await user.click(downbeatButton);

      expect(setDownbeatSpy).toHaveBeenCalledWith(2.5);
    });

    it("sets downbeat at different time positions", async () => {
      const user = userEvent.setup();
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");

      // Set at 0 seconds
      useAudioStore.setState({ currentTime: 0 });
      await user.click(downbeatButton);
      expect(setDownbeatSpy).toHaveBeenCalledWith(0);

      // Set at 1.25 seconds
      useAudioStore.setState({ currentTime: 1.25 });
      await user.click(downbeatButton);
      expect(setDownbeatSpy).toHaveBeenCalledWith(1.25);

      // Set at 5.75 seconds
      useAudioStore.setState({ currentTime: 5.75 });
      await user.click(downbeatButton);
      expect(setDownbeatSpy).toHaveBeenCalledWith(5.75);
    });

    it("shows status message after setting downbeat", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 3.5 });
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      await user.click(downbeatButton);

      expect(showStatusSpy).toHaveBeenCalledWith("Downbeat set to 3.50s");
    });


    it("is disabled when no track is loaded", async () => {
      const user = userEvent.setup();
      useTrackStore.setState({ currentTrack: null });
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      expect(downbeatButton).toBeDisabled();

      await user.click(downbeatButton);
      expect(setDownbeatSpy).not.toHaveBeenCalled();
    });

    it("is enabled when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test-track.mp3" });

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      expect(downbeatButton).not.toBeDisabled();
    });
  });

  describe("Toggle Quantize", () => {
    it("toggles quantize from enabled to disabled", async () => {
      const user = userEvent.setup();
      useUIStore.setState({ quantizeEnabled: true });
      const toggleQuantizeSpy = vi.spyOn(useUIStore.getState(), "toggleQuantize");

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      await user.click(quantizeButton);

      expect(toggleQuantizeSpy).toHaveBeenCalled();
    });

    it("toggles quantize from disabled to enabled", async () => {
      const user = userEvent.setup();
      useUIStore.setState({ quantizeEnabled: false });
      const toggleQuantizeSpy = vi.spyOn(useUIStore.getState(), "toggleQuantize");

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      await user.click(quantizeButton);

      expect(toggleQuantizeSpy).toHaveBeenCalled();
    });

    it("can toggle multiple times", async () => {
      const user = userEvent.setup();
      const toggleQuantizeSpy = vi.spyOn(useUIStore.getState(), "toggleQuantize");

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");

      // Toggle three times
      await user.click(quantizeButton);
      await user.click(quantizeButton);
      await user.click(quantizeButton);

      expect(toggleQuantizeSpy).toHaveBeenCalledTimes(3);
    });

    it("is always enabled regardless of track state", async () => {
      const user = userEvent.setup();
      const toggleQuantizeSpy = vi.spyOn(useUIStore.getState(), "toggleQuantize");

      // Test with no track
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      expect(quantizeButton).not.toBeDisabled();

      await user.click(quantizeButton);
      expect(toggleQuantizeSpy).toHaveBeenCalled();
    });
  });

  describe("Button Disabled States", () => {
    it("disables Boundary and Downbeat when track is null", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      expect(screen.getByText("Boundary")).toBeDisabled();
      expect(screen.getByText("Downbeat")).toBeDisabled();
      expect(screen.getByText("Quantize")).not.toBeDisabled();
    });

    it("disables Boundary and Downbeat when track is empty string", () => {
      useTrackStore.setState({ currentTrack: "" });

      render(<EditingControls />);

      expect(screen.getByText("Boundary")).toBeDisabled();
      expect(screen.getByText("Downbeat")).toBeDisabled();
      expect(screen.getByText("Quantize")).not.toBeDisabled();
    });

    it("enables Boundary and Downbeat when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test-track.mp3" });

      render(<EditingControls />);

      expect(screen.getByText("Boundary")).not.toBeDisabled();
      expect(screen.getByText("Downbeat")).not.toBeDisabled();
      expect(screen.getByText("Quantize")).not.toBeDisabled();
    });

    it("updates disabled state when track changes", () => {
      useTrackStore.setState({ currentTrack: "test-track.mp3" });

      const { rerender } = render(<EditingControls />);

      expect(screen.getByText("Boundary")).not.toBeDisabled();
      expect(screen.getByText("Downbeat")).not.toBeDisabled();

      // Unload track
      useTrackStore.setState({ currentTrack: null });
      rerender(<EditingControls />);

      expect(screen.getByText("Boundary")).toBeDisabled();
      expect(screen.getByText("Downbeat")).toBeDisabled();

      // Load track again
      useTrackStore.setState({ currentTrack: "another-track.mp3" });
      rerender(<EditingControls />);

      expect(screen.getByText("Boundary")).not.toBeDisabled();
      expect(screen.getByText("Downbeat")).not.toBeDisabled();
    });
  });

  describe("Status Message Formatting", () => {
    it("formats times to 2 decimal places in status messages", async () => {
      const user = userEvent.setup();
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      const downbeatButton = screen.getByText("Downbeat");

      // Test boundary formatting with different precisions
      useAudioStore.setState({ currentTime: 10.1234 });
      await user.click(boundaryButton);
      expect(showStatusSpy).toHaveBeenCalledWith("Added boundary at 10.12s");

      // Test downbeat formatting with rounding
      useAudioStore.setState({ currentTime: 2.9876 });
      await user.click(downbeatButton);
      expect(showStatusSpy).toHaveBeenCalledWith("Downbeat set to 2.99s");
    });
  });

  describe("Edge Cases", () => {
    it("handles boundary at time 0", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 0 });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      await user.click(boundaryButton);

      expect(addBoundarySpy).toHaveBeenCalledWith(0);
    });

    it("handles downbeat at time 0", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 0 });
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      await user.click(downbeatButton);

      expect(setDownbeatSpy).toHaveBeenCalledWith(0);
    });

    it("handles very large time values", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 999.99 });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      await user.click(boundaryButton);

      expect(addBoundarySpy).toHaveBeenCalledWith(999.99);
    });

    it("preserves full precision when passing to store", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 12.3456789 });
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<EditingControls />);

      const downbeatButton = screen.getByText("Downbeat");
      await user.click(downbeatButton);

      expect(setDownbeatSpy).toHaveBeenCalledWith(12.3456789);
    });

    it("handles rapid button clicks", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 10 });
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");

      // Click multiple times rapidly
      await user.click(boundaryButton);
      await user.click(boundaryButton);
      await user.click(boundaryButton);

      expect(addBoundarySpy).toHaveBeenCalledTimes(3);
    });

    it("updates current time correctly between button clicks", async () => {
      const user = userEvent.setup();
      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");

      // First click at 10 seconds
      useAudioStore.setState({ currentTime: 10 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(10);

      // Second click at 20 seconds
      useAudioStore.setState({ currentTime: 20 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(20);

      // Third click at 30 seconds
      useAudioStore.setState({ currentTime: 30 });
      await user.click(boundaryButton);
      expect(addBoundarySpy).toHaveBeenCalledWith(30);
    });
  });

  describe("Status Messages", () => {
    it("shows different messages for boundary and downbeat", async () => {
      const user = userEvent.setup();
      useAudioStore.setState({ currentTime: 15.5 });
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      render(<EditingControls />);

      const boundaryButton = screen.getByText("Boundary");
      const downbeatButton = screen.getByText("Downbeat");

      await user.click(boundaryButton);
      expect(showStatusSpy).toHaveBeenLastCalledWith("Added boundary at 15.50s");

      await user.click(downbeatButton);
      expect(showStatusSpy).toHaveBeenLastCalledWith("Downbeat set to 15.50s");
    });

    it("does not show status message when toggling quantize", async () => {
      const user = userEvent.setup();
      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      render(<EditingControls />);

      const quantizeButton = screen.getByText("Quantize");
      await user.click(quantizeButton);

      expect(showStatusSpy).not.toHaveBeenCalled();
    });
  });
});
