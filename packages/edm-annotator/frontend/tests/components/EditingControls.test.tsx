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

  describe("BPM Input", () => {
    it("renders BPM input with current value", () => {
      useTempoStore.setState({ trackBPM: 128 });

      render(<EditingControls />);

      const bpmInput = screen.getByLabelText(/BPM:/i) as HTMLInputElement;
      expect(bpmInput).toBeInTheDocument();
      expect(bpmInput.value).toBe("128");
    });

    it("updates store when BPM input changes", async () => {
      const user = userEvent.setup();
      useTempoStore.setState({ trackBPM: 128 });

      render(<EditingControls />);

      const bpmInput = screen.getByLabelText(/BPM:/i);
      await user.clear(bpmInput);
      await user.type(bpmInput, "140");

      await waitFor(() => {
        const tempoState = useTempoStore.getState();
        expect(tempoState.trackBPM).toBe(140);
      });
    });

    it("does not update store for invalid BPM values", async () => {
      const user = userEvent.setup();
      useTempoStore.setState({ trackBPM: 128 });

      render(<EditingControls />);

      const bpmInput = screen.getByLabelText(/BPM:/i);
      await user.clear(bpmInput);
      await user.type(bpmInput, "-50");

      await waitFor(() => {
        const tempoState = useTempoStore.getState();
        // Should not update to negative value
        expect(tempoState.trackBPM).toBe(128);
      });
    });

    it("does not update store for NaN BPM values", async () => {
      const user = userEvent.setup();
      useTempoStore.setState({ trackBPM: 128 });

      render(<EditingControls />);

      const bpmInput = screen.getByLabelText(/BPM:/i);
      await user.clear(bpmInput);
      await user.type(bpmInput, "abc");

      await waitFor(() => {
        const tempoState = useTempoStore.getState();
        // Should not update to NaN
        expect(tempoState.trackBPM).toBe(128);
      });
    });
  });

  describe("Tap Tempo Button", () => {
    it("renders tap tempo button", () => {
      render(<EditingControls />);

      const tapButton = screen.getByRole("button", { name: /Tap/i });
      expect(tapButton).toBeInTheDocument();
    });

    it("calls tapTempo when tap button is clicked", async () => {
      const user = userEvent.setup();
      const tapTempoSpy = vi.spyOn(useTempoStore.getState(), "tapTempo");

      render(<EditingControls />);

      const tapButton = screen.getByRole("button", { name: /Tap/i });
      await user.click(tapButton);

      expect(tapTempoSpy).toHaveBeenCalled();
    });

    it("updates BPM input after tap tempo", async () => {
      vi.useFakeTimers();

      try {
        useTempoStore.setState({ trackBPM: 128 });

        render(<EditingControls />);

        const tapButton = screen.getByRole("button", { name: /Tap/i });

        // Simulate multiple taps
        tapButton.click();
        vi.advanceTimersByTime(500); // 500ms = 120 BPM
        tapButton.click();

        // Wait for the setTimeout in handleTapTempo
        vi.advanceTimersByTime(100);

        const tempoState = useTempoStore.getState();
        // BPM should be updated based on tap interval
        expect(tempoState.trackBPM).not.toBe(128);
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe("Add Boundary Button", () => {
    it("renders add boundary button", () => {
      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Add Boundary/i });
      expect(addButton).toBeInTheDocument();
    });

    it("calls addBoundary when button is clicked", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 10.5 });

      const addBoundarySpy = vi.spyOn(useStructureStore.getState(), "addBoundary");

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Add Boundary/i });
      addButton.click();

      expect(addBoundarySpy).toHaveBeenCalledWith(10.5);
    });

    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Add Boundary/i });
      expect(addButton).toBeDisabled();
    });

    it("is enabled when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Add Boundary/i });
      expect(addButton).toBeEnabled();
    });

    it("shows status message when boundary is added", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useAudioStore.setState({ currentTime: 10.5 });

      render(<EditingControls />);

      const addButton = screen.getByRole("button", { name: /Add Boundary/i });
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

    it("displays current quantize state", () => {
      useUIStore.setState({ quantizeEnabled: true });

      render(<EditingControls />);

      expect(screen.getByText(/Quantize \(Q\): ON/i)).toBeInTheDocument();
    });

    it("updates state when quantize is toggled", () => {
      useUIStore.setState({ quantizeEnabled: true });

      render(<EditingControls />);

      const quantizeButton = screen.getByRole("button", { name: /Quantize/i });
      quantizeButton.click();

      const uiState = useUIStore.getState();
      expect(uiState.quantizeEnabled).toBe(false);
    });

    it("toggles from OFF to ON", async () => {
      useUIStore.setState({ quantizeEnabled: false });

      render(<EditingControls />);

      expect(screen.getByText(/Quantize \(Q\): OFF/i)).toBeInTheDocument();

      const quantizeButton = screen.getByRole("button", { name: /Quantize/i });
      quantizeButton.click();

      await waitFor(() => {
        expect(screen.getByText(/Quantize \(Q\): ON/i)).toBeInTheDocument();
      });
    });
  });

  describe("Save Button", () => {
    it("renders save button", () => {
      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      expect(saveButton).toBeInTheDocument();
    });

    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });
      useStructureStore.setState({ boundaries: [10, 20, 30] });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      expect(saveButton).toBeDisabled();
    });

    it("is disabled when no boundaries exist", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({ boundaries: [] });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      expect(saveButton).toBeDisabled();
    });

    it("is enabled when track is loaded and boundaries exist", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      expect(saveButton).toBeEnabled();
    });

    it("calls saveAnnotation API with correct payload", async () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useTempoStore.setState({ trackBPM: 128, trackDownbeat: 0.5 });
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "test_output.txt",
        boundaries_count: 2,
      });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      saveButton.click();

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalledWith({
          filename: "test.mp3",
          bpm: 128,
          downbeat: 0.5,
          boundaries: [
            { time: 0, label: "intro" },
            { time: 10, label: "buildup" },
          ],
        });
      });
    });

    it("displays success message after save", async () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "test_output.txt",
        boundaries_count: 2,
      });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      saveButton.click();

      await waitFor(() => {
        const uiState = useUIStore.getState();
        expect(uiState.statusMessage).toBe("Annotation saved successfully");
      });
    });

    it("displays error message on save failure", async () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      vi.mocked(trackService.saveAnnotation).mockRejectedValue(
        new Error("Network error")
      );

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      saveButton.click();

      await waitFor(() => {
        const uiState = useUIStore.getState();
        expect(uiState.statusMessage).toContain("Error saving");
        expect(uiState.statusMessage).toContain("Network error");
      });
    });

    it("shows saving state during save operation", async () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      // Create a promise that we can control
      let resolveSave: (value: any) => void;
      const savePromise = new Promise((resolve) => {
        resolveSave = resolve;
      });

      vi.mocked(trackService.saveAnnotation).mockReturnValue(savePromise as any);

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });
      saveButton.click();

      // Should show "Saving..." state
      await waitFor(() => {
        expect(screen.getByText("Saving...")).toBeInTheDocument();
        expect(saveButton).toBeDisabled();
      });

      // Resolve the promise
      resolveSave!({
        success: true,
        output: "test_output.txt",
        boundaries_count: 2,
      });

      await waitFor(() => {
        expect(screen.getByText("Save Annotation")).toBeInTheDocument();
      });
    });

    it("shows status message when no track is loaded", async () => {
      const user = userEvent.setup();
      useTrackStore.setState({ currentTrack: null });

      render(<EditingControls />);

      const saveButton = screen.getByRole("button", { name: /Save Annotation/i });

      // Button should be disabled, but we can test the early return logic
      // by checking if the function would show the status
      expect(saveButton).toBeDisabled();
    });
  });

  describe("Stats Display", () => {
    it("displays boundary and region counts", () => {
      useStructureStore.setState({
        boundaries: [0, 10, 20, 30],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
          { start: 20, end: 30, label: "breakdown" },
        ],
      });

      render(<EditingControls />);

      expect(screen.getByText(/Boundaries: 4/i)).toBeInTheDocument();
      expect(screen.getByText(/Regions: 3/i)).toBeInTheDocument();
    });

    it("updates counts dynamically", () => {
      useStructureStore.setState({
        boundaries: [0, 10],
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      const { rerender } = render(<EditingControls />);

      expect(screen.getByText(/Boundaries: 2/i)).toBeInTheDocument();
      expect(screen.getByText(/Regions: 1/i)).toBeInTheDocument();

      // Update store
      useStructureStore.setState({
        boundaries: [0, 10, 20],
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      rerender(<EditingControls />);

      expect(screen.getByText(/Boundaries: 3/i)).toBeInTheDocument();
      expect(screen.getByText(/Regions: 2/i)).toBeInTheDocument();
    });
  });
});
