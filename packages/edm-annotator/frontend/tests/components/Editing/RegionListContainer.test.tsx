import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RegionListContainer } from "@/components/Editing/RegionListContainer";
import { useStructureStore, useTrackStore, useUIStore, useTempoStore, useWaveformStore } from "@/stores";
import { trackService } from "@/services/api";

vi.mock("@/services/api", () => ({
  trackService: {
    loadGeneratedAnnotation: vi.fn(),
    saveAnnotation: vi.fn(),
  },
}));

// Mock RegionList component
vi.mock("@/components/Editing/RegionList", () => ({
  RegionList: () => <div data-testid="region-list">Region List</div>,
}));

// Mock SaveButton component
vi.mock("@/components/Editing/SaveButton", () => ({
  SaveButton: () => <button>Save</button>,
}));

describe("RegionListContainer", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useTrackStore.getState().reset();
    useUIStore.getState().reset();
    useTempoStore.getState().reset();
    useWaveformStore.getState().reset();

    useWaveformStore.setState({ duration: 180 });

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Header and Tier Indicator", () => {
    it("shows 'Generated' badge when annotation tier is 2", () => {
      useStructureStore.setState({
        annotationTier: 2,
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      render(<RegionListContainer />);

      expect(screen.getByText("Generated")).toBeInTheDocument();
    });

    it("does not show badge when annotation tier is 1 (reference)", () => {
      useStructureStore.setState({
        annotationTier: 1,
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      render(<RegionListContainer />);

      expect(screen.queryByText("Generated")).not.toBeInTheDocument();
    });

    it("does not show badge when annotation tier is null", () => {
      useStructureStore.setState({
        annotationTier: null,
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      render(<RegionListContainer />);

      expect(screen.queryByText("Generated")).not.toBeInTheDocument();
    });
  });

  describe("Load Generated Button", () => {
    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<RegionListContainer />);

      const button = screen.getByText("Load Generated").closest("button");
      expect(button).toBeDisabled();
    });

    it("is enabled when track is loaded", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });

      render(<RegionListContainer />);

      const button = screen.getByText("Load Generated").closest("button");
      expect(button).not.toBeDisabled();
    });

    it("loads generated annotation with boundaries", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
          { time: 20, label: "breakdown" },
        ],
        bpm: 128,
        downbeat: 0.5,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        const { boundaries } = useStructureStore.getState();
        // Should include track duration as final boundary
        expect(boundaries).toEqual([0, 10, 20, 180]);
      });
    });

    it("sets region labels from loaded annotation", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
        ],
        bpm: 128,
        downbeat: 0,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        const { regions } = useStructureStore.getState();
        expect(regions[0].label).toBe("intro");
        expect(regions[1].label).toBe("buildup");
      });
    });

    it("sets annotation tier to 2 (generated)", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [{ time: 0, label: "intro" }],
        bpm: 128,
        downbeat: 0,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        const { annotationTier } = useStructureStore.getState();
        expect(annotationTier).toBe(2);
      });
    });

    it("updates BPM and downbeat from loaded annotation", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [{ time: 0, label: "intro" }],
        bpm: 140,
        downbeat: 0.75,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      const setBPMSpy = vi.spyOn(useTempoStore.getState(), "setBPM");
      const setDownbeatSpy = vi.spyOn(useTempoStore.getState(), "setDownbeat");

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        expect(setBPMSpy).toHaveBeenCalledWith(140);
        expect(setDownbeatSpy).toHaveBeenCalledWith(0.75);
      });
    });

    it("marks state as saved after loading", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [{ time: 0, label: "intro" }],
        bpm: 128,
        downbeat: 0,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      const markAsSavedSpy = vi.spyOn(useStructureStore.getState(), "markAsSaved");

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        expect(markAsSavedSpy).toHaveBeenCalled();
      });
    });

    it("shows success status message", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
          { time: 20, label: "breakdown" },
        ],
        bpm: 128,
        downbeat: 0,
      });

      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        expect(showStatusSpy).toHaveBeenCalledWith(
          expect.stringContaining("Loaded 3 boundaries")
        );
      });
    });
  });

  describe("Unsaved Changes Confirmation - Load", () => {
    it("shows confirmation dialog when loading with unsaved changes", async () => {
      const user = userEvent.setup();

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        expect(screen.getByText("Unsaved Changes")).toBeInTheDocument();
        expect(screen.getByText(/Loading generated boundaries will discard/)).toBeInTheDocument();
      });
    });

    it("loads when confirmed", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [{ time: 0, label: "intro" }],
        bpm: 128,
        downbeat: 0,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));
      await user.click(screen.getByText("Load Anyway"));

      await waitFor(() => {
        expect(trackService.loadGeneratedAnnotation).toHaveBeenCalled();
      });
    });

    it("cancels load when cancelled", async () => {
      const user = userEvent.setup();

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));
      await user.click(screen.getByText("Cancel"));

      expect(trackService.loadGeneratedAnnotation).not.toHaveBeenCalled();
    });

    it("does not show confirmation when no unsaved changes", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockResolvedValue({
        boundaries: [{ time: 0, label: "intro" }],
        bpm: 128,
        downbeat: 0,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      // Should load directly without confirmation
      await waitFor(() => {
        expect(trackService.loadGeneratedAnnotation).toHaveBeenCalled();
      });

      expect(screen.queryByText("Unsaved Changes")).not.toBeInTheDocument();
    });
  });

  describe("Clear Button", () => {
    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<RegionListContainer />);

      const button = screen.getByText("Clear").closest("button");
      expect(button).toBeDisabled();
    });

    it("is disabled when no regions exist", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({ regions: [] });

      render(<RegionListContainer />);

      const button = screen.getByText("Clear").closest("button");
      expect(button).toBeDisabled();
    });

    it("is enabled when track loaded with regions", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      render(<RegionListContainer />);

      const button = screen.getByText("Clear").closest("button");
      expect(button).not.toBeDisabled();
    });

    it("shows confirmation dialog when clearing with unsaved changes", async () => {
      const user = userEvent.setup();

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Clear"));

      await waitFor(() => {
        expect(screen.getByText("Unsaved Changes")).toBeInTheDocument();
        expect(screen.getByText(/Clearing boundaries will discard/)).toBeInTheDocument();
      });
    });

    it("clears boundaries when confirmed", async () => {
      const user = userEvent.setup();

      const clearBoundariesSpy = vi.spyOn(useStructureStore.getState(), "clearBoundaries");

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Clear"));
      await user.click(screen.getByText("Clear Anyway"));

      await waitFor(() => {
        expect(clearBoundariesSpy).toHaveBeenCalled();
      });
    });
  });

  describe("Error Handling", () => {
    it("shows error message when load fails", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.loadGeneratedAnnotation).mockRejectedValue(
        new Error("Network error")
      );

      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useWaveformStore.setState({ duration: 180 });

      render(<RegionListContainer />);

      await user.click(screen.getByText("Load Generated"));

      await waitFor(() => {
        expect(showStatusSpy).toHaveBeenCalledWith(
          expect.stringContaining("Error loading generated annotation")
        );
      });
    });
  });

  describe("Empty State", () => {
    it("shows empty state message when no regions", () => {
      useStructureStore.setState({ regions: [] });

      render(<RegionListContainer />);

      expect(screen.getByText("No annotations defined")).toBeInTheDocument();
    });

    it("shows region list when regions exist", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
      });

      render(<RegionListContainer />);

      expect(screen.getByTestId("region-list")).toBeInTheDocument();
      expect(screen.queryByText("No annotations defined")).not.toBeInTheDocument();
    });
  });
});
