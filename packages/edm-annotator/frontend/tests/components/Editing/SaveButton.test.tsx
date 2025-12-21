import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SaveButton } from "@/components/Editing/SaveButton";
import { useStructureStore, useTrackStore, useUIStore, useTempoStore } from "@/stores";
import { trackService } from "@/services/api";

// Mock the API service
vi.mock("@/services/api", () => ({
  trackService: {
    saveAnnotation: vi.fn(),
  },
}));

describe("SaveButton", () => {
  beforeEach(() => {
    useStructureStore.getState().reset();
    useTrackStore.getState().reset();
    useUIStore.getState().reset();
    useTempoStore.getState().reset();

    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Button State", () => {
    it("is disabled when no track is loaded", () => {
      useTrackStore.setState({ currentTrack: null });

      render(<SaveButton />);

      const button = screen.getByText("Save").closest("button");
      expect(button).toBeDisabled();
    });

    it("is disabled when no changes have been made", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<SaveButton />);

      const button = screen.getByText("Save").closest("button");
      expect(button).toBeDisabled();
    });

    it("is enabled when there are unsaved changes", () => {
      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<SaveButton />);

      const button = screen.getByText("Save").closest("button");
      expect(button).not.toBeDisabled();
    });

    it("shows loading state while saving", async () => {
      const user = userEvent.setup();

      // Mock a slow save
      vi.mocked(trackService.saveAnnotation).mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({
          success: true,
          output: "annotations/test.jams",
          boundaries_count: 1,
        }), 100))
      );

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "buildup" }],
        boundaries: [0, 10],
        savedState: {
          regions: [{ start: 0, end: 10, label: "intro" }],
          boundaries: [0, 10],
        },
      });

      render(<SaveButton />);

      const button = screen.getByText("Save").closest("button");
      await user.click(button!);

      // Button should be disabled while saving
      expect(button).toBeDisabled();
    });
  });

  describe("Save Operation", () => {
    it("saves annotation with correct data", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 2,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
        boundaries: [0, 10, 20],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalledWith({
          filename: "test.mp3",
          bpm: 0,
          downbeat: 0,
          boundaries: [
            { time: 0, label: "intro" },
            { time: 10, label: "buildup" },
          ],
        });
      });
    });

    it("sets annotation tier to 1 (reference) after successful save", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 1,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [],
          boundaries: [],
        },
        annotationTier: 2, // Was generated
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        const { annotationTier } = useStructureStore.getState();
        expect(annotationTier).toBe(1); // Now reference
      });
    });

    it("marks state as saved after successful save", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 1,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        const { savedState } = useStructureStore.getState();
        expect(savedState).not.toBeNull();
        expect(savedState?.regions).toEqual([{ start: 0, end: 10, label: "intro" }]);
      });
    });

    it("shows success status message", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 2,
      });

      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
        boundaries: [0, 10, 20],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(showStatusSpy).toHaveBeenCalledWith(
          expect.stringContaining("Saved 2 boundaries")
        );
      });
    });
  });

  describe("Error Handling", () => {
    it("shows error status message on save failure", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockRejectedValue(
        new Error("Network error")
      );

      const showStatusSpy = vi.spyOn(useUIStore.getState(), "showStatus");

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(showStatusSpy).toHaveBeenCalledWith(
          expect.stringContaining("Error saving annotation")
        );
      });
    });

    it("does not mark as saved on error", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockRejectedValue(
        new Error("Network error")
      );

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalled();
      });

      const { savedState } = useStructureStore.getState();
      // savedState should still be the old empty state, not updated
      expect(savedState?.regions).toEqual([]);
    });

    it("does not set tier to 1 on error", async () => {
      const user = userEvent.setup();

      // Clear and reset mock to ensure clean state
      vi.clearAllMocks();
      vi.mocked(trackService.saveAnnotation).mockReset();
      vi.mocked(trackService.saveAnnotation).mockRejectedValue(
        new Error("Network error")
      );

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
        savedState: {
          regions: [],
          boundaries: [],
        },
        annotationTier: 2,
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalled();
      });

      const { annotationTier } = useStructureStore.getState();
      expect(annotationTier).toBe(2); // Unchanged
    });
  });

  describe("Boundary Format", () => {
    it("excludes final boundary (track end) from saved data", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 2,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 180, label: "buildup" },
        ],
        boundaries: [0, 10, 180], // 180 is track end
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalledWith({
          filename: "test.mp3",
          bpm: 0,
          downbeat: 0,
          boundaries: [
            { time: 0, label: "intro" },
            { time: 10, label: "buildup" },
            // Note: 180 is NOT included (it's the track end)
          ],
        });
      });
    });

    it("saves correct labels for each boundary", async () => {
      const user = userEvent.setup();

      vi.mocked(trackService.saveAnnotation).mockResolvedValue({
        success: true,
        output: "annotations/test.jams",
        boundaries_count: 4,
      });

      useTrackStore.setState({ currentTrack: "test.mp3" });
      useStructureStore.setState({
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
          { start: 20, end: 30, label: "breakdown" },
          { start: 30, end: 40, label: "breakdown-buildup" },
        ],
        boundaries: [0, 10, 20, 30, 40],
        savedState: {
          regions: [],
          boundaries: [],
        },
      });

      render(<SaveButton />);

      await user.click(screen.getByText("Save"));

      await waitFor(() => {
        expect(trackService.saveAnnotation).toHaveBeenCalledWith({
          filename: "test.mp3",
          bpm: 0,
          downbeat: 0,
          boundaries: [
            { time: 0, label: "intro" },
            { time: 10, label: "buildup" },
            { time: 20, label: "breakdown" },
            { time: 30, label: "breakdown-buildup" },
          ],
        });
      });
    });
  });
});
