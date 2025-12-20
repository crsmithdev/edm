import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, within, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { RegionList } from "@/components/Editing/RegionList";
import { useStructureStore, useAudioStore } from "@/stores";
import type { Region } from "@/types/structure";

describe("RegionList", () => {
  beforeEach(() => {
    // Reset stores
    useStructureStore.getState().reset();
    useAudioStore.getState().reset();

    // Clear all mocks
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("Empty State", () => {
    it("renders column headers when no regions exist", () => {
      useStructureStore.setState({ regions: [], boundaries: [] });

      render(<RegionList />);

      expect(screen.getByText("Time")).toBeInTheDocument();
      expect(screen.getByText("Bars")).toBeInTheDocument();
      expect(screen.getByText("Label")).toBeInTheDocument();
    });
  });

  describe("Region Rendering", () => {
    it("renders list of regions", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
        { start: 10, end: 20, label: "buildup" },
        { start: 20, end: 30, label: "breakdown" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      // Check that all regions are displayed
      expect(screen.getByText(/00:00\.000 - 00:10\.000/)).toBeInTheDocument();
      expect(screen.getByText(/00:10\.000 - 00:20\.000/)).toBeInTheDocument();
      expect(screen.getByText(/00:20\.000 - 00:30\.000/)).toBeInTheDocument();
    });

    it("displays region durations correctly", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 15.5, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:15\.500/)).toBeInTheDocument();
    });

    it("renders region with formatted timestamps", () => {
      const mockRegions: Region[] = [
        { start: 65, end: 125, label: "intro" }, // 1:05 - 2:05
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      expect(screen.getByText(/01:05\.000 - 02:05\.000/)).toBeInTheDocument();
    });

    it("displays correct label for each region", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
        { start: 10, end: 20, label: "buildup" },
        { start: 20, end: 30, label: "breakdown" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      // Check that label selectors have correct values
      const selects = screen.getAllByRole("combobox");
      expect((selects[0] as HTMLSelectElement).value).toBe("intro");
      expect((selects[1] as HTMLSelectElement).value).toBe("buildup");
      expect((selects[2] as HTMLSelectElement).value).toBe("breakdown");
    });
  });

  describe("Label Editing", () => {
    it("allows changing region label", async () => {
      const user = userEvent.setup();
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "unlabeled" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const setRegionLabelSpy = vi.spyOn(useStructureStore.getState(), "setRegionLabel");

      render(<RegionList />);

      const select = screen.getByRole("combobox");
      await user.selectOptions(select, "intro");

      expect(setRegionLabelSpy).toHaveBeenCalledWith(0, "intro");
    });

    it("displays all valid label options", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      const select = screen.getByRole("combobox");
      const options = within(select).getAllByRole("option");

      const optionValues = options.map((opt) => (opt as HTMLOptionElement).value);

      expect(optionValues).toEqual([
        "intro",
        "buildup",
        "breakdown",
        "breakbuild",
        "outro",
        "unlabeled",
      ]);
    });

    it("updates multiple regions independently", async () => {
      const user = userEvent.setup();
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
        { start: 10, end: 20, label: "buildup" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const setRegionLabelSpy = vi.spyOn(useStructureStore.getState(), "setRegionLabel");

      render(<RegionList />);

      const selects = screen.getAllByRole("combobox");

      // Change first region
      await user.selectOptions(selects[0], "breakdown");
      expect(setRegionLabelSpy).toHaveBeenCalledWith(0, "breakdown");

      // Change second region
      await user.selectOptions(selects[1], "outro");
      expect(setRegionLabelSpy).toHaveBeenCalledWith(1, "outro");
    });

    it("prevents click event propagation when changing label", async () => {
      const user = userEvent.setup();
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      render(<RegionList />);

      const select = screen.getByRole("combobox");

      // Clicking the select should not trigger seek
      await user.click(select);
      expect(seekSpy).not.toHaveBeenCalled();

      // Selecting an option should not trigger seek
      await user.selectOptions(select, "buildup");
      expect(seekSpy).not.toHaveBeenCalled();
    });
  });

  describe("Region Click Behavior", () => {
    it("seeks to region start when region is clicked", async () => {
      const user = userEvent.setup();
      const mockRegions: Region[] = [
        { start: 15.5, end: 30.2, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      render(<RegionList />);

      // Click on the region (not the select element)
      const regionText = screen.getByText(/00:15\.500 - 00:30\.200/);
      await user.click(regionText);

      expect(seekSpy).toHaveBeenCalledWith(15.5);
    });

    it("seeks to correct region when multiple regions exist", async () => {
      const user = userEvent.setup();
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
        { start: 10, end: 20, label: "buildup" },
        { start: 20, end: 30, label: "breakdown" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      render(<RegionList />);

      // Click on second region
      const region2Text = screen.getByText(/00:10\.000 - 00:20\.000/);
      await user.click(region2Text);

      expect(seekSpy).toHaveBeenCalledWith(10);

      // Click on third region
      const region3Text = screen.getByText(/00:20\.000 - 00:30\.000/);
      await user.click(region3Text);

      expect(seekSpy).toHaveBeenCalledWith(20);
    });
  });

  describe("Visual Styling", () => {
    it("applies hover effect to regions", () => {
      const mockRegions: Region[] = [
        { start: 10, end: 20, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });
      // Set currentTime outside the region bounds
      useAudioStore.setState({ currentTime: 0 });

      const { container } = render(<RegionList />);

      // Find the region container directly by looking for element with cursor: pointer
      const regionElement = container.querySelector('[style*="cursor: pointer"]') as HTMLElement;
      expect(regionElement).toBeInTheDocument();

      // Trigger mouseenter event
      fireEvent.mouseEnter(regionElement);

      // Should change background on hover (uses CSS variable)
      expect(regionElement.style.background).toBe("var(--bg-elevated)");

      // Trigger mouseleave event
      fireEvent.mouseLeave(regionElement);

      // Should revert to transparent
      expect(regionElement.style.background).toBe("transparent");
    });
  });

  describe("Edge Cases", () => {
    it("handles single region correctly", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 180, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 03:00\.000/)).toBeInTheDocument();
    });

    it("handles regions with very small durations", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 0.1, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:00\.100/)).toBeInTheDocument();
    });

    it("handles many regions with scrolling", () => {
      const mockRegions: Region[] = Array.from({ length: 20 }, (_, i) => ({
        start: i * 10,
        end: (i + 1) * 10,
        label: "intro" as const,
      }));

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      // All regions should be rendered
      const selects = screen.getAllByRole("combobox");
      expect(selects).toHaveLength(20);
    });

    it("handles regions at the end of track correctly", () => {
      const mockRegions: Region[] = [
        { start: 170, end: 180, label: "outro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      render(<RegionList />);

      expect(screen.getByText(/02:50\.000 - 03:00\.000/)).toBeInTheDocument();
      const select = screen.getByRole("combobox");
      expect((select as HTMLSelectElement).value).toBe("outro");
    });
  });

  describe("Dynamic Updates", () => {
    it("updates when regions change", () => {
      const mockRegions: Region[] = [
        { start: 0, end: 10, label: "intro" },
      ];

      useStructureStore.setState({ regions: mockRegions });

      const { rerender } = render(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:10\.000/)).toBeInTheDocument();

      // Update regions
      useStructureStore.setState({
        regions: [
          { start: 0, end: 10, label: "intro" },
          { start: 10, end: 20, label: "buildup" },
        ],
      });

      rerender(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:10\.000/)).toBeInTheDocument();
      expect(screen.getByText(/00:10\.000 - 00:20\.000/)).toBeInTheDocument();
    });

    it("transitions from empty to populated state", () => {
      useStructureStore.setState({ regions: [], boundaries: [] });

      const { rerender } = render(<RegionList />);

      expect(screen.getByText("Time")).toBeInTheDocument();

      // Add regions
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
      });

      rerender(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:10\.000/)).toBeInTheDocument();
    });

    it("transitions from populated to empty state", () => {
      useStructureStore.setState({
        regions: [{ start: 0, end: 10, label: "intro" }],
        boundaries: [0, 10],
      });

      const { rerender } = render(<RegionList />);

      expect(screen.getByText(/00:00\.000 - 00:10\.000/)).toBeInTheDocument();

      // Clear regions
      useStructureStore.setState({ regions: [], boundaries: [] });

      rerender(<RegionList />);

      expect(screen.getByText("Time")).toBeInTheDocument();
      expect(screen.queryByText(/00:00\.000/)).not.toBeInTheDocument();
    });
  });
});
