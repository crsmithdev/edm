import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { userEvent } from "@testing-library/user-event";
import { NavigationControls } from "@/components/Transport/NavigationControls";
import { useAudioStore, useTempoStore, useUIStore } from "@/stores";

describe("NavigationControls", () => {
  beforeEach(() => {
    // Reset all stores
    useAudioStore.getState().reset();
    useTempoStore.getState().reset();
    useUIStore.getState().reset();

    // Set up basic tempo
    useTempoStore.setState({
      trackBPM: 120, // 2 seconds per bar (60/120 * 4)
      trackDownbeat: 0,
    });

    vi.clearAllMocks();
  });

  describe("Jump functionality", () => {
    it("jumps forward by 1 bar from current position", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      // Start at 10 seconds
      useAudioStore.setState({ currentTime: 10.0 });

      render(<NavigationControls />);

      // Get all buttons and find the forward +1 button (has "1" followed by icon, not icon followed by "1")
      const allButtons = screen.getAllByRole("button");
      // Forward buttons come after the mode button - button at index 6 is forward +1
      const forwardOneButton = allButtons[6];
      await user.click(forwardOneButton);

      // Should jump forward 1 bar (2 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(12.0);
    });

    it("jumps forward by 4 bars from current position", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 10.0 });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const forwardFourButton = allButtons[8]; // Forward +4 button
      await user.click(forwardFourButton);

      // Should jump forward 4 bars (8 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(18.0);
    });

    it("jumps forward by 16 bars from current position", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 10.0 });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const forwardSixteenButton = allButtons[10]; // Forward +16 button
      await user.click(forwardSixteenButton);

      // Should jump forward 16 bars (32 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(42.0);
    });

    it("jumps backward by 1 bar from current position", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 10.0 });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const backwardOneButton = allButtons[4]; // Backward -1 button
      await user.click(backwardOneButton);

      // Should jump backward 1 bar (2 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(8.0);
    });

    it("jumps backward by 8 bars from current position", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 20.0 });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const backwardEightButton = allButtons[1]; // Backward -8 button
      await user.click(backwardEightButton);

      // Should jump backward 8 bars (16 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(4.0);
    });

    it("clamps to 0 when jumping backward past start", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 5.0 });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const backwardEightButton = allButtons[1]; // Backward -8 button
      await user.click(backwardEightButton);

      // Would jump to -11, but should clamp to 0
      expect(seekSpy).toHaveBeenCalledWith(0);
    });

    it("uses actual current time, not stale closure value", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      // Start at 10 seconds
      useAudioStore.setState({ currentTime: 10.0 });

      const { rerender } = render(<NavigationControls />);

      // Simulate time passing (e.g., audio playing)
      useAudioStore.setState({ currentTime: 15.0 });
      rerender(<NavigationControls />);

      // Click forward button - should use current time (15.0), not initial (10.0)
      const allButtons = screen.getAllByRole("button");
      const forwardOneButton = allButtons[6]; // Forward +1 button
      await user.click(forwardOneButton);

      // Should jump from 15.0, not from 10.0
      expect(seekSpy).toHaveBeenCalledWith(17.0);
    });
  });

  describe("Jump mode", () => {
    it("defaults to bars mode", () => {
      render(<NavigationControls />);

      const modeButton = screen.getByRole("button", { name: "Bars" });
      expect(modeButton).toBeDefined();
    });

    it("toggles between bars and beats mode", async () => {
      const user = userEvent.setup();

      render(<NavigationControls />);

      // Should start as "Bars"
      let modeButton = screen.getByRole("button", { name: "Bars" });
      expect(modeButton).toBeDefined();

      // Click to toggle
      await user.click(modeButton);

      // Should now be "Beats"
      modeButton = screen.getByRole("button", { name: "Beats" });
      expect(modeButton).toBeDefined();

      // Click to toggle back
      await user.click(modeButton);

      // Should be "Bars" again
      modeButton = screen.getByRole("button", { name: "Bars" });
      expect(modeButton).toBeDefined();
    });

    it("jumps by beats when in beats mode", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 10.0 });
      useUIStore.setState({ jumpMode: "beats" });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const forwardOneButton = allButtons[6]; // Forward +1 button
      await user.click(forwardOneButton);

      // Should jump forward 1 beat (0.5 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(10.5);
    });

    it("jumps by bars when in bars mode", async () => {
      const user = userEvent.setup();
      const seekSpy = vi.spyOn(useAudioStore.getState(), "seek");

      useAudioStore.setState({ currentTime: 10.0 });
      useUIStore.setState({ jumpMode: "bars" });

      render(<NavigationControls />);

      const allButtons = screen.getAllByRole("button");
      const forwardOneButton = allButtons[6]; // Forward +1 button
      await user.click(forwardOneButton);

      // Should jump forward 1 bar (2 seconds at 120 BPM)
      expect(seekSpy).toHaveBeenCalledWith(12.0);
    });
  });
});
