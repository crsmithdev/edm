import { describe, it, expect, beforeEach } from "vitest";
import { useUIStore } from "@/stores/uiStore";

describe("uiStore", () => {
  beforeEach(() => {
    // Reset store before each test
    const store = useUIStore.getState();
    store.reset();
    // Manually restore initial state for fields not reset by reset()
    useUIStore.setState({
      quantizeEnabled: true,
      jumpMode: "beats",
    });
  });

  describe("setDragging", () => {
    it("should start dragging with position and viewport", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 100, 50);

      const { isDragging, dragStartX, dragStartViewport } =
        useUIStore.getState();
      expect(isDragging).toBe(true);
      expect(dragStartX).toBe(100);
      expect(dragStartViewport).toBe(50);
    });

    it("should stop dragging", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 100, 50);
      setDragging(false);

      const { isDragging, dragStartX, dragStartViewport } =
        useUIStore.getState();
      expect(isDragging).toBe(false);
      expect(dragStartX).toBe(0); // Defaults to 0
      expect(dragStartViewport).toBe(0);
    });

    it("should handle dragging with only boolean parameter", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true);

      const { isDragging, dragStartX, dragStartViewport } =
        useUIStore.getState();
      expect(isDragging).toBe(true);
      expect(dragStartX).toBe(0);
      expect(dragStartViewport).toBe(0);
    });

    it("should update drag position while dragging", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 100, 50);
      setDragging(true, 150, 75);

      const { dragStartX, dragStartViewport } = useUIStore.getState();
      expect(dragStartX).toBe(150);
      expect(dragStartViewport).toBe(75);
    });

    it("should handle negative positions", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, -10, -5);

      const { dragStartX, dragStartViewport } = useUIStore.getState();
      expect(dragStartX).toBe(-10);
      expect(dragStartViewport).toBe(-5);
    });
  });

  describe("toggleQuantize", () => {
    it("should toggle quantize from enabled to disabled", () => {
      const { toggleQuantize } = useUIStore.getState();

      // Initial state is enabled (true)
      let { quantizeEnabled } = useUIStore.getState();
      expect(quantizeEnabled).toBe(true);

      toggleQuantize();

      quantizeEnabled = useUIStore.getState().quantizeEnabled;
      expect(quantizeEnabled).toBe(false);
    });

    it("should toggle quantize from disabled to enabled", () => {
      const { toggleQuantize } = useUIStore.getState();

      // Toggle to disabled
      toggleQuantize();
      // Toggle back to enabled
      toggleQuantize();

      const { quantizeEnabled } = useUIStore.getState();
      expect(quantizeEnabled).toBe(true);
    });

    it("should toggle multiple times", () => {
      const { toggleQuantize } = useUIStore.getState();

      const states: boolean[] = [];
      for (let i = 0; i < 5; i++) {
        toggleQuantize();
        states.push(useUIStore.getState().quantizeEnabled);
      }

      expect(states).toEqual([false, true, false, true, false]);
    });
  });

  describe("toggleJumpMode", () => {
    it("should toggle from beats to bars", () => {
      const { toggleJumpMode } = useUIStore.getState();

      // Initial state is "beats"
      let { jumpMode } = useUIStore.getState();
      expect(jumpMode).toBe("beats");

      toggleJumpMode();

      jumpMode = useUIStore.getState().jumpMode;
      expect(jumpMode).toBe("bars");
    });

    it("should toggle from bars to beats", () => {
      const { toggleJumpMode } = useUIStore.getState();

      // Toggle to bars
      toggleJumpMode();
      // Toggle back to beats
      toggleJumpMode();

      const { jumpMode } = useUIStore.getState();
      expect(jumpMode).toBe("beats");
    });

    it("should cycle between modes", () => {
      const { toggleJumpMode } = useUIStore.getState();

      const modes: string[] = [];
      for (let i = 0; i < 6; i++) {
        modes.push(useUIStore.getState().jumpMode);
        toggleJumpMode();
      }

      expect(modes).toEqual([
        "beats",
        "bars",
        "beats",
        "bars",
        "beats",
        "bars",
      ]);
    });
  });

  describe("showStatus", () => {
    it("should set status message", () => {
      const { showStatus } = useUIStore.getState();

      showStatus("Track loaded successfully");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Track loaded successfully");
    });

    it("should update status message", () => {
      const { showStatus } = useUIStore.getState();

      showStatus("Loading...");
      showStatus("Complete");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Complete");
    });

    it("should handle empty string", () => {
      const { showStatus } = useUIStore.getState();

      showStatus("");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("");
    });

    it("should handle long messages", () => {
      const { showStatus } = useUIStore.getState();

      const longMessage = "A".repeat(1000);
      showStatus(longMessage);

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe(longMessage);
    });
  });

  describe("clearStatus", () => {
    it("should clear status message", () => {
      const { showStatus, clearStatus } = useUIStore.getState();

      showStatus("Test message");
      clearStatus();

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe(null);
    });

    it("should handle clearing when already null", () => {
      const { clearStatus } = useUIStore.getState();

      clearStatus();

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe(null);
    });

    it("should allow setting message after clear", () => {
      const { showStatus, clearStatus } = useUIStore.getState();

      showStatus("First message");
      clearStatus();
      showStatus("Second message");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Second message");
    });
  });

  describe("reset", () => {
    it("should reset drag state", () => {
      const { setDragging, reset } = useUIStore.getState();

      setDragging(true, 100, 50);
      reset();

      const { isDragging, dragStartX, dragStartViewport } =
        useUIStore.getState();
      expect(isDragging).toBe(false);
      expect(dragStartX).toBe(0);
      expect(dragStartViewport).toBe(0);
    });

    it("should clear status message", () => {
      const { showStatus, reset } = useUIStore.getState();

      showStatus("Test message");
      reset();

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe(null);
    });

    it("should not reset quantize setting", () => {
      const { toggleQuantize, reset } = useUIStore.getState();

      toggleQuantize(); // Disable quantize
      reset();

      const { quantizeEnabled } = useUIStore.getState();
      expect(quantizeEnabled).toBe(false); // Not reset
    });

    it("should not reset jump mode", () => {
      const { toggleJumpMode, reset } = useUIStore.getState();

      toggleJumpMode(); // Switch to bars
      reset();

      const { jumpMode } = useUIStore.getState();
      expect(jumpMode).toBe("bars"); // Not reset
    });
  });

  describe("integration scenarios", () => {
    it("should handle drag workflow", () => {
      const { setDragging } = useUIStore.getState();

      // Start drag
      setDragging(true, 200, 100);
      let { isDragging } = useUIStore.getState();
      expect(isDragging).toBe(true);

      // Update during drag
      setDragging(true, 250, 125);
      const { dragStartX } = useUIStore.getState();
      expect(dragStartX).toBe(250);

      // End drag
      setDragging(false);
      isDragging = useUIStore.getState().isDragging;
      expect(isDragging).toBe(false);
    });

    it("should handle status message workflow", () => {
      const { showStatus, clearStatus } = useUIStore.getState();

      showStatus("Loading track...");
      let { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Loading track...");

      showStatus("Processing...");
      statusMessage = useUIStore.getState().statusMessage;
      expect(statusMessage).toBe("Processing...");

      clearStatus();
      statusMessage = useUIStore.getState().statusMessage;
      expect(statusMessage).toBe(null);
    });

    it("should handle multiple feature toggles", () => {
      const { toggleQuantize, toggleJumpMode } = useUIStore.getState();

      toggleQuantize();
      toggleJumpMode();

      const { quantizeEnabled, jumpMode } = useUIStore.getState();
      expect(quantizeEnabled).toBe(false);
      expect(jumpMode).toBe("bars");

      toggleQuantize();
      toggleJumpMode();

      const state = useUIStore.getState();
      expect(state.quantizeEnabled).toBe(true);
      expect(state.jumpMode).toBe("beats");
    });

    it("should maintain independent state for different features", () => {
      const {
        setDragging,
        toggleQuantize,
        toggleJumpMode,
        showStatus,
      } = useUIStore.getState();

      setDragging(true, 100, 50);
      toggleQuantize();
      toggleJumpMode();
      showStatus("Test");

      const {
        isDragging,
        quantizeEnabled,
        jumpMode,
        statusMessage,
      } = useUIStore.getState();

      expect(isDragging).toBe(true);
      expect(quantizeEnabled).toBe(false);
      expect(jumpMode).toBe("bars");
      expect(statusMessage).toBe("Test");
    });

    it("should handle rapid state changes", () => {
      const { toggleQuantize, showStatus } = useUIStore.getState();

      for (let i = 0; i < 10; i++) {
        toggleQuantize();
        showStatus(`Message ${i}`);
      }

      const { quantizeEnabled, statusMessage } = useUIStore.getState();
      expect(quantizeEnabled).toBe(true); // 10 toggles from true → back to true
      expect(statusMessage).toBe("Message 9");
    });
  });

  describe("edge cases", () => {
    it("should handle drag with zero coordinates", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 0, 0);

      const { isDragging, dragStartX, dragStartViewport } =
        useUIStore.getState();
      expect(isDragging).toBe(true);
      expect(dragStartX).toBe(0);
      expect(dragStartViewport).toBe(0);
    });

    it("should handle very large drag coordinates", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 999999, 999999);

      const { dragStartX, dragStartViewport } = useUIStore.getState();
      expect(dragStartX).toBe(999999);
      expect(dragStartViewport).toBe(999999);
    });

    it("should handle fractional drag coordinates", () => {
      const { setDragging } = useUIStore.getState();

      setDragging(true, 123.456, 78.901);

      const { dragStartX, dragStartViewport } = useUIStore.getState();
      expect(dragStartX).toBe(123.456);
      expect(dragStartViewport).toBe(78.901);
    });

    it("should handle status messages with special characters", () => {
      const { showStatus } = useUIStore.getState();

      showStatus("Track: [test.mp3] - 128 BPM @ 0.5s");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Track: [test.mp3] - 128 BPM @ 0.5s");
    });

    it("should handle Unicode in status messages", () => {
      const { showStatus } = useUIStore.getState();

      showStatus("Loading... ⏳");

      const { statusMessage } = useUIStore.getState();
      expect(statusMessage).toBe("Loading... ⏳");
    });
  });

  describe("default state", () => {
    it("should have correct initial state", () => {
      const state = useUIStore.getState();

      expect(state.isDragging).toBe(false);
      expect(state.dragStartX).toBe(0);
      expect(state.dragStartViewport).toBe(0);
      expect(state.quantizeEnabled).toBe(true);
      expect(state.jumpMode).toBe("beats");
      expect(state.statusMessage).toBe(null);
    });
  });
});
