import { useCallback, useRef } from "react";
import {
  useWaveformStore,
  useAudioStore,
  useStructureStore,
  useTempoStore,
  useUIStore,
} from "@/stores";

const DRAG_THRESHOLD = 5; // pixels

/**
 * Handles mouse interaction with waveform (click, drag, zoom)
 */
export function useWaveformInteraction() {
  const dragStartPos = useRef({ x: 0, y: 0 });
  const isDraggingRef = useRef(false);

  const { viewportStart, viewportEnd, pan, zoom } = useWaveformStore();
  const { seek, setCuePoint } = useAudioStore();
  const { addBoundary } = useStructureStore();
  const { quantizeToBeat } = useTempoStore();
  const { quantizeEnabled, setDragging, showStatus } = useUIStore();

  /**
   * Convert mouse X position to time in seconds
   */
  const pixelToTime = useCallback(
    (x: number, width: number): number => {
      const viewportDuration = viewportEnd - viewportStart;
      const ratio = x / width;
      return viewportStart + ratio * viewportDuration;
    },
    [viewportStart, viewportEnd]
  );

  /**
   * Handle mouse down - start potential drag or prepare for click
   */
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      dragStartPos.current = { x: e.clientX, y: e.clientY };
      isDraggingRef.current = false;
      setDragging(true, e.clientX, viewportStart);
    },
    [setDragging, viewportStart]
  );

  /**
   * Handle mouse move - pan viewport if dragging
   */
  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!dragStartPos.current) return;

      const deltaX = e.clientX - dragStartPos.current.x;
      const distance = Math.sqrt(
        deltaX ** 2 + (e.clientY - dragStartPos.current.y) ** 2
      );

      // Check if we've moved enough to be considered a drag
      if (distance > DRAG_THRESHOLD) {
        isDraggingRef.current = true;

        // Calculate time delta based on pixel movement
        const width = e.currentTarget.clientWidth;
        const viewportDuration = viewportEnd - viewportStart;
        const deltaTime = (-deltaX / width) * viewportDuration;

        pan(deltaTime);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
      }
    },
    [viewportStart, viewportEnd, pan]
  );

  /**
   * Handle mouse up - execute click action or end drag
   */
  const handleMouseUp = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      setDragging(false);

      if (!isDraggingRef.current) {
        // This was a click, not a drag
        const rect = e.currentTarget.getBoundingClientRect();
        const x = e.clientX - rect.left;
        let time = pixelToTime(x, rect.width);

        // Apply quantization if enabled
        if (quantizeEnabled) {
          time = quantizeToBeat(time);
        }

        if (e.shiftKey) {
          // Shift+Click: add boundary
          addBoundary(time);
          showStatus(`Added boundary at ${time.toFixed(2)}s`);
        } else {
          // Regular click: set cue point
          setCuePoint(time);
          seek(time);
          showStatus(`Cue set to ${time.toFixed(2)}s`);
        }
      }

      isDraggingRef.current = false;
      dragStartPos.current = { x: 0, y: 0 };
    },
    [
      setDragging,
      pixelToTime,
      quantizeEnabled,
      quantizeToBeat,
      addBoundary,
      setCuePoint,
      seek,
      showStatus,
    ]
  );

  /**
   * Handle wheel - zoom in/out
   */
  const handleWheel = useCallback(
    (e: React.WheelEvent<HTMLDivElement>) => {
      e.preventDefault();

      const rect = e.currentTarget.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const centerTime = pixelToTime(x, rect.width);

      // Zoom direction: negative deltaY = zoom in, positive = zoom out
      const direction = e.deltaY < 0 ? 1 : -1;
      zoom(direction, centerTime);
    },
    [pixelToTime, zoom]
  );

  return {
    handleMouseDown,
    handleMouseMove,
    handleMouseUp,
    handleWheel,
  };
}
