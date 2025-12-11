import { useMemo, useEffect, useState, useRef, useCallback } from "react";
import { useWaveformStore, useAudioStore, useStructureStore, useUIStore, useTempoStore } from "@/stores";
import { getBeatDuration } from "@/utils/barCalculations";
import { BeatGrid } from "./BeatGrid";
import { BoundaryMarkers } from "./BoundaryMarkers";
import { RegionOverlays } from "./RegionOverlays";

interface DetailWaveformProps {
  /** Time span to display (seconds) */
  span: number;
}

/**
 * Detail waveform with centered playhead - waveform scrolls around fixed center position
 * Shows empty space at track start/end when playhead is near boundaries
 */
export function DetailWaveform({ span }: DetailWaveformProps) {
  const {
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    duration,
    setViewport,
  } = useWaveformStore();
  const { currentTime, seek, cuePoint } = useAudioStore();
  const { addBoundary } = useStructureStore();
  const { quantizeEnabled } = useUIStore();
  const { trackBPM, trackDownbeat } = useTempoStore();

  // Calculate the viewport centered on currentTime
  const viewport = useMemo(() => {
    const halfSpan = span / 2;
    // Always center on currentTime, even if it means showing empty space
    const viewStart = currentTime - halfSpan;
    const viewEnd = currentTime + halfSpan;
    return { start: viewStart, end: viewEnd };
  }, [currentTime, span]);

  // Update the store viewport for BeatGrid and overlays to use
  useEffect(() => {
    setViewport(
      Math.max(0, viewport.start),
      Math.min(duration, viewport.end)
    );
  }, [viewport.start, viewport.end, duration, setViewport]);

  // Calculate global max amplitude once for consistent scaling
  const globalMaxAmplitude = useMemo(() => {
    if (waveformTimes.length === 0) return 0.001;
    let max = 0;
    for (let i = 0; i < waveformTimes.length; i++) {
      const total =
        Math.abs(waveformBass[i] || 0) +
        Math.abs(waveformMids[i] || 0) +
        Math.abs(waveformHighs[i] || 0);
      if (total > max) max = total;
    }
    return Math.max(max, 0.001);
  }, [waveformBass, waveformMids, waveformHighs, waveformTimes]);

  // Generate stacked area paths for the visible portion
  const { bassPath, midsPath, highsPath } = useMemo(() => {
    if (waveformTimes.length === 0) {
      return { bassPath: "", midsPath: "", highsPath: "" };
    }

    const viewportDuration = viewport.end - viewport.start;
    if (viewportDuration <= 0) {
      return { bassPath: "", midsPath: "", highsPath: "" };
    }

    // Find samples within the actual track bounds that fall in our viewport
    const actualStart = Math.max(0, viewport.start);
    const actualEnd = Math.min(duration, viewport.end);

    const startIdx = waveformTimes.findIndex((t) => t >= actualStart);
    const endIdx = waveformTimes.findIndex((t) => t >= actualEnd);

    const start = startIdx === -1 ? 0 : startIdx;
    const end = endIdx === -1 ? waveformTimes.length : endIdx;
    const indices = Array.from({ length: end - start }, (_, i) => start + i);

    if (indices.length === 0) {
      return { bassPath: "", midsPath: "", highsPath: "" };
    }

    const width = 100;
    const height = 100;

    // Apply temporal smoothing with adaptive window size based on zoom level
    // More smoothing when zoomed out (longer spans), less when zoomed in
    const smoothingWindowSize = Math.max(1, Math.floor(span / 2)); // 1 sample per 2 seconds of visible span

    const smoothArray = (arr: number[], windowSize: number): number[] => {
      if (windowSize <= 1) return arr;

      const smoothed: number[] = [];
      for (let i = 0; i < arr.length; i++) {
        const halfWindow = Math.floor(windowSize / 2);
        const windowStart = Math.max(0, i - halfWindow);
        const windowEnd = Math.min(arr.length, i + halfWindow + 1);

        let sum = 0;
        for (let j = windowStart; j < windowEnd; j++) {
          sum += arr[j];
        }
        smoothed[i] = sum / (windowEnd - windowStart);
      }
      return smoothed;
    };

    // Extract raw amplitude values
    const rawBass = indices.map((idx) => Math.abs(waveformBass[idx] || 0));
    const rawMids = indices.map((idx) => Math.abs(waveformMids[idx] || 0));
    const rawHighs = indices.map((idx) => Math.abs(waveformHighs[idx] || 0));

    // Apply smoothing to each band
    const smoothedBass = smoothArray(rawBass, smoothingWindowSize);
    const smoothedMids = smoothArray(rawMids, smoothingWindowSize);
    const smoothedHighs = smoothArray(rawHighs, smoothingWindowSize);

    // Calculate cumulative heights for each sample
    // Map time to x position based on the full viewport (including empty space)
    const cumulativeData = indices.map((idx, i) => {
      const bass = smoothedBass[i];
      const mids = smoothedMids[i];
      const highs = smoothedHighs[i];
      const time = waveformTimes[idx];
      // Position relative to viewport start (which may be negative)
      const x = ((time - viewport.start) / viewportDuration) * width;

      return {
        x,
        bass,
        bassTop: bass,
        midsTop: bass + mids,
        highsTop: bass + mids + highs,
      };
    });

    // Use global max for consistent scaling across all viewport positions
    const scale = (height * 0.9) / globalMaxAmplitude;

    const center = height / 2;
    const halfScale = scale / 2;

    // Generate bass area (innermost layer, mirrored)
    const bassTopPoints = cumulativeData.map((d) => {
      const y = center - d.bassTop * halfScale;
      return `${d.x},${y}`;
    });

    const bassBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.bassTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const bassPath =
      bassTopPoints.length > 0 && bassBottomPoints.length > 0
        ? `M${bassTopPoints.join(" L")} L${bassBottomPoints.join(" L")} Z`
        : "";

    // Generate mids area (middle layer, mirrored)
    const midsTopPoints = cumulativeData.map((d) => {
      const y = center - d.midsTop * halfScale;
      return `${d.x},${y}`;
    });

    const midsTopBasePoints = cumulativeData
      .map((d) => {
        const y = center - d.bassTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const midsBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.midsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const midsBottomBasePoints = cumulativeData.map((d) => {
      const y = center + d.bassTop * halfScale;
      return `${d.x},${y}`;
    });

    const midsPath =
      midsTopPoints.length > 0
        ? `M${midsTopPoints.join(" L")} L${midsTopBasePoints.join(" L")} M${midsBottomBasePoints.join(" L")} L${midsBottomPoints.join(" L")} Z`
        : "";

    // Generate highs area (outermost layer, mirrored)
    const highsTopPoints = cumulativeData.map((d) => {
      const y = center - d.highsTop * halfScale;
      return `${d.x},${y}`;
    });

    const highsTopBasePoints = cumulativeData
      .map((d) => {
        const y = center - d.midsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const highsBottomPoints = cumulativeData
      .map((d) => {
        const y = center + d.highsTop * halfScale;
        return `${d.x},${y}`;
      })
      .reverse();

    const highsBottomBasePoints = cumulativeData.map((d) => {
      const y = center + d.midsTop * halfScale;
      return `${d.x},${y}`;
    });

    const highsPath =
      highsTopPoints.length > 0
        ? `M${highsTopPoints.join(" L")} L${highsTopBasePoints.join(" L")} M${highsBottomBasePoints.join(" L")} L${highsBottomPoints.join(" L")} Z`
        : "";

    return { bassPath, midsPath, highsPath };
  }, [
    waveformBass,
    waveformMids,
    waveformHighs,
    waveformTimes,
    viewport.start,
    viewport.end,
    duration,
    globalMaxAmplitude,
    span,
  ]);

  // Drag state for scrubbing
  const [isDragging, setIsDragging] = useState(false);
  const [isShiftDown, setIsShiftDown] = useState(false);
  const dragStartX = useRef<number>(0);
  const dragStartTime = useRef<number>(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Track ctrl key state for cursor changes (ctrl+click adds boundary)
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Control") setIsShiftDown(true);
    };
    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.key === "Control") setIsShiftDown(false);
    };
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // Handle ctrl+click for boundaries (at click position, respecting quantize)
  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    // Only handle ctrl+click for boundaries
    if (!e.ctrlKey && !e.metaKey) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const percent = x / rect.width;
    const viewportDuration = viewport.end - viewport.start;
    const rawTime = viewport.start + percent * viewportDuration;

    // Clamp to valid track time
    let time = Math.max(0, Math.min(duration, rawTime));

    // Snap to nearest beat if quantize enabled
    if (quantizeEnabled && trackBPM > 0) {
      const beatDuration = getBeatDuration(trackBPM);
      const beatsFromDownbeat = (time - trackDownbeat) / beatDuration;
      const nearestBeat = Math.round(beatsFromDownbeat);
      time = trackDownbeat + nearestBeat * beatDuration;
      time = Math.max(0, Math.min(duration, time));
    }

    addBoundary(time);
  };

  // Track if shift was held when drag started (to bypass quantize)
  const shiftHeldOnDragStart = useRef(false);

  // Drag to scrub - dragging left moves playback forward, right moves backward
  const handleMouseDown = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.ctrlKey || e.metaKey) return; // Don't start drag on ctrl+click (boundary mode)
      e.preventDefault();
      setIsDragging(true);
      dragStartX.current = e.clientX;
      dragStartTime.current = currentTime;
      shiftHeldOnDragStart.current = e.shiftKey; // Remember if shift was held
    },
    [currentTime]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const deltaX = e.clientX - dragStartX.current;
      const containerWidth = containerRef.current.getBoundingClientRect().width;
      const viewportDuration = viewport.end - viewport.start;

      // Convert pixel delta to time delta
      // Dragging left (negative deltaX) should move forward in time
      const timeDelta = (-deltaX / containerWidth) * viewportDuration;
      let newTime = Math.max(0, Math.min(duration, dragStartTime.current + timeDelta));

      // Snap to nearest beat if quantize enabled (shift bypasses quantize)
      if (quantizeEnabled && trackBPM > 0 && !shiftHeldOnDragStart.current) {
        const beatDuration = getBeatDuration(trackBPM);
        const beatsFromDownbeat = (newTime - trackDownbeat) / beatDuration;
        const nearestBeat = Math.round(beatsFromDownbeat);
        newTime = trackDownbeat + nearestBeat * beatDuration;
        newTime = Math.max(0, Math.min(duration, newTime));
      }

      seek(newTime);
    },
    [isDragging, viewport.end, viewport.start, duration, seek, quantizeEnabled, trackBPM, trackDownbeat]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Add global mouse listeners for drag
  useEffect(() => {
    if (isDragging) {
      window.addEventListener("mousemove", handleMouseMove);
      window.addEventListener("mouseup", handleMouseUp);
      return () => {
        window.removeEventListener("mousemove", handleMouseMove);
        window.removeEventListener("mouseup", handleMouseUp);
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp]);

  return (
    <div
      ref={containerRef}
      onClick={handleClick}
      onMouseDown={handleMouseDown}
      style={{
        position: "relative",
        width: "100%",
        height: "var(--waveform-height)",
        background: "var(--bg-tertiary)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-lg)",
        cursor: isDragging ? "grabbing" : isShiftDown ? "crosshair" : "grab",
        overflow: "hidden",
        userSelect: "none",
      }}
    >
      {/* Region overlays (behind everything) */}
      <RegionOverlays viewportStart={viewport.start} viewportEnd={viewport.end} />

      {/* Waveform SVG */}
      <svg
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          width: "100%",
          height: "100%",
          background: "#0a0a12",
        }}
      >
        {/* Bass layer (innermost, mirrored) - cyan */}
        <path d={bassPath} fill="rgba(0, 229, 204, 0.8)" stroke="none" />

        {/* Mids layer (middle, mirrored) - purple */}
        <path d={midsPath} fill="rgba(123, 106, 255, 0.8)" stroke="none" />

        {/* Highs layer (outermost, mirrored) - pink */}
        <path d={highsPath} fill="rgba(255, 107, 181, 0.8)" stroke="none" />

        {/* Center baseline */}
        <line
          x1="0"
          y1="50"
          x2="100"
          y2="50"
          stroke="rgba(255, 255, 255, 0.15)"
          strokeWidth="0.2"
        />
      </svg>

      {/* Beat grid overlay */}
      <BeatGrid viewportStart={viewport.start} viewportEnd={viewport.end} />

      {/* Boundary markers */}
      <BoundaryMarkers viewportStart={viewport.start} viewportEnd={viewport.end} />

      {/* Cue point marker (moves with waveform) */}
      {cuePoint >= viewport.start && cuePoint <= viewport.end && (
        <div
          style={{
            position: "absolute",
            left: `${((cuePoint - viewport.start) / (viewport.end - viewport.start)) * 100}%`,
            top: 0,
            width: "2px",
            height: "100%",
            background: "linear-gradient(180deg, #ff9500 0%, #ff6b00 100%)",
            boxShadow: "0 0 8px rgba(255, 149, 0, 0.6)",
            pointerEvents: "none",
            zIndex: 9,
            transform: "translateX(-1px)",
          }}
        />
      )}

      {/* Fixed center playhead */}
      <div
        style={{
          position: "absolute",
          left: "50%",
          top: 0,
          width: "2px",
          height: "100%",
          background: "linear-gradient(180deg, #1affef 0%, #00e5cc 100%)",
          boxShadow:
            "0 0 15px rgba(26, 255, 239, 0.6), 0 0 30px rgba(26, 255, 239, 0.3)",
          pointerEvents: "none",
          zIndex: 10,
          transform: "translateX(-1px)",
        }}
      />
    </div>
  );
}
