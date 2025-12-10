import type { CSSProperties } from "react";

interface SkeletonProps {
  width?: string | number;
  height?: string | number;
  style?: CSSProperties;
}

/**
 * Loading skeleton component for placeholder content
 */
export function Skeleton({ width = "100%", height = "20px", style }: SkeletonProps) {
  return (
    <div
      style={{
        width,
        height,
        background: "linear-gradient(90deg, var(--bg-tertiary) 25%, var(--bg-elevated) 50%, var(--bg-tertiary) 75%)",
        backgroundSize: "200% 100%",
        animation: "shimmer 1.5s infinite",
        borderRadius: "var(--radius-sm)",
        ...style,
      }}
    />
  );
}

/**
 * Track list item skeleton
 */
export function TrackItemSkeleton() {
  return (
    <div style={{ padding: "var(--space-3) var(--space-4)" }}>
      <Skeleton height="14px" width="80%" style={{ marginBottom: "var(--space-2)" }} />
      <Skeleton height="11px" width="40%" />
    </div>
  );
}
