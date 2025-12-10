import type { ReactNode } from "react";

interface InfoCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
}

/**
 * Reusable info card component for displaying labeled values
 */
export function InfoCard({ label, value }: InfoCardProps) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: "var(--space-2)",
      }}
    >
      <span
        style={{
          fontSize: "var(--font-size-xs)",
          fontWeight: "var(--font-weight-semibold)",
          color: "var(--text-tertiary)",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: "var(--font-size-lg)",
          fontWeight: "var(--font-weight-bold)",
          color: "var(--color-accent)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </span>
    </div>
  );
}
