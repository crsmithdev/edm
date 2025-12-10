import type { ReactNode } from "react";

interface InfoCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
}

/**
 * Reusable info card component for displaying labeled values
 */
export function InfoCard({ label, value, icon }: InfoCardProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "var(--space-1)",
        padding: "var(--space-2) var(--space-4)",
        background: "var(--bg-tertiary)",
        border: "1px solid var(--border-subtle)",
        borderRadius: "var(--radius-md)",
        height: "var(--button-height)",
        minWidth: "80px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "var(--space-1)",
        }}
      >
        {icon && (
          <span
            style={{
              display: "flex",
              alignItems: "center",
              color: "var(--text-muted)",
            }}
          >
            {icon}
          </span>
        )}
        <span
          style={{
            fontSize: "var(--font-size-xs)",
            fontWeight: "var(--font-weight-semibold)",
            color: "var(--text-muted)",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
          }}
        >
          {label}
        </span>
      </div>
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
