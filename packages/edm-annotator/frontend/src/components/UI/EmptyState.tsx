import type { ReactNode } from "react";

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
}

/**
 * Empty state component for when no content is available
 */
export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "var(--space-12)",
        textAlign: "center",
        color: "var(--text-tertiary)",
      }}
    >
      {icon && (
        <div
          style={{
            marginBottom: "var(--space-4)",
            opacity: 0.5,
          }}
        >
          {icon}
        </div>
      )}
      <h3
        style={{
          fontSize: "var(--font-size-lg)",
          fontWeight: "var(--font-weight-semibold)",
          color: "var(--text-secondary)",
          marginBottom: "var(--space-2)",
        }}
      >
        {title}
      </h3>
      <p
        style={{
          fontSize: "var(--font-size-sm)",
          color: "var(--text-muted)",
          marginBottom: action ? "var(--space-4)" : "0",
          maxWidth: "400px",
        }}
      >
        {description}
      </p>
      {action && <div>{action}</div>}
    </div>
  );
}
