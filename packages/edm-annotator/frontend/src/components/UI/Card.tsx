import type { CSSProperties, ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  padding?: "sm" | "md" | "lg";
  className?: string;
  style?: CSSProperties;
}

const paddingStyles = {
  sm: "var(--space-4)",
  md: "var(--space-5)",
  lg: "var(--space-6)",
};

/**
 * Reusable card container component
 */
export function Card({ children, padding = "md", className, style }: CardProps) {
  return (
    <div
      className={className}
      style={{
        background: "var(--bg-secondary)",
        padding: paddingStyles[padding],
        borderRadius: "var(--radius-xl)",
        border: "1px solid var(--border-primary)",
        boxShadow: "var(--shadow-md)",
        ...style,
      }}
    >
      {children}
    </div>
  );
}
