import type { ButtonHTMLAttributes, ReactNode } from "react";

export type ButtonVariant = "primary" | "secondary" | "accent" | "danger" | "warning" | "ghost";
export type ButtonSize = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  children: ReactNode;
  icon?: ReactNode;
}

const variantStyles: Record<ButtonVariant, React.CSSProperties> = {
  primary: {
    background: "var(--color-primary)",
    color: "var(--text-primary)",
    border: "none",
  },
  secondary: {
    background: "var(--border-subtle)",
    color: "var(--text-secondary)",
    border: "none",
  },
  accent: {
    background: "var(--color-accent)",
    color: "var(--bg-primary)",
    border: "none",
  },
  danger: {
    background: "var(--color-danger)",
    color: "var(--text-primary)",
    border: "none",
  },
  warning: {
    background: "var(--color-warning)",
    color: "var(--bg-primary)",
    border: "none",
  },
  ghost: {
    background: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--border-subtle)",
  },
};

const sizeStyles: Record<ButtonSize, React.CSSProperties> = {
  sm: {
    padding: "var(--space-2) var(--space-3)",
    fontSize: "var(--font-size-sm)",
    height: "32px",
  },
  md: {
    padding: "var(--space-3) var(--space-5)",
    fontSize: "var(--font-size-base)",
    height: "var(--button-height)",
  },
  lg: {
    padding: "var(--space-4) var(--space-6)",
    fontSize: "var(--font-size-lg)",
    height: "52px",
  },
};

/**
 * Reusable button component with variants and sizes
 */
export function Button({
  variant = "primary",
  size = "md",
  children,
  icon,
  disabled,
  style,
  ...props
}: ButtonProps) {
  return (
    <button
      disabled={disabled}
      style={{
        ...variantStyles[variant],
        ...sizeStyles[size],
        borderRadius: "var(--radius-md)",
        cursor: disabled ? "not-allowed" : "pointer",
        fontWeight: "var(--font-weight-semibold)",
        transition: "all var(--transition-base)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: icon ? "var(--space-2)" : "0",
        whiteSpace: "nowrap",
        ...style,
      }}
      {...props}
    >
      {icon && <span style={{ display: "flex", alignItems: "center" }}>{icon}</span>}
      {children}
    </button>
  );
}
