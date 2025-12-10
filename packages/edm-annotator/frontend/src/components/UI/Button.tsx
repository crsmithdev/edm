import { useState } from "react";
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
    background: "linear-gradient(135deg, #7b6aff 0%, #6b5eff 100%)",
    color: "var(--text-primary)",
    border: "none",
    boxShadow: "0 4px 12px rgba(123, 106, 255, 0.25)",
  },
  secondary: {
    background: "var(--bg-elevated)",
    color: "var(--text-secondary)",
    border: "1px solid var(--border-subtle)",
  },
  accent: {
    background: "linear-gradient(135deg, #00e5cc 0%, #00d4bb 100%)",
    color: "var(--bg-primary)",
    border: "none",
    boxShadow: "0 4px 12px rgba(0, 229, 204, 0.25)",
  },
  danger: {
    background: "var(--color-danger)",
    color: "var(--text-primary)",
    border: "none",
    boxShadow: "0 4px 12px rgba(255, 107, 107, 0.25)",
  },
  warning: {
    background: "linear-gradient(135deg, #ffd966 0%, #ffcc4d 100%)",
    color: "var(--bg-primary)",
    border: "none",
    boxShadow: "0 4px 12px rgba(255, 217, 102, 0.25)",
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
    minHeight: "32px",
  },
  md: {
    padding: "var(--space-3) var(--space-5)",
    fontSize: "var(--font-size-base)",
    minHeight: "var(--button-height)",
  },
  lg: {
    padding: "var(--space-4) var(--space-6)",
    fontSize: "var(--font-size-lg)",
    minHeight: "52px",
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
  const [isHovered, setIsHovered] = useState(false);

  const getHoverStyle = (): React.CSSProperties => {
    if (disabled || !isHovered) return {};

    switch (variant) {
      case "primary":
        return { boxShadow: "var(--shadow-glow-purple)", transform: "translateY(-1px)" };
      case "accent":
        return { boxShadow: "var(--shadow-glow-cyan)", transform: "translateY(-1px)" };
      case "warning":
        return { boxShadow: "var(--shadow-glow-yellow)", transform: "translateY(-1px)" };
      default:
        return { transform: "translateY(-1px)" };
    }
  };

  return (
    <button
      disabled={disabled}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      style={{
        ...variantStyles[variant],
        ...sizeStyles[size],
        ...getHoverStyle(),
        borderRadius: "var(--radius-md)",
        cursor: disabled ? "not-allowed" : "pointer",
        fontWeight: "var(--font-weight-semibold)",
        transition: "all var(--transition-base)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: icon ? "var(--space-2)" : "0",
        whiteSpace: "nowrap",
        width: "100%",
        ...style,
      }}
      {...props}
    >
      {icon && <span style={{ display: "flex", alignItems: "center" }}>{icon}</span>}
      {children}
    </button>
  );
}
