import { useState, type ReactNode } from "react";

interface TooltipProps {
  content: string;
  children: ReactNode;
  shortcut?: string;
}

/**
 * Tooltip component with optional keyboard shortcut display
 */
export function Tooltip({ content, children, shortcut }: TooltipProps) {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div
      style={{
        position: "relative",
        display: "block",
        width: "100%",
        height: "100%",
        overflow: "visible",
      }}
      onMouseEnter={() => setIsVisible(true)}
      onMouseLeave={() => setIsVisible(false)}
      onFocus={() => setIsVisible(true)}
      onBlur={() => setIsVisible(false)}
    >
      {children}
      {isVisible && (
        <div
          style={{
            position: "absolute",
            bottom: "calc(100% + var(--space-2))",
            left: "50%",
            transform: "translateX(-50%)",
            background: "var(--bg-elevated)",
            color: "var(--text-primary)",
            padding: "var(--space-2) var(--space-3)",
            borderRadius: "var(--radius-sm)",
            fontSize: "var(--font-size-xs)",
            whiteSpace: "nowrap",
            border: "1px solid var(--border-subtle)",
            boxShadow: "var(--shadow-lg)",
            zIndex: 9999,
            pointerEvents: "none",
            animation: "tooltipFadeIn 0.15s ease-out",
            willChange: "opacity",
          }}
        >
          {content}
          {shortcut && (
            <span
              style={{
                marginLeft: "var(--space-2)",
                padding: "var(--space-1) var(--space-2)",
                background: "var(--bg-tertiary)",
                borderRadius: "var(--radius-sm)",
                fontSize: "var(--font-size-xs)",
                fontWeight: "var(--font-weight-semibold)",
                border: "1px solid var(--border-subtle)",
              }}
            >
              {shortcut}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
