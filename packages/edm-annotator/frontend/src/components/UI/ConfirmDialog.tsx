import { useEffect } from "react";
import { AlertTriangle } from "lucide-react";
import { Button } from "./Button";

interface ConfirmDialogProps {
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  variant?: "danger" | "warning" | "primary";
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Confirmation dialog modal
 */
export function ConfirmDialog({
  title,
  message,
  confirmText = "Confirm",
  cancelText = "Cancel",
  variant = "danger",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onCancel();
      } else if (e.key === "Enter") {
        onConfirm();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onConfirm, onCancel]);

  return (
    <>
      {/* Modal Overlay */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: "rgba(0, 0, 0, 0.7)",
          zIndex: 1000,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "var(--space-4)",
        }}
        onClick={onCancel}
      >
        {/* Modal Dialog */}
        <div
          style={{
            background: "var(--bg-primary)",
            border: "1px solid var(--border-primary)",
            borderRadius: "var(--radius-lg)",
            maxWidth: "480px",
            width: "100%",
            boxShadow: "0 20px 60px rgba(0, 0, 0, 0.5)",
          }}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div
            style={{
              padding: "var(--space-4)",
              borderBottom: "1px solid var(--border-primary)",
              display: "flex",
              alignItems: "center",
              gap: "var(--space-3)",
            }}
          >
            <AlertTriangle
              size={24}
              style={{
                color:
                  variant === "danger"
                    ? "var(--color-error)"
                    : variant === "warning"
                      ? "var(--color-warning)"
                      : "var(--color-primary)",
              }}
            />
            <h3
              style={{
                margin: 0,
                fontSize: "var(--font-size-lg)",
                fontWeight: "var(--font-weight-semibold)",
                color: "var(--text-primary)",
              }}
            >
              {title}
            </h3>
          </div>

          {/* Body */}
          <div
            style={{
              padding: "var(--space-5)",
              color: "var(--text-secondary)",
              fontSize: "var(--font-size-base)",
              lineHeight: "1.6",
            }}
          >
            {message}
          </div>

          {/* Footer */}
          <div
            style={{
              padding: "var(--space-4)",
              borderTop: "1px solid var(--border-primary)",
              display: "flex",
              justifyContent: "flex-end",
              gap: "var(--space-3)",
            }}
          >
            <Button onClick={onCancel} variant="secondary">
              {cancelText}
            </Button>
            <Button onClick={onConfirm} variant={variant}>
              {confirmText}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
