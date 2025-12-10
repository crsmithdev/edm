import { useEffect, useState } from "react";
import { CheckCircle, AlertCircle, Info, X } from "lucide-react";
import { useUIStore } from "@/stores";

type ToastType = "success" | "error" | "info";

/**
 * Enhanced status notification toast with animations and icons
 */
export function StatusToast() {
  const { statusMessage, clearStatus } = useUIStore();
  const [isExiting, setIsExiting] = useState(false);

  // Determine toast type based on message content
  const getToastType = (message: string): ToastType => {
    const lower = message.toLowerCase();
    if (lower.includes("success") || lower.includes("saved") || lower.includes("loaded")) {
      return "success";
    }
    if (lower.includes("error") || lower.includes("fail")) {
      return "error";
    }
    return "info";
  };

  const type = statusMessage ? getToastType(statusMessage) : "info";

  const icons = {
    success: <CheckCircle size={18} />,
    error: <AlertCircle size={18} />,
    info: <Info size={18} />,
  };

  const colors = {
    success: "var(--color-success)",
    error: "var(--color-danger)",
    info: "var(--color-primary)",
  };

  useEffect(() => {
    if (statusMessage) {
      setIsExiting(false);
      const timer = setTimeout(() => {
        setIsExiting(true);
        setTimeout(() => clearStatus(), 200); // Allow exit animation to complete
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [statusMessage, clearStatus]);

  if (!statusMessage) return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: "var(--space-6)",
        right: "var(--space-6)",
        padding: "var(--space-3) var(--space-4)",
        background: "var(--bg-secondary)",
        border: `1px solid ${colors[type]}`,
        borderRadius: "var(--radius-md)",
        color: "var(--text-secondary)",
        fontSize: "var(--font-size-base)",
        boxShadow: "var(--shadow-lg)",
        zIndex: "var(--z-toast)",
        animation: isExiting ? "slideOut 0.2s ease" : "slideIn 0.3s ease",
        display: "flex",
        alignItems: "center",
        gap: "var(--space-3)",
        maxWidth: "400px",
      }}
    >
      <div style={{ color: colors[type], display: "flex", alignItems: "center" }}>
        {icons[type]}
      </div>
      <div style={{ flex: 1 }}>{statusMessage}</div>
      <button
        onClick={() => {
          setIsExiting(true);
          setTimeout(() => clearStatus(), 200);
        }}
        style={{
          background: "transparent",
          border: "none",
          color: "var(--text-muted)",
          cursor: "pointer",
          padding: "var(--space-1)",
          display: "flex",
          alignItems: "center",
          transition: "color var(--transition-fast)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "var(--text-secondary)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "var(--text-muted)";
        }}
      >
        <X size={16} />
      </button>
    </div>
  );
}
