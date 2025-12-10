import { useEffect } from "react";
import { useUIStore } from "@/stores";

/**
 * Status notification toast
 */
export function StatusToast() {
  const { statusMessage, clearStatus } = useUIStore();

  useEffect(() => {
    if (statusMessage) {
      const timer = setTimeout(() => {
        clearStatus();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [statusMessage, clearStatus]);

  if (!statusMessage) return null;

  return (
    <div
      style={{
        position: "fixed",
        bottom: "24px",
        right: "24px",
        padding: "12px 20px",
        background: "#1E2139",
        border: "1px solid #5B7CFF",
        borderRadius: "8px",
        color: "#E5E7EB",
        fontSize: "14px",
        boxShadow: "0 4px 12px rgba(0, 0, 0, 0.3)",
        zIndex: 1000,
        animation: "slideIn 0.3s ease",
      }}
    >
      {statusMessage}
    </div>
  );
}
