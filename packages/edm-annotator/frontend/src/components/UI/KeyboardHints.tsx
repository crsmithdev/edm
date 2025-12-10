import { useState, useEffect } from "react";
import { X, Keyboard } from "lucide-react";
import { Button } from "./Button";

interface Shortcut {
  key: string;
  description: string;
  category: string;
}

const shortcuts: Shortcut[] = [
  { key: "Space", description: "Play/Pause", category: "Playback" },
  { key: "C", description: "Return to cue point", category: "Playback" },
  { key: "R", description: "Return to cue point", category: "Playback" },
  { key: "B", description: "Add boundary at current time", category: "Editing" },
  { key: "D", description: "Set downbeat", category: "Editing" },
  { key: "Q", description: "Toggle quantize", category: "Editing" },
  { key: "←", description: "Jump back 4 bars", category: "Navigation" },
  { key: "→", description: "Jump forward 4 bars", category: "Navigation" },
  { key: "Ctrl + ←", description: "Jump back 1 bar", category: "Navigation" },
  { key: "Ctrl + →", description: "Jump forward 1 bar", category: "Navigation" },
  { key: "Shift + ←", description: "Jump back 8 bars", category: "Navigation" },
  { key: "Shift + →", description: "Jump forward 8 bars", category: "Navigation" },
  { key: "↑", description: "Previous track", category: "Navigation" },
  { key: "↓", description: "Next track", category: "Navigation" },
  { key: "Shift + Click", description: "Add boundary on waveform", category: "Waveform" },
  { key: "Click", description: "Set cue point", category: "Waveform" },
  { key: "Drag", description: "Pan waveform", category: "Waveform" },
  { key: "Scroll", description: "Zoom in/out", category: "Waveform" },
];

/**
 * Keyboard shortcuts overlay
 */
export function KeyboardHints() {
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "?" && e.shiftKey) {
        setIsOpen((prev) => !prev);
      }
      if (e.key === "Escape" && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen]);

  const categories = Array.from(new Set(shortcuts.map((s) => s.category)));

  return (
    <>
      {/* Trigger Button */}
      <Button
        onClick={() => setIsOpen(true)}
        variant="ghost"
        size="sm"
        icon={<Keyboard size={16} />}
        style={{
          position: "fixed",
          bottom: "var(--space-4)",
          right: "var(--space-4)",
          zIndex: "var(--z-overlay)",
        }}
        title="Keyboard shortcuts (Shift+?)"
      >
        ?
      </Button>

      {/* Modal Overlay */}
      {isOpen && (
        <>
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0, 0, 0, 0.7)",
              zIndex: "var(--z-modal)",
              animation: "fadeIn 0.2s ease-out",
            }}
            onClick={() => setIsOpen(false)}
          />
          <div
            style={{
              position: "fixed",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              background: "var(--bg-secondary)",
              border: "1px solid var(--border-subtle)",
              borderRadius: "var(--radius-xl)",
              boxShadow: "var(--shadow-lg)",
              maxWidth: "800px",
              maxHeight: "80vh",
              width: "90%",
              overflow: "hidden",
              zIndex: "calc(var(--z-modal) + 1)",
              animation: "slideUp 0.2s ease-out",
            }}
          >
            {/* Header */}
            <div
              style={{
                padding: "var(--space-5)",
                borderBottom: "1px solid var(--border-subtle)",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <h2
                style={{
                  fontSize: "var(--font-size-xl)",
                  fontWeight: "var(--font-weight-bold)",
                  color: "var(--text-primary)",
                  margin: 0,
                }}
              >
                Keyboard Shortcuts
              </h2>
              <button
                onClick={() => setIsOpen(false)}
                style={{
                  background: "transparent",
                  border: "none",
                  color: "var(--text-tertiary)",
                  cursor: "pointer",
                  padding: "var(--space-2)",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Content */}
            <div
              style={{
                padding: "var(--space-5)",
                overflowY: "auto",
                maxHeight: "calc(80vh - 80px)",
              }}
            >
              {categories.map((category) => (
                <div key={category} style={{ marginBottom: "var(--space-6)" }}>
                  <h3
                    style={{
                      fontSize: "var(--font-size-sm)",
                      fontWeight: "var(--font-weight-semibold)",
                      color: "var(--text-tertiary)",
                      textTransform: "uppercase",
                      letterSpacing: "0.5px",
                      marginBottom: "var(--space-3)",
                    }}
                  >
                    {category}
                  </h3>
                  <div style={{ display: "flex", flexDirection: "column", gap: "var(--space-2)" }}>
                    {shortcuts
                      .filter((s) => s.category === category)
                      .map((shortcut, idx) => (
                        <div
                          key={idx}
                          style={{
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            padding: "var(--space-2)",
                            background: "var(--bg-tertiary)",
                            borderRadius: "var(--radius-sm)",
                          }}
                        >
                          <span
                            style={{
                              fontSize: "var(--font-size-sm)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {shortcut.description}
                          </span>
                          <kbd
                            style={{
                              padding: "var(--space-1) var(--space-2)",
                              background: "var(--bg-primary)",
                              border: "1px solid var(--border-subtle)",
                              borderRadius: "var(--radius-sm)",
                              fontSize: "var(--font-size-xs)",
                              fontWeight: "var(--font-weight-semibold)",
                              color: "var(--color-primary)",
                              fontFamily: "monospace",
                            }}
                          >
                            {shortcut.key}
                          </kbd>
                        </div>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </>
  );
}
