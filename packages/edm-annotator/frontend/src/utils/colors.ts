import type { SectionLabel } from "@/types/structure";

/**
 * Color mapping for section labels (ToneForge-inspired)
 */
export const labelColors: Record<SectionLabel, string> = {
  intro: "rgba(123, 106, 255, 0.2)", // Purple
  buildup: "rgba(0, 229, 204, 0.2)", // Cyan
  breakdown: "rgba(255, 107, 181, 0.2)", // Pink
  "breakdown-buildup": "rgba(167, 139, 250, 0.2)", // Light purple
  outro: "rgba(139, 122, 255, 0.2)", // Medium purple
  default: "rgba(96, 96, 104, 0.1)", // Gray
};

/**
 * Border color mapping for section labels (ToneForge-inspired)
 */
export const labelBorderColors: Record<SectionLabel, string> = {
  intro: "#7b6aff", // Purple
  buildup: "#00e5cc", // Cyan
  breakdown: "#ff6bb5", // Pink
  "breakdown-buildup": "#a78bfa", // Light purple
  outro: "#8b7aff", // Medium purple
  default: "#606068", // Gray
};

/**
 * Solid fill colors for waveforms (ToneForge-inspired)
 */
export const labelWaveformColors: Record<SectionLabel, string> = {
  intro: "#7b6aff", // Purple
  buildup: "#00e5cc", // Cyan
  breakdown: "#ff6bb5", // Pink
  "breakdown-buildup": "#a78bfa", // Light purple
  outro: "#8b7aff", // Medium purple
  default: "#e8e8ea", // Light gray
};
