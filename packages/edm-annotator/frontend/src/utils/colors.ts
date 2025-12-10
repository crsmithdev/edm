import type { SectionLabel } from "@/types/structure";

/**
 * Color mapping for section labels
 */
export const labelColors: Record<SectionLabel, string> = {
  intro: "rgba(91, 124, 255, 0.2)", // Blue
  buildup: "rgba(255, 184, 0, 0.2)", // Orange
  breakdown: "rgba(0, 230, 184, 0.2)", // Cyan
  breakbuild: "rgba(255, 107, 107, 0.2)", // Red
  outro: "rgba(156, 39, 176, 0.2)", // Purple
  unlabeled: "rgba(128, 128, 128, 0.1)", // Gray
};

/**
 * Border color mapping for section labels
 */
export const labelBorderColors: Record<SectionLabel, string> = {
  intro: "rgba(91, 124, 255, 0.8)",
  buildup: "rgba(255, 184, 0, 0.8)",
  breakdown: "rgba(0, 230, 184, 0.8)",
  breakbuild: "rgba(255, 107, 107, 0.8)",
  outro: "rgba(156, 39, 176, 0.8)",
  unlabeled: "rgba(128, 128, 128, 0.5)",
};
