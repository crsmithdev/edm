export type SectionLabel =
  | "intro"
  | "buildup"
  | "breakdown"
  | "breakdown-buildup"
  | "outro"
  | "default";

export interface Region {
  start: number;
  end: number;
  label: SectionLabel;
}

export interface Boundary {
  time: number;
  label: SectionLabel;
}
