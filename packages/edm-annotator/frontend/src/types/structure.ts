export type SectionLabel =
  | "intro"
  | "buildup"
  | "breakdown"
  | "breakbuild"
  | "outro"
  | "unlabeled";

export interface Region {
  start: number;
  end: number;
  label: SectionLabel;
}

export interface Boundary {
  time: number;
  label: SectionLabel;
}
