export interface Track {
  filename: string;
  path: string;
  has_reference: boolean;
  has_generated: boolean;
}

export interface LoadedTrack {
  filename: string;
  duration: number;
  bpm: number | null;
  downbeat: number;
  sample_rate: number;
}
