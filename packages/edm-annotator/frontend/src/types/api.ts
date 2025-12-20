import type { Track } from "./track";
import type { WaveformData } from "./waveform";
import type { Boundary } from "./structure";

export interface TrackListResponse extends Array<Track> {}

export interface LoadTrackResponse extends WaveformData {
  filename: string;
  bpm: number | null;
  downbeat: number;
  boundaries?: Boundary[];
  annotation_tier?: number | null; // 1 = reference (hand-tagged), 2 = generated
}

export interface LoadGeneratedAnnotationResponse {
  bpm: number | null;
  downbeat: number;
  boundaries: Boundary[];
}

export interface SaveAnnotationRequest {
  filename: string;
  bpm: number;
  downbeat: number;
  boundaries: Boundary[];
}

export interface SaveAnnotationResponse {
  success: boolean;
  output: string;
  boundaries_count: number;
}
