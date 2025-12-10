import axios from "axios";
import type {
  TrackListResponse,
  LoadTrackResponse,
  SaveAnnotationRequest,
  SaveAnnotationResponse,
} from "@/types/api";

const api = axios.create({
  baseURL: "/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export const trackService = {
  /**
   * Fetch list of available tracks
   */
  async getTracks(): Promise<TrackListResponse> {
    const response = await api.get<TrackListResponse>("/tracks");
    return response.data;
  },

  /**
   * Get audio file URL for playback
   */
  getAudioUrl(filename: string): string {
    return `/api/audio/${filename}`;
  },

  /**
   * Load track waveform and metadata
   */
  async loadTrack(filename: string): Promise<LoadTrackResponse> {
    const response = await api.get<LoadTrackResponse>(`/load/${filename}`);
    return response.data;
  },

  /**
   * Save annotation to backend
   */
  async saveAnnotation(
    data: SaveAnnotationRequest
  ): Promise<SaveAnnotationResponse> {
    const response = await api.post<SaveAnnotationResponse>("/save", data);
    return response.data;
  },
};

export default api;
