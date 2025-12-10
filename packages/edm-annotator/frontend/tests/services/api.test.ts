import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import axios from "axios";
import type { Track } from "@/types/track";
import type { SaveAnnotationRequest } from "@/types/api";

// Mock axios with a default implementation
const mockGet = vi.fn();
const mockPost = vi.fn();
const mockCreate = vi.fn(() => ({
  get: mockGet,
  post: mockPost,
}));

vi.mock("axios", () => ({
  default: {
    create: mockCreate,
  },
}));

// Import after mocking
const { trackService } = await import("@/services/api");

// Save axios.create calls before they get cleared
const createCalls = mockCreate.mock.calls.slice();

describe("trackService", () => {
  beforeEach(() => {
    // Clear mocks but preserve axios.create call history for API Configuration tests
    mockGet.mockClear();
    mockPost.mockClear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe("getTracks", () => {
    it("fetches tracks from API endpoint", async () => {
      const mockTracks: Track[] = [
        {
          filename: "track1.mp3",
          path: "/path/to/track1.mp3",
          has_reference: true,
          has_generated: false,
        },
        {
          filename: "track2.mp3",
          path: "/path/to/track2.mp3",
          has_reference: false,
          has_generated: true,
        },
      ];

      const mockResponse = {
        data: mockTracks,
        status: 200,
        statusText: "OK",
      };

      mockGet.mockResolvedValue(mockResponse);

      const result = await trackService.getTracks();

      expect(result).toEqual(mockTracks);
    });

    it("makes GET request to /tracks endpoint", async () => {
      const mockTracks: Track[] = [];
      const mockResponse = { data: mockTracks };

      mockGet.mockResolvedValue(mockResponse);

      await trackService.getTracks();

      expect(mockGet).toHaveBeenCalledWith("/tracks");
    });

    it("handles empty track list", async () => {
      const mockResponse = { data: [] };

      mockGet.mockResolvedValue(mockResponse);

      const result = await trackService.getTracks();

      expect(result).toEqual([]);
    });

    it("propagates errors from API", async () => {
      const mockError = new Error("Network error");

      mockGet.mockRejectedValue(mockError);

      await expect(trackService.getTracks()).rejects.toThrow("Network error");
    });

    it("handles 500 server error", async () => {
      const mockError = {
        response: {
          status: 500,
          data: { error: "Internal server error" },
        },
      };

      mockGet.mockRejectedValue(mockError);

      await expect(trackService.getTracks()).rejects.toMatchObject(mockError);
    });
  });

  describe("getAudioUrl", () => {
    it("constructs correct URL for audio file", () => {
      const filename = "test.mp3";
      const url = trackService.getAudioUrl(filename);

      expect(url).toBe("/api/audio/test.mp3");
    });

    it("handles filenames with spaces", () => {
      const filename = "test track.mp3";
      const url = trackService.getAudioUrl(filename);

      expect(url).toBe("/api/audio/test track.mp3");
    });

    it("handles filenames with special characters", () => {
      const filename = "test-track_001.mp3";
      const url = trackService.getAudioUrl(filename);

      expect(url).toBe("/api/audio/test-track_001.mp3");
    });

    it("returns URL without encoding", () => {
      // The function should return the raw URL path
      const filename = "track with spaces.mp3";
      const url = trackService.getAudioUrl(filename);

      expect(url).toBe("/api/audio/track with spaces.mp3");
    });
  });

  describe("loadTrack", () => {
    it("loads track with waveform data successfully", async () => {
      const mockLoadTrackResponse = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0.5,
        waveform_bass: [0.1, 0.2, 0.3],
        waveform_mids: [0.2, 0.3, 0.4],
        waveform_highs: [0.3, 0.4, 0.5],
        waveform_times: [0, 1, 2],
        duration: 180,
      };

      const mockResponse = { data: mockLoadTrackResponse };

      mockGet.mockResolvedValue(mockResponse);

      const result = await trackService.loadTrack("track1.mp3");

      expect(result).toEqual(mockLoadTrackResponse);
    });

    it("makes GET request to /load/:filename endpoint", async () => {
      const mockLoadTrackResponse = {
        filename: "test.mp3",
        bpm: 128,
        downbeat: 0,
        waveform_bass: [],
        waveform_mids: [],
        waveform_highs: [],
        waveform_times: [],
        duration: 180,
      };

      mockGet.mockResolvedValue({ data: mockLoadTrackResponse });

      await trackService.loadTrack("test.mp3");

      expect(mockGet).toHaveBeenCalledWith("/load/test.mp3");
    });

    it("loads track without BPM (null bpm)", async () => {
      const mockLoadTrackResponse = {
        filename: "track1.mp3",
        bpm: null,
        downbeat: 0,
        waveform_bass: [0.1, 0.2],
        waveform_mids: [0.2, 0.3],
        waveform_highs: [0.3, 0.4],
        waveform_times: [0, 1],
        duration: 120,
      };

      const mockResponse = { data: mockLoadTrackResponse };

      mockGet.mockResolvedValue(mockResponse);

      const result = await trackService.loadTrack("track1.mp3");

      expect(result.bpm).toBeNull();
    });

    it("handles 404 error for missing track", async () => {
      const mockError = {
        response: {
          status: 404,
          data: { error: "Track not found" },
        },
      };

      mockGet.mockRejectedValue(mockError);

      await expect(trackService.loadTrack("nonexistent.mp3")).rejects.toMatchObject(mockError);
    });

    it("handles 500 error during track load", async () => {
      const mockError = {
        response: {
          status: 500,
          data: { error: "Failed to process waveform" },
        },
      };

      mockGet.mockRejectedValue(mockError);

      await expect(trackService.loadTrack("track1.mp3")).rejects.toMatchObject(mockError);
    });

    it("handles network timeout", async () => {
      const mockError = new Error("timeout of 5000ms exceeded");

      mockGet.mockRejectedValue(mockError);

      await expect(trackService.loadTrack("large-track.mp3")).rejects.toThrow("timeout");
    });
  });

  describe("saveAnnotation", () => {
    it("saves annotation with correct payload structure", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0.5,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 10, label: "buildup" },
          { time: 20, label: "breakdown" },
        ],
      };

      const mockResponse = {
        data: {
          success: true,
          output: "test_output.txt",
          boundaries_count: 3,
        },
      };

      mockPost.mockResolvedValue(mockResponse);

      const result = await trackService.saveAnnotation(mockRequest);

      expect(mockPost).toHaveBeenCalledWith("/save", mockRequest);
      expect(result).toEqual(mockResponse.data);
    });

    it("saves annotation with single boundary", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
      };

      const mockResponse = {
        data: {
          success: true,
          output: "test_output.txt",
          boundaries_count: 1,
        },
      };

      mockPost.mockResolvedValue(mockResponse);

      const result = await trackService.saveAnnotation(mockRequest);

      expect(result.boundaries_count).toBe(1);
    });

    it("saves annotation with multiple boundaries", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 140,
        downbeat: 0.25,
        boundaries: [
          { time: 0, label: "intro" },
          { time: 16, label: "buildup" },
          { time: 32, label: "breakdown" },
          { time: 48, label: "breakbuild" },
          { time: 64, label: "outro" },
        ],
      };

      const mockResponse = {
        data: {
          success: true,
          output: "test_output_5_boundaries.txt",
          boundaries_count: 5,
        },
      };

      mockPost.mockResolvedValue(mockResponse);

      const result = await trackService.saveAnnotation(mockRequest);

      expect(result.success).toBe(true);
      expect(result.boundaries_count).toBe(5);
    });

    it("includes all required fields in request", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0.5,
        boundaries: [{ time: 0, label: "intro" }],
      };

      mockPost.mockResolvedValue({
        data: { success: true, output: "test.txt", boundaries_count: 1 },
      });

      await trackService.saveAnnotation(mockRequest);

      const calledWith = mockPost.mock.calls[0][1];
      expect(calledWith).toHaveProperty("filename");
      expect(calledWith).toHaveProperty("bpm");
      expect(calledWith).toHaveProperty("downbeat");
      expect(calledWith).toHaveProperty("boundaries");
    });

    it("handles validation errors (400)", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: -10, // Invalid BPM
        downbeat: 0,
        boundaries: [],
      };

      const mockError = {
        response: {
          status: 400,
          data: { error: "Invalid BPM value" },
        },
      };

      mockPost.mockRejectedValue(mockError);

      await expect(trackService.saveAnnotation(mockRequest)).rejects.toMatchObject(mockError);
    });

    it("handles server error during save (500)", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
      };

      const mockError = {
        response: {
          status: 500,
          data: { error: "Failed to write annotation file" },
        },
      };

      mockPost.mockRejectedValue(mockError);

      await expect(trackService.saveAnnotation(mockRequest)).rejects.toMatchObject(mockError);
    });

    it("handles network error during save", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
      };

      const mockError = new Error("Network Error");

      mockPost.mockRejectedValue(mockError);

      await expect(trackService.saveAnnotation(mockRequest)).rejects.toThrow("Network Error");
    });

    it("returns success response with correct structure", async () => {
      const mockRequest: SaveAnnotationRequest = {
        filename: "track1.mp3",
        bpm: 128,
        downbeat: 0,
        boundaries: [{ time: 0, label: "intro" }],
      };

      const mockResponse = {
        data: {
          success: true,
          output: "annotations/track1.txt",
          boundaries_count: 1,
        },
      };

      mockPost.mockResolvedValue(mockResponse);

      const result = await trackService.saveAnnotation(mockRequest);

      expect(result).toHaveProperty("success");
      expect(result).toHaveProperty("output");
      expect(result).toHaveProperty("boundaries_count");
      expect(result.success).toBe(true);
    });
  });

  describe("API Configuration", () => {
    it("uses correct base URL", () => {
      // The axios instance should be created with baseURL: "/api"
      expect(createCalls.length).toBeGreaterThan(0);
      expect(createCalls[0][0]).toMatchObject({
        baseURL: "/api",
      });
    });

    it("sets correct content-type header", () => {
      expect(createCalls.length).toBeGreaterThan(0);
      expect(createCalls[0][0]).toMatchObject({
        headers: {
          "Content-Type": "application/json",
        },
      });
    });
  });
});
