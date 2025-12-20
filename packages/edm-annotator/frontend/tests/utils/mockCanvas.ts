import { vi } from "vitest";

/**
 * Mock implementation of CanvasRenderingContext2D for testing canvas-based components
 */
export class MockCanvasRenderingContext2D {
  // Drawing rectangles
  fillRect = vi.fn();
  strokeRect = vi.fn();
  clearRect = vi.fn();

  // Drawing paths
  beginPath = vi.fn();
  closePath = vi.fn();
  moveTo = vi.fn();
  lineTo = vi.fn();
  bezierCurveTo = vi.fn();
  quadraticCurveTo = vi.fn();
  arc = vi.fn();
  arcTo = vi.fn();
  ellipse = vi.fn();
  rect = vi.fn();

  // Drawing
  stroke = vi.fn();
  fill = vi.fn();
  clip = vi.fn();

  // Styles
  fillStyle: string | CanvasGradient | CanvasPattern = "#000000";
  strokeStyle: string | CanvasGradient | CanvasPattern = "#000000";
  lineWidth = 1;
  lineCap: CanvasLineCap = "butt";
  lineJoin: CanvasLineJoin = "miter";
  globalAlpha = 1;
  globalCompositeOperation: GlobalCompositeOperation = "source-over";
  shadowColor = "rgba(0, 0, 0, 0)";
  shadowBlur = 0;
  shadowOffsetX = 0;
  shadowOffsetY = 0;

  // Transformations
  save = vi.fn();
  restore = vi.fn();
  scale = vi.fn();
  rotate = vi.fn();
  translate = vi.fn();
  transform = vi.fn();
  setTransform = vi.fn();
  resetTransform = vi.fn();

  // Text
  font = "10px sans-serif";
  textAlign: CanvasTextAlign = "start";
  textBaseline: CanvasTextBaseline = "alphabetic";
  fillText = vi.fn();
  strokeText = vi.fn();
  measureText = vi.fn((text: string) => ({
    width: text.length * 8,
    actualBoundingBoxAscent: 8,
    actualBoundingBoxDescent: 2,
    actualBoundingBoxLeft: 0,
    actualBoundingBoxRight: text.length * 8,
    fontBoundingBoxAscent: 10,
    fontBoundingBoxDescent: 2,
  }));

  // Images
  drawImage = vi.fn();
  createImageData = vi.fn();
  getImageData = vi.fn();
  putImageData = vi.fn();

  // Gradients and patterns
  createLinearGradient = vi.fn();
  createRadialGradient = vi.fn();
  createPattern = vi.fn();

  // Line styles
  setLineDash = vi.fn();
  getLineDash = vi.fn(() => []);
  lineDashOffset = 0;

  // Canvas state
  canvas = {
    width: 800,
    height: 600,
  } as HTMLCanvasElement;

  /**
   * Reset all mock function calls
   */
  reset() {
    vi.clearAllMocks();
  }
}

/**
 * Creates a mock canvas element with a mock 2D context
 * @returns Object containing the canvas element and mock context
 */
export function mockCanvasElement() {
  const context = new MockCanvasRenderingContext2D();
  const canvas = document.createElement("canvas");

  // Mock getContext to return our mock context
  vi.spyOn(canvas, "getContext").mockReturnValue(context as any);

  return { canvas, context };
}

/**
 * Sets up canvas mocking for a test suite
 * Call this in beforeEach to ensure createElement("canvas") returns a mocked canvas
 */
export function setupCanvasMock() {
  const originalCreateElement = document.createElement.bind(document);

  vi.spyOn(document, "createElement").mockImplementation((tagName: string) => {
    if (tagName === "canvas") {
      const { canvas } = mockCanvasElement();
      return canvas;
    }
    return originalCreateElement(tagName);
  });
}
