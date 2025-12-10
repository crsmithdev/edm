/**
 * Custom rounding for beat quantization
 * - 0.5 rounds up to 1
 * - All other X.5 values round toward zero
 */
function roundHalfDownBeat(x: number): number {
  const fractional = x - Math.trunc(x);

  // Special case: 0.5 rounds up to 1
  if (Math.abs(x - 0.5) < 1e-10) {
    return 1;
  }

  // Tie case (exactly X.5): round toward zero
  if (Math.abs(Math.abs(fractional) - 0.5) < 1e-10) {
    return Math.trunc(x);
  }

  // Not a tie, use standard rounding
  return Math.round(x);
}

/**
 * Custom rounding for bar quantization
 * - All X.5 values round toward zero
 */
function roundHalfDownBar(x: number): number {
  const fractional = x - Math.trunc(x);

  // Tie case (exactly X.5): round toward zero
  if (Math.abs(Math.abs(fractional) - 0.5) < 1e-10) {
    return Math.trunc(x);
  }

  // Not a tie, use standard rounding
  return Math.round(x);
}

/**
 * Quantize time to nearest beat boundary
 */
export function quantizeToBeat(
  time: number,
  bpm: number,
  downbeat: number
): number {
  const beatDuration = 60.0 / bpm;
  const relativeTime = time - downbeat;
  const rawIndex = relativeTime / beatDuration;
  // Round to 10 decimals to handle floating point errors
  const cleanIndex = Math.round(rawIndex * 1e10) / 1e10;
  const beatIndex = roundHalfDownBeat(cleanIndex);
  return downbeat + beatIndex * beatDuration;
}

/**
 * Quantize time to nearest bar boundary
 */
export function quantizeToBar(
  time: number,
  bpm: number,
  downbeat: number
): number {
  const barDuration = (60.0 / bpm) * 4.0;
  const relativeTime = time - downbeat;
  const rawIndex = relativeTime / barDuration;
  // Round to 10 decimals to handle floating point errors
  const cleanIndex = Math.round(rawIndex * 1e10) / 1e10;
  const barIndex = roundHalfDownBar(cleanIndex);
  return downbeat + barIndex * barDuration;
}
