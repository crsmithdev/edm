/**
 * Tempo and timing utilities for bar/beat calculations
 */

/**
 * Calculate the duration of a single beat in seconds
 * @param bpm Beats per minute (must be > 0)
 * @returns Beat duration in seconds
 */
export function getBeatDuration(bpm: number): number {
  if (bpm <= 0) return 0;
  return 60.0 / bpm;
}

/**
 * Calculate the duration of a bar (4 beats) in seconds
 * Assumes 4/4 time signature
 * @param bpm Beats per minute (must be > 0)
 * @returns Bar duration in seconds
 */
export function getBarDuration(bpm: number): number {
  if (bpm <= 0) return 0;
  return (60.0 / bpm) * 4.0;
}

/**
 * Quantize a time value to the nearest beat
 * @param time Time in seconds
 * @param bpm Beats per minute
 * @param downbeat Downbeat offset in seconds (default: 0)
 * @returns Quantized time in seconds
 */
export function quantizeToBeat(time: number, bpm: number, downbeat: number = 0): number {
  if (bpm <= 0) return time;

  const beatDuration = getBeatDuration(bpm);
  const relativeTime = time - downbeat;
  const beatNumber = Math.round(relativeTime / beatDuration);

  return downbeat + beatNumber * beatDuration;
}

/**
 * Quantize a time value to the nearest bar
 * @param time Time in seconds
 * @param bpm Beats per minute
 * @param downbeat Downbeat offset in seconds (default: 0)
 * @returns Quantized time in seconds
 */
export function quantizeToBar(time: number, bpm: number, downbeat: number = 0): number {
  if (bpm <= 0) return time;

  const barDuration = getBarDuration(bpm);
  const relativeTime = time - downbeat;
  const barNumber = Math.round(relativeTime / barDuration);

  return downbeat + barNumber * barDuration;
}

/**
 * Calculate bar number at given time
 * @param time Time in seconds
 * @param bpm Beats per minute
 * @param downbeat Downbeat offset in seconds (default: 0)
 * @returns Bar number (1-indexed)
 */
export function getBarNumber(time: number, bpm: number, downbeat: number = 0): number {
  if (bpm <= 0) return 1;

  const barDuration = getBarDuration(bpm);
  const relativeTime = Math.max(0, time - downbeat);

  return Math.floor(relativeTime / barDuration) + 1;
}

/**
 * Alias for getBarNumber
 * @param time Time in seconds
 * @param bpm Beats per minute
 * @param downbeat Downbeat offset in seconds (default: 0)
 * @returns Bar number (1-indexed)
 */
export function timeToBar(time: number, bpm: number, downbeat: number = 0): number {
  return getBarNumber(time, bpm, downbeat);
}

/**
 * Convert bar number to time in seconds
 * @param bar Bar number (1-indexed)
 * @param bpm Beats per minute
 * @param downbeat Downbeat offset in seconds (default: 0)
 * @returns Time in seconds
 */
export function barToTime(bar: number, bpm: number, downbeat: number = 0): number {
  if (bpm <= 0) return downbeat;

  const barDuration = getBarDuration(bpm);
  return downbeat + (bar - 1) * barDuration;
}
