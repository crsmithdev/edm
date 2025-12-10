/**
 * Convert time to bar number (1-indexed)
 */
export function timeToBar(time: number, bpm: number, downbeat: number): number {
  const barDuration = (60.0 / bpm) * 4.0; // Duration of one bar in 4/4 time
  const bar = Math.floor((time - downbeat) / barDuration) + 1;
  return Math.max(1, bar);
}

/**
 * Convert bar number to time
 */
export function barToTime(bar: number, bpm: number, downbeat: number): number {
  const barDuration = (60.0 / bpm) * 4.0;
  return downbeat + (bar - 1) * barDuration;
}

/**
 * Get beat duration in seconds
 */
export function getBeatDuration(bpm: number): number {
  return 60.0 / bpm;
}

/**
 * Get bar duration in seconds
 */
export function getBarDuration(bpm: number): number {
  return (60.0 / bpm) * 4.0;
}
