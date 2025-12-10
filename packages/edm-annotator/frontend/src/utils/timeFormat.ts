/**
 * Format time in seconds as MM:SS.mmm
 */
export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  const secs = Math.floor(remainingSeconds);
  // Add small epsilon to handle floating point precision before floor
  const millisFloat = (remainingSeconds - secs) * 1000;
  const millis = Math.floor(millisFloat + 1e-10);
  return `${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}.${Math.abs(millis).toString().padStart(3, "0")}`;
}

/**
 * Format time in seconds as MM:SS
 */
export function formatTimeShort(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}
