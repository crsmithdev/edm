import { Page, expect } from '@playwright/test';

/**
 * Common test helpers for E2E tests
 */

/**
 * Load the first available track in the track list
 */
export async function loadFirstTrack(page: Page): Promise<string> {
  // Wait for track list to load
  await page.waitForSelector('text=Tracks (', { timeout: 10000 });

  // Get first track
  const firstTrack = page
    .locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]')
    .first();

  // Get track name
  const trackNameElement = await firstTrack.locator('div').first();
  const trackText = await trackNameElement.textContent();
  const trackName = trackText?.replace(/✓\s*/, '').trim() || '';

  // Click to load
  await firstTrack.click();

  // Wait for waveform to load
  await page.waitForSelector('canvas', { timeout: 10000 });
  await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

  return trackName;
}

/**
 * Add a boundary at a specific position on the waveform
 * @param page Playwright page
 * @param positionRatio Position from 0.0 (left) to 1.0 (right)
 */
export async function addBoundaryAtPosition(
  page: Page,
  positionRatio: number = 0.5
): Promise<void> {
  const canvas = page.locator('canvas').first();
  const canvasBox = await canvas.boundingBox();

  if (!canvasBox) {
    throw new Error('Canvas not found or not visible');
  }

  const x = canvasBox.x + canvasBox.width * positionRatio;
  const y = canvasBox.y + canvasBox.height * 0.5;

  await page.mouse.click(x, y, { modifiers: ['Shift'] });
  await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });
}

/**
 * Add multiple boundaries at evenly spaced positions
 */
export async function addMultipleBoundaries(
  page: Page,
  count: number
): Promise<void> {
  for (let i = 1; i <= count; i++) {
    const position = i / (count + 1);
    await addBoundaryAtPosition(page, position);
    await page.waitForTimeout(200); // Small delay between additions
  }
}

/**
 * Get the current boundary count from the stats display
 */
export async function getBoundaryCount(page: Page): Promise<number> {
  const boundaryStats = page.getByText(/Boundaries: \d+/);
  const statsText = await boundaryStats.textContent();
  return parseInt(statsText?.match(/Boundaries: (\d+)/)?.[1] || '0');
}

/**
 * Get the current region count from the stats display
 */
export async function getRegionCount(page: Page): Promise<number> {
  const regionStats = page.getByText(/Regions: \d+/);
  const statsText = await regionStats.textContent();
  return parseInt(statsText?.match(/Regions: (\d+)/)?.[1] || '0');
}

/**
 * Set BPM value
 */
export async function setBPM(page: Page, bpm: number): Promise<void> {
  const bpmInput = page.locator('input[type="number"]').first();
  await bpmInput.fill(bpm.toString());
  await expect(bpmInput).toHaveValue(bpm.toString());
}

/**
 * Click the save button and wait for success
 */
export async function saveAnnotation(page: Page): Promise<void> {
  const saveButton = page.getByRole('button', { name: /Save Annotation/ });
  await expect(saveButton).toBeVisible();
  await expect(saveButton).toBeEnabled();
  await saveButton.click();

  // Wait for success message
  await page.waitForSelector('text=/Annotation saved successfully/', {
    timeout: 5000,
  });
}

/**
 * Toggle quantize on or off
 */
export async function toggleQuantize(page: Page): Promise<void> {
  const quantizeButton = page.getByRole('button', { name: /Quantize/ });
  await quantizeButton.click();
  await page.waitForTimeout(200);
}

/**
 * Check if quantize is enabled
 */
export async function isQuantizeEnabled(page: Page): Promise<boolean> {
  const quantizeButton = page.getByRole('button', { name: /Quantize/ });
  const text = await quantizeButton.textContent();
  return text?.includes('ON') || false;
}

/**
 * Click on the waveform to set a cue point
 */
export async function setCuePoint(
  page: Page,
  positionRatio: number = 0.5
): Promise<void> {
  const canvas = page.locator('canvas').first();
  const canvasBox = await canvas.boundingBox();

  if (!canvasBox) {
    throw new Error('Canvas not found or not visible');
  }

  const x = canvasBox.x + canvasBox.width * positionRatio;
  const y = canvasBox.y + canvasBox.height * 0.5;

  await page.mouse.click(x, y);
  await page.waitForSelector('text=/Cue set to/', { timeout: 2000 });
}

/**
 * Get track count from header
 */
export async function getTrackCount(page: Page): Promise<number> {
  const trackListHeader = await page.getByText(/Tracks \((\d+)\)/);
  const headerText = await trackListHeader.textContent();
  return parseInt(headerText?.match(/Tracks \((\d+)\)/)?.[1] || '0');
}

/**
 * Wait for status message with specific text
 */
export async function waitForStatus(
  page: Page,
  textPattern: string | RegExp,
  timeout: number = 5000
): Promise<void> {
  await page.waitForSelector(`text=/${textPattern.toString()}/`, { timeout });
}

/**
 * Verify track has reference annotation
 */
export async function verifyTrackHasReference(
  page: Page,
  trackName: string
): Promise<void> {
  // Reload page to see updated status
  await page.reload();
  await page.waitForLoadState('networkidle');
  await page.waitForSelector('text=Tracks (', { timeout: 10000 });

  // Find the track and check for reference label
  const trackLocator = page.locator(`text=/${trackName}/`).first();
  const trackContainer = trackLocator.locator('../..');
  const referenceLabel = trackContainer.getByText('Reference');

  await expect(referenceLabel).toBeVisible({ timeout: 5000 });
}
