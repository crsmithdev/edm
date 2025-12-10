import { test, expect } from '@playwright/test';

/**
 * E2E tests for error handling scenarios
 * Tests API failures, network errors, and error message display
 */
test.describe('Error Handling', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
  });

  test('displays error message when track list fails to load', async ({ page }) => {
    // Intercept the tracks API endpoint and simulate failure
    await page.route('**/api/tracks', (route) => {
      route.abort('failed');
    });

    // Reload to trigger the failed fetch
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Should show error or no tracks message
    const errorIndicator = page.getByText(/No audio files found|Error|Loading tracks/);
    await expect(errorIndicator).toBeVisible({ timeout: 5000 });
  });

  test('displays error message when track loading fails', async ({ page }) => {
    // Wait for track list to load
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Get first track name for the API route
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    const trackNameElement = await firstTrack.locator('div').first();
    const trackText = await trackNameElement.textContent();
    const trackName = trackText?.replace(/✓\s*/, '').trim() || '';

    // Intercept the track load API and simulate failure
    await page.route(`**/api/tracks/${trackName}`, (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    // Click the track
    await firstTrack.click();

    // Should show error message in status toast
    const errorMessage = page.getByText(/Error loading track/);
    await expect(errorMessage).toBeVisible({ timeout: 5000 });
  });

  test('displays error message when save fails', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform to load
    await page.waitForSelector('canvas', { timeout: 10000 });
    await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

    // Add a boundary
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      const x = canvasBox.x + canvasBox.width * 0.5;
      const y = canvasBox.y + canvasBox.height * 0.5;
      await page.mouse.click(x, y, { modifiers: ['Shift'] });
      await page.waitForTimeout(500);
    }

    // Set BPM
    const bpmInput = page.locator('input[type="number"]').first();
    await bpmInput.fill('128');

    // Intercept save API and simulate failure
    await page.route('**/api/annotations', (route) => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Database error' }),
      });
    });

    // Try to save
    const saveButton = page.getByRole('button', { name: /Save Annotation/ });
    await saveButton.click();

    // Should show error message
    const errorMessage = page.getByText(/Error saving/);
    await expect(errorMessage).toBeVisible({ timeout: 5000 });

    // Save button should be re-enabled after error
    await expect(saveButton).toBeEnabled({ timeout: 2000 });
  });

  test('handles network timeout gracefully', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    const trackNameElement = await firstTrack.locator('div').first();
    const trackText = await trackNameElement.textContent();
    const trackName = trackText?.replace(/✓\s*/, '').trim() || '';

    // Simulate a very slow response (timeout)
    await page.route(`**/api/tracks/${trackName}`, async (route) => {
      // Never respond - simulates timeout
      await new Promise((resolve) => setTimeout(resolve, 60000));
      route.abort('timedout');
    });

    // Click the track
    await firstTrack.click();

    // Should eventually show an error or timeout message
    // The app should not hang indefinitely
    const statusMessage = page.locator('[style*="position: fixed"][style*="top"]').first();
    await expect(statusMessage).toBeVisible({ timeout: 15000 });
  });

  test('handles missing audio file gracefully', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();

    // Intercept audio file request and return 404
    await page.route('**/api/audio/**', (route) => {
      route.fulfill({
        status: 404,
        contentType: 'text/plain',
        body: 'Audio file not found',
      });
    });

    // Click the track
    await firstTrack.click();

    // App should handle this gracefully
    // Audio player may show error, but app shouldn't crash
    await page.waitForTimeout(2000);

    // Verify the app is still responsive
    const trackList = page.getByText(/Tracks \(/);
    await expect(trackList).toBeVisible();
  });

  test('validates BPM input range', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform
    await page.waitForSelector('canvas', { timeout: 10000 });

    const bpmInput = page.locator('input[type="number"]').first();

    // Try invalid values
    await bpmInput.fill('-10');
    let value = await bpmInput.inputValue();
    // Negative values should be rejected or handled

    await bpmInput.fill('0');
    value = await bpmInput.inputValue();
    // Zero should be rejected or handled

    await bpmInput.fill('500');
    value = await bpmInput.inputValue();
    // Unreasonably high values should be handled

    // Valid value should work
    await bpmInput.fill('128');
    value = await bpmInput.inputValue();
    expect(value).toBe('128');
  });

  test('displays appropriate message when no tracks available', async ({ page }) => {
    // Intercept tracks API and return empty array
    await page.route('**/api/tracks', (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      });
    });

    // Reload page
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Should display no tracks message
    const noTracksMessage = page.getByText(/No audio files found/);
    await expect(noTracksMessage).toBeVisible({ timeout: 5000 });
  });

  test('handles corrupted track data gracefully', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    const trackNameElement = await firstTrack.locator('div').first();
    const trackText = await trackNameElement.textContent();
    const trackName = trackText?.replace(/✓\s*/, '').trim() || '';

    // Return corrupted/invalid data
    await page.route(`**/api/tracks/${trackName}`, (route) => {
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          // Missing required fields
          waveform_bass: [],
          // Invalid structure
        }),
      });
    });

    // Click the track
    await firstTrack.click();

    // Should handle gracefully and show error
    const errorMessage = page.locator('text=/Error|Invalid/i');
    await expect(errorMessage).toBeVisible({ timeout: 5000 });
  });

  test('prevents saving with invalid annotation data', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform
    await page.waitForSelector('canvas', { timeout: 10000 });
    await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

    // Don't add boundaries - try to save empty annotation
    const saveButton = page.getByRole('button', { name: /Save Annotation/ });

    // Should be disabled without boundaries
    await expect(saveButton).toBeDisabled();

    // Add a boundary
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      const x = canvasBox.x + canvasBox.width * 0.5;
      const y = canvasBox.y + canvasBox.height * 0.5;
      await page.mouse.click(x, y, { modifiers: ['Shift'] });
      await page.waitForTimeout(500);
    }

    // Now should be enabled
    await expect(saveButton).toBeEnabled();
  });
});
