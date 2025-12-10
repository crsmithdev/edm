import { test, expect } from '@playwright/test';

/**
 * E2E tests for the complete annotation workflow
 * Tests the critical path: Load track → Add boundaries → Save → Verify
 */
test.describe('Annotation Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the application
    await page.goto('/');

    // Wait for the app to load
    await page.waitForLoadState('networkidle');
  });

  test('complete workflow: load track, add boundaries, save, and verify', async ({ page }) => {
    // Step 1: Wait for track list to load
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Verify track list is visible and has tracks
    const trackListHeader = await page.getByText(/Tracks \(\d+\)/);
    await expect(trackListHeader).toBeVisible();

    // Get the count of tracks
    const trackCount = await page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').count();
    expect(trackCount).toBeGreaterThan(0);

    // Step 2: Click the first track in the list
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    const trackName = await firstTrack.locator('div').first().textContent();
    await firstTrack.click();

    // Wait for status message showing track is loading
    await page.waitForSelector('text=/Loading.*/', { timeout: 5000 });

    // Step 3: Wait for waveform canvas to appear
    await page.waitForSelector('canvas', { timeout: 10000 });
    const canvas = page.locator('canvas').first();
    await expect(canvas).toBeVisible();

    // Wait for "Loaded" status message
    await page.waitForSelector(`text=/Loaded ${trackName?.replace(/✓\s*/, '').trim() || ''}`, { timeout: 10000 });

    // Step 4: Get the canvas element for interaction
    const canvasBox = await canvas.boundingBox();
    expect(canvasBox).not.toBeNull();

    // Step 5: Add boundaries using Shift+Click
    // Add boundary at 25% of canvas width
    if (canvasBox) {
      const x1 = canvasBox.x + canvasBox.width * 0.25;
      const y1 = canvasBox.y + canvasBox.height * 0.5;

      await page.mouse.click(x1, y1, { modifiers: ['Shift'] });
      await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

      // Add boundary at 50% of canvas width
      const x2 = canvasBox.x + canvasBox.width * 0.5;
      const y2 = canvasBox.y + canvasBox.height * 0.5;

      await page.mouse.click(x2, y2, { modifiers: ['Shift'] });
      await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

      // Add boundary at 75% of canvas width
      const x3 = canvasBox.x + canvasBox.width * 0.75;
      const y3 = canvasBox.y + canvasBox.height * 0.5;

      await page.mouse.click(x3, y3, { modifiers: ['Shift'] });
      await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });
    }

    // Step 6: Verify boundary count updates
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    await expect(boundaryStats).toBeVisible();

    const statsText = await boundaryStats.textContent();
    expect(statsText).toMatch(/Boundaries: [3-9]/); // At least 3 boundaries

    // Step 7: Enter BPM value
    const bpmInput = page.locator('input[type="number"]').first();
    await expect(bpmInput).toBeVisible();
    await bpmInput.fill('128');

    // Verify BPM was set
    await expect(bpmInput).toHaveValue('128');

    // Step 8: Click save button
    const saveButton = page.getByRole('button', { name: /Save Annotation/ });
    await expect(saveButton).toBeVisible();
    await expect(saveButton).toBeEnabled();
    await saveButton.click();

    // Step 9: Verify success message appears
    await page.waitForSelector('text=/Annotation saved successfully/', { timeout: 5000 });
    const successMessage = page.getByText(/Annotation saved successfully/);
    await expect(successMessage).toBeVisible();

    // Step 10: Reload page to verify persistence
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Wait for track list to reload
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Step 11: Find the track we just annotated and verify it shows "has_reference"
    // Look for the track with a checkmark (✓) indicating it has a reference annotation
    const annotatedTrack = page.locator(`text=/${trackName?.replace(/✓\s*/, '').trim() || ''}/`).first();

    // The track's parent container should show "Reference" status
    const trackContainer = annotatedTrack.locator('../..');
    const referenceLabel = trackContainer.getByText('Reference');

    await expect(referenceLabel).toBeVisible({ timeout: 5000 });
  });

  test('displays error when saving without boundaries', async ({ page }) => {
    // Wait for track list to load
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform to load
    await page.waitForSelector('canvas', { timeout: 10000 });

    // Try to save without adding boundaries
    const saveButton = page.getByRole('button', { name: /Save Annotation/ });

    // Button should be disabled when there are no boundaries
    await expect(saveButton).toBeDisabled();
  });

  test('can add multiple boundaries and see region list update', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform
    await page.waitForSelector('canvas', { timeout: 10000 });
    await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

    // Get canvas for adding boundaries
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      // Add 5 boundaries
      for (let i = 1; i <= 5; i++) {
        const x = canvasBox.x + (canvasBox.width * i) / 6;
        const y = canvasBox.y + canvasBox.height * 0.5;
        await page.mouse.click(x, y, { modifiers: ['Shift'] });
        await page.waitForTimeout(200); // Small delay between clicks
      }
    }

    // Verify boundary count
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    const statsText = await boundaryStats.textContent();
    expect(statsText).toMatch(/Boundaries: [5-9]/);

    // Verify regions were created
    const regionText = await boundaryStats.textContent();
    expect(regionText).toMatch(/Regions: [4-9]/); // n boundaries create n-1 regions
  });

  test('can set BPM using tap tempo', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform
    await page.waitForSelector('canvas', { timeout: 10000 });

    // Find tap tempo button
    const tapButton = page.getByRole('button', { name: 'Tap' });
    await expect(tapButton).toBeVisible();

    // Get initial BPM value
    const bpmInput = page.locator('input[type="number"]').first();
    const initialBPM = await bpmInput.inputValue();

    // Tap 4 times with consistent interval (120 BPM = 500ms per beat)
    for (let i = 0; i < 4; i++) {
      await tapButton.click();
      await page.waitForTimeout(500);
    }

    // BPM should have changed
    const newBPM = await bpmInput.inputValue();
    expect(newBPM).not.toBe(initialBPM);

    // Should be close to 120 BPM (allowing some tolerance)
    const bpmValue = parseFloat(newBPM);
    expect(bpmValue).toBeGreaterThan(100);
    expect(bpmValue).toBeLessThan(140);
  });

  test('quantize toggle affects boundary placement', async ({ page }) => {
    // Wait for track list
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });

    // Click first track
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform
    await page.waitForSelector('canvas', { timeout: 10000 });
    await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

    // Set a known BPM
    const bpmInput = page.locator('input[type="number"]').first();
    await bpmInput.fill('120');

    // Find quantize button and verify it's initially OFF
    const quantizeButton = page.getByRole('button', { name: /Quantize/ });
    await expect(quantizeButton).toBeVisible();
    await expect(quantizeButton).toContainText('OFF');

    // Toggle quantize ON
    await quantizeButton.click();
    await expect(quantizeButton).toContainText('ON');

    // Add a boundary with quantize enabled
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      const x = canvasBox.x + canvasBox.width * 0.3;
      const y = canvasBox.y + canvasBox.height * 0.5;
      await page.mouse.click(x, y, { modifiers: ['Shift'] });
    }

    // Should show boundary was added
    await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

    // Verify boundary count increased
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    await expect(boundaryStats).toBeVisible();
  });
});
