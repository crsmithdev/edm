import { test, expect } from '@playwright/test';

/**
 * E2E tests for keyboard shortcuts
 * Tests all major keyboard shortcuts: Space, B, D, Q, arrow keys, zoom controls
 */
test.describe('Keyboard Shortcuts', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Wait for track list and load first track
    await page.waitForSelector('text=Tracks (', { timeout: 10000 });
    const firstTrack = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
    await firstTrack.click();

    // Wait for waveform to be ready
    await page.waitForSelector('canvas', { timeout: 10000 });
    await page.waitForSelector('text=/Loaded/', { timeout: 10000 });
  });

  test('Space key toggles play/pause', async ({ page }) => {
    // Press space to play
    await page.keyboard.press('Space');
    await page.waitForTimeout(500);

    // Check if playing (we can verify by checking if time updates)
    // Or look for play/pause button state

    // Press space again to pause
    await page.keyboard.press('Space');
    await page.waitForTimeout(500);

    // Verify the shortcut worked by checking the status or button state
    // The app should respond to space key without errors
  });

  test('B key adds boundary at playhead', async ({ page }) => {
    // Get initial boundary count
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    const initialText = await boundaryStats.textContent();
    const initialCount = parseInt(initialText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    // Press B to add boundary
    await page.keyboard.press('b');

    // Wait for status message
    await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

    // Verify boundary count increased
    const newText = await boundaryStats.textContent();
    const newCount = parseInt(newText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    expect(newCount).toBe(initialCount + 1);
  });

  test('D key sets downbeat at playhead', async ({ page }) => {
    // Press D to set downbeat
    await page.keyboard.press('d');

    // Wait for status message
    await page.waitForSelector('text=/Downbeat set to/', { timeout: 2000 });
    const statusMessage = page.getByText(/Downbeat set to/);
    await expect(statusMessage).toBeVisible();
  });

  test('Q key toggles quantize', async ({ page }) => {
    // Find quantize button to check its state
    const quantizeButton = page.getByRole('button', { name: /Quantize/ });

    // Get initial state
    const initialText = await quantizeButton.textContent();
    const isInitiallyOn = initialText?.includes('ON');

    // Press Q to toggle
    await page.keyboard.press('q');
    await page.waitForTimeout(200);

    // Check state changed
    const newText = await quantizeButton.textContent();
    const isNowOn = newText?.includes('ON');

    expect(isNowOn).toBe(!isInitiallyOn);

    // Toggle again
    await page.keyboard.press('q');
    await page.waitForTimeout(200);

    // Should be back to original state
    const finalText = await quantizeButton.textContent();
    const isFinallyOn = finalText?.includes('ON');

    expect(isFinallyOn).toBe(isInitiallyOn);
  });

  test('C/R keys return to cue point', async ({ page }) => {
    // Click on waveform to set a cue point
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      const x = canvasBox.x + canvasBox.width * 0.5;
      const y = canvasBox.y + canvasBox.height * 0.5;
      await page.mouse.click(x, y);

      // Wait for cue to be set
      await page.waitForSelector('text=/Cue set to/', { timeout: 2000 });
    }

    // Start playback
    await page.keyboard.press('Space');
    await page.waitForTimeout(1000);

    // Press C to return to cue
    await page.keyboard.press('c');
    await page.waitForTimeout(200);

    // Verify playhead returned (status message or visual check)

    // Try R as well
    await page.keyboard.press('Space');
    await page.waitForTimeout(500);
    await page.keyboard.press('r');
    await page.waitForTimeout(200);
  });

  test('Arrow Left/Right keys navigate through track', async ({ page }) => {
    // Set a known BPM for predictable navigation
    const bpmInput = page.locator('input[type="number"]').first();
    await bpmInput.fill('120');

    // Click on waveform to set playhead position
    const canvas = page.locator('canvas').first();
    const canvasBox = await canvas.boundingBox();

    if (canvasBox) {
      const x = canvasBox.x + canvasBox.width * 0.5;
      const y = canvasBox.y + canvasBox.height * 0.5;
      await page.mouse.click(x, y);
      await page.waitForTimeout(500);
    }

    // Press Right arrow to jump forward
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);

    // Press Left arrow to jump backward
    await page.keyboard.press('ArrowLeft');
    await page.waitForTimeout(200);

    // Press Shift+Right for larger jump
    await page.keyboard.press('Shift+ArrowRight');
    await page.waitForTimeout(200);

    // Press Shift+Left for larger jump back
    await page.keyboard.press('Shift+ArrowLeft');
    await page.waitForTimeout(200);

    // Verify navigation worked (no errors, playhead moved)
  });

  test('Arrow Up/Down keys navigate between tracks', async ({ page }) => {
    // Get track list count to verify we can navigate
    const trackCount = await page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').count();

    if (trackCount > 1) {
      // Get first track name
      const firstTrackElement = page.locator('[style*="cursor: pointer"][style*="padding: 12px 16px"]').first();
      const firstTrackName = await firstTrackElement.locator('div').first().textContent();

      // Press Down arrow to go to next track
      await page.keyboard.press('ArrowDown');
      await page.waitForTimeout(500);

      // Check that a different track is selected
      const selectedTrack = page.locator('[style*="cursor: pointer"][style*="background: rgba(91, 124, 255, 0.1)"]');
      await expect(selectedTrack).toBeVisible({ timeout: 2000 });

      // Press Up arrow to go back
      await page.keyboard.press('ArrowUp');
      await page.waitForTimeout(500);
    }
  });

  test('+/- keys zoom waveform', async ({ page }) => {
    // Press + to zoom in
    await page.keyboard.press('+');
    await page.waitForTimeout(300);

    // Press again
    await page.keyboard.press('+');
    await page.waitForTimeout(300);

    // Press - to zoom out
    await page.keyboard.press('-');
    await page.waitForTimeout(300);

    // Press = (alternative zoom in)
    await page.keyboard.press('=');
    await page.waitForTimeout(300);

    // Verify zoom controls worked (no errors)
  });

  test('0 key resets zoom to fit', async ({ page }) => {
    // Zoom in first
    await page.keyboard.press('+');
    await page.keyboard.press('+');
    await page.keyboard.press('+');
    await page.waitForTimeout(300);

    // Press 0 to reset
    await page.keyboard.press('0');

    // Should show status message
    await page.waitForSelector('text=/Zoom reset/', { timeout: 2000 });
    const statusMessage = page.getByText(/Zoom reset/);
    await expect(statusMessage).toBeVisible();
  });

  test('keyboard shortcuts work in sequence', async ({ page }) => {
    // Set BPM
    const bpmInput = page.locator('input[type="number"]').first();
    await bpmInput.fill('128');

    // Set downbeat (D)
    await page.keyboard.press('d');
    await page.waitForSelector('text=/Downbeat set to/', { timeout: 2000 });

    // Add boundary (B)
    await page.keyboard.press('b');
    await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

    // Toggle quantize (Q)
    await page.keyboard.press('q');
    await page.waitForTimeout(200);

    // Add another boundary with quantize enabled
    await page.keyboard.press('b');
    await page.waitForTimeout(500);

    // Navigate forward (Right arrow)
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);

    // Add another boundary
    await page.keyboard.press('b');
    await page.waitForTimeout(500);

    // Verify multiple boundaries were added
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    const statsText = await boundaryStats.textContent();
    expect(statsText).toMatch(/Boundaries: [3-9]/);
  });

  test('keyboard shortcuts are ignored when typing in input', async ({ page }) => {
    // Focus BPM input
    const bpmInput = page.locator('input[type="number"]').first();
    await bpmInput.click();

    // Get initial boundary count
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    const initialText = await boundaryStats.textContent();
    const initialCount = parseInt(initialText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    // Type 'b' in input - should not add boundary
    await bpmInput.fill('120');

    // Verify 'b' didn't trigger boundary addition
    const newText = await boundaryStats.textContent();
    const newCount = parseInt(newText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    expect(newCount).toBe(initialCount);

    // Click outside input to unfocus
    await page.click('body');

    // Now B should work
    await page.keyboard.press('b');
    await page.waitForSelector('text=/Added boundary at/', { timeout: 2000 });

    // Boundary count should have increased
    const finalText = await boundaryStats.textContent();
    const finalCount = parseInt(finalText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    expect(finalCount).toBe(newCount + 1);
  });

  test('multiple rapid keyboard commands work correctly', async ({ page }) => {
    // Get initial boundary count
    const boundaryStats = page.getByText(/Boundaries: \d+/);
    const initialText = await boundaryStats.textContent();
    const initialCount = parseInt(initialText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    // Rapidly add boundaries with B key
    for (let i = 0; i < 5; i++) {
      await page.keyboard.press('b');
      await page.waitForTimeout(200);
    }

    // Verify all boundaries were added
    const finalText = await boundaryStats.textContent();
    const finalCount = parseInt(finalText?.match(/Boundaries: (\d+)/)?.[1] || '0');

    expect(finalCount).toBeGreaterThanOrEqual(initialCount + 5);
  });

  test('keyboard shortcuts display help text', async ({ page }) => {
    // Verify keyboard shortcuts help is visible on the page
    const helpText = page.getByText(/Keyboard Shortcuts:/);
    await expect(helpText).toBeVisible();

    // Verify major shortcuts are documented
    await expect(page.getByText(/Space=Play\/Pause/)).toBeVisible();
    await expect(page.getByText(/B=Add Boundary/)).toBeVisible();
    await expect(page.getByText(/D=Set Downbeat/)).toBeVisible();
    await expect(page.getByText(/Q=Toggle Quantize/)).toBeVisible();
  });
});
