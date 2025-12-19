# End-to-End Tests

Playwright-based E2E tests for the EDM Annotator frontend application.

## Setup

1. Install dependencies (including Playwright):
   ```bash
   npm install
   ```

2. Install Playwright browsers:
   ```bash
   npx playwright install chromium
   ```

## Running Tests

### Run all tests (headless)
```bash
npm run test:e2e
```

### Run tests with UI mode (interactive)
```bash
npm run test:e2e:ui
```

### Run tests in debug mode
```bash
npm run test:e2e:debug
```

### Run specific test file
```bash
npx playwright test annotation-workflow.spec.ts
```

### Run tests in headed mode (see browser)
```bash
npx playwright test --headed
```

## Test Suites

### 1. annotation-workflow.spec.ts (CRITICAL)
Tests the complete annotation workflow:
- Load track from track list
- Add boundaries via Shift+click
- Set BPM value
- Save annotation
- Verify persistence after reload
- Additional scenarios:
  - Saving without boundaries (validation)
  - Multiple boundaries and region list updates
  - Tap tempo functionality
  - Quantize toggle behavior

### 2. error-handling.spec.ts
Tests error scenarios:
- Track list loading failures
- Track loading failures
- Save operation failures
- Network timeouts
- Missing audio files
- BPM input validation
- Empty track list handling
- Corrupted track data
- Invalid annotation data prevention

### 3. keyboard-shortcuts.spec.ts
Tests keyboard shortcuts:
- Space: Play/pause toggle
- B: Add boundary at playhead
- D: Set downbeat at playhead
- Q: Toggle quantize
- C/R: Return to cue point
- Arrow Left/Right: Navigate through track (with Shift/Ctrl modifiers)
- Arrow Up/Down: Navigate between tracks
- +/-: Zoom in/out
- 0: Reset zoom to fit
- Input field focus handling (shortcuts disabled while typing)
- Rapid command handling
- Keyboard shortcuts help text visibility

## Test Strategy

### Waiting Strategies
Tests use proper Playwright waiting strategies:
- `waitForSelector()` - Wait for elements to appear
- `waitForLoadState('networkidle')` - Wait for network to be idle
- `expect().toBeVisible()` - Verify element visibility with auto-waiting
- Small `waitForTimeout()` only when necessary for UI animations

### Assertions
- Clear, descriptive assertions
- Verify both UI state and data persistence
- Check error messages and user feedback

### Test Isolation
- Each test is independent
- `beforeEach` hook sets up clean state
- Tests don't depend on execution order

## Configuration

See `playwright.config.ts` for configuration:
- Base URL: `http://localhost:5174` (Vite dev server)
- Browser: Chromium (Desktop Chrome)
- Screenshots: On failure only
- Videos: Retained on failure
- Auto-starts dev server for tests

## Debugging

### View test report
```bash
npx playwright show-report
```

### Debug specific test
```bash
npx playwright test --debug annotation-workflow.spec.ts
```

### View trace for failed tests
Traces are automatically collected on first retry. View with:
```bash
npx playwright show-trace trace.zip
```

## Prerequisites for Tests

The backend server must have:
- At least one audio track available
- Proper API endpoints responding:
  - `GET /api/tracks` - List tracks
  - `GET /api/tracks/:filename` - Load track data
  - `POST /api/annotations` - Save annotations
  - `GET /api/audio/:filename` - Serve audio files

## CI/CD Integration

Tests are configured for CI environments:
- Retries: 2 attempts on CI
- Workers: 1 worker on CI (sequential execution)
- Screenshots and videos retained on failure
- HTML report generated

## Known Limitations

- Tests require a running backend with test data
- Audio playback may not work in headless mode (but tests verify UI state)
- WSL/Windows path issues may require manual installation: `npx playwright install chromium`

## Troubleshooting

### Installation issues in WSL
If you encounter path issues during installation:
```bash
# Install from within a pure Linux terminal
npx playwright install chromium --with-deps
```

### Tests timing out
- Ensure backend is running and responsive
- Check network timeouts in playwright.config.ts
- Increase timeout for specific tests if needed:
  ```typescript
  test('slow test', async ({ page }) => {
    test.setTimeout(60000); // 60 seconds
    // ... test code
  });
  ```

### Browser not found
```bash
# Reinstall browsers
npx playwright install --force chromium
```
