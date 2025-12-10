# E2E Testing Setup and Usage

This document provides setup instructions and usage guidelines for the Playwright E2E tests.

## Quick Start

### 1. Install Dependencies

```bash
cd packages/edm-annotator/frontend
npm install
```

### 2. Install Playwright Browsers

```bash
npx playwright install chromium
```

If you're on Linux/WSL and encounter issues, install with dependencies:

```bash
npx playwright install chromium --with-deps
```

### 3. Start the Backend

The E2E tests require a running backend server with test data. Ensure you have:
- Backend server running (usually on port 8000)
- At least one audio track available in the configured directory
- All API endpoints functional

### 4. Run Tests

```bash
# Run all E2E tests (headless)
npm run test:e2e

# Run with interactive UI mode (recommended for development)
npm run test:e2e:ui

# Run in debug mode
npm run test:e2e:debug

# Run specific test file
npx playwright test annotation-workflow.spec.ts
```

## Test Structure

```
tests/e2e/
├── README.md                      # Detailed test documentation
├── test-helpers.ts               # Reusable test utilities
├── annotation-workflow.spec.ts   # Critical workflow tests
├── error-handling.spec.ts        # Error scenario tests
└── keyboard-shortcuts.spec.ts    # Keyboard shortcut tests
```

## Test Coverage

### 1. Annotation Workflow (CRITICAL)
- Complete end-to-end annotation workflow
- Load track → Add boundaries → Save → Verify persistence
- BPM setting and tap tempo
- Quantize functionality
- Validation (prevent saving without boundaries)

### 2. Error Handling
- Network failures and timeouts
- API errors (track loading, saving)
- Invalid data handling
- Missing audio files
- Input validation

### 3. Keyboard Shortcuts
- All major shortcuts (Space, B, D, Q, arrows, +/-, 0)
- Modifier key combinations (Shift, Ctrl)
- Input field focus handling
- Rapid command execution

## Development Workflow

### Writing New Tests

1. Create a new `.spec.ts` file in `tests/e2e/`
2. Import test helpers for common operations:
   ```typescript
   import { test, expect } from '@playwright/test';
   import { loadFirstTrack, addBoundaryAtPosition, saveAnnotation } from './test-helpers';
   ```

3. Use proper waiting strategies:
   ```typescript
   // Good - wait for specific condition
   await page.waitForSelector('text=/Loaded/', { timeout: 10000 });

   // Better - use auto-waiting assertions
   await expect(element).toBeVisible();

   // Avoid - arbitrary timeouts
   await page.waitForTimeout(5000); // Only use when necessary
   ```

### Running Tests During Development

Use UI mode for the best development experience:

```bash
npm run test:e2e:ui
```

This provides:
- Interactive test selection
- Time travel debugging
- Watch mode
- Visual trace timeline

### Debugging Failed Tests

1. **View HTML report** (generated after test run):
   ```bash
   npx playwright show-report
   ```

2. **Run in headed mode** (see browser):
   ```bash
   npx playwright test --headed
   ```

3. **Run single test with debug**:
   ```bash
   npx playwright test --debug annotation-workflow.spec.ts -g "complete workflow"
   ```

4. **View trace** (on first retry):
   ```bash
   npx playwright show-trace trace.zip
   ```

## Best Practices

### Selectors
- Prefer accessible selectors: `getByRole()`, `getByText()`, `getByLabel()`
- Use data-testid attributes for complex components
- Avoid CSS selectors based on styling (use semantic selectors)

### Waiting
- Use explicit waits: `waitForSelector()`, `waitForLoadState()`
- Leverage auto-waiting with expect assertions
- Avoid hardcoded timeouts when possible

### Test Independence
- Each test should be independent
- Use `beforeEach` for setup
- Don't rely on test execution order
- Clean up state after tests

### Assertions
- Use descriptive assertion messages
- Verify both UI state and data
- Check error messages and feedback

## CI/CD Integration

The tests are pre-configured for CI:

```yaml
# Example GitHub Actions workflow
- name: Install dependencies
  run: npm install

- name: Install Playwright browsers
  run: npx playwright install chromium --with-deps

- name: Start backend
  run: npm run dev &

- name: Run E2E tests
  run: npm run test:e2e

- name: Upload test results
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: playwright-report/
```

## Configuration

Edit `playwright.config.ts` to customize:
- Test timeout
- Retries
- Browsers
- Base URL
- Screenshot/video settings

## Common Issues

### WSL Path Issues
If you see UNC path errors in WSL:
```bash
# Run installation from a pure Linux terminal
npx playwright install chromium
```

### Backend Not Running
Tests will timeout if backend is not accessible:
```
Error: page.goto: Timeout 30000ms exceeded.
```
Solution: Ensure backend is running on http://localhost:8000

### Browser Installation Failed
```bash
# Reinstall with system dependencies
npx playwright install --force chromium --with-deps
```

### Tests Pass Locally But Fail in CI
- Check for timing issues (add more explicit waits)
- Verify CI has sufficient resources
- Ensure test data is available in CI environment

## Test Helpers Reference

The `test-helpers.ts` module provides utilities:

- `loadFirstTrack()` - Load first track from list
- `addBoundaryAtPosition()` - Add boundary at specific position
- `addMultipleBoundaries()` - Add multiple evenly-spaced boundaries
- `getBoundaryCount()` - Get current boundary count
- `getRegionCount()` - Get current region count
- `setBPM()` - Set BPM value
- `saveAnnotation()` - Save and verify success
- `toggleQuantize()` - Toggle quantize on/off
- `isQuantizeEnabled()` - Check quantize state
- `setCuePoint()` - Set cue point by clicking
- `verifyTrackHasReference()` - Verify track has saved annotation

## Resources

- [Playwright Documentation](https://playwright.dev/)
- [Best Practices](https://playwright.dev/docs/best-practices)
- [Debugging Guide](https://playwright.dev/docs/debug)
- [CI/CD Guide](https://playwright.dev/docs/ci)
