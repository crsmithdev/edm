# Testing Roadmap

## Current Status (Phase 5 In Progress - Dec 20, 2025)

**Completed:** 260 passing tests across 8 waveform components (24 failures being fixed)
**Total Tests:** ~589+ tests
**Component Coverage:** 33% (8/24 components)
**Critical Path Coverage:** 100% (stores, hooks, utils, services)

---

## Remaining Work

### 🔴 HIGH PRIORITY - Complete Phase 5 (Waveform Components)

**5 components remaining, ~600-850 lines:**

1. **DetailWaveform.test.tsx** (200-250 lines)
   - Centered playhead rendering
   - Waveform scrolling
   - Boundary interaction
   - Region overlay rendering
   - Beat grid rendering

2. **OverviewWaveform.test.tsx** (150-200 lines)
   - Full track rendering
   - Playhead movement
   - Click-to-seek
   - Zoom synchronization

3. **RegionOverlays.test.tsx** (100-150 lines)
   - Region color mapping
   - Current region highlighting
   - Opacity management
   - Viewport clipping

4. **BeatGrid.test.tsx** (100-150 lines)
   - Grid calculation from BPM
   - Downbeat alignment
   - Viewport filtering
   - Bar number labels

5. **Playhead.test.tsx** (50-100 lines)
   - Position tracking
   - Viewport conversion
   - Animation smoothness

---

### 🟡 MEDIUM PRIORITY - Phase 6 (Components & Utilities)

**Transport/Editing Components:**

6. **NavigationControls.test.tsx** (200-250 lines)
   - Jump calculations (beats vs bars)
   - Mode toggle
   - BPM-based jump distances
   - Boundary clamping

7. **EditingControls.test.tsx** (150-200 lines)
   - Add boundary at current time
   - Set downbeat
   - Toggle quantize
   - Button disabled states
   - Status messages

**Test Utilities:**

8. **fixtures.ts** (200 lines)
   ```typescript
   export function createMockAnnotation(options: {
     boundaryCount?: number
     duration?: number
     bpm?: number
     labels?: SectionLabel[]
     tier?: number
   })

   export function createMockWaveform(duration: number, sampleRate = 10)
   ```

---

### 🟡 MEDIUM-HIGH PRIORITY - Phase 7 (Integration Tests)

9. **quantization-workflows.test.tsx** (200-250 lines)
   - Enable quantize → add boundaries → verify snapping
   - Change BPM with quantized boundaries
   - Drag boundary with quantize enabled
   - Re-snap on quantize toggle

10. **error-recovery.test.tsx** (250-300 lines)
    - Network failure → retry logic
    - Corrupted annotation → fallback
    - Save failure → preserve unsaved changes
    - State recovery from errors

11. **performance.test.tsx** (150-200 lines)
    - Large track loading (>10 min)
    - Many boundaries (>100)
    - Rapid zoom/pan
    - Waveform rendering efficiency

---

### 🟢 MEDIUM PRIORITY - Phase 8 (Backend API)

**Backend Error Handling Expansion:**

12. **Expand test_annotations_api.py** (200-250 lines)
    - Load missing generated annotation (404)
    - Invalid YAML handling
    - Invalid boundary data
    - File permission errors
    - Concurrent writes
    - Invalid BPM validation
    - Negative/out-of-order boundaries

13. **Expand test_tracks_api.py** (150-200 lines)
    - Empty directory handling
    - Permission errors
    - Non-audio files filtering
    - Annotation status accuracy
    - Subdirectory handling
    - Mixed format support
    - Sorting validation

14. **test_audio_api.py** (new file, ~150 lines)
    - Audio file serving
    - MIME type handling
    - Range requests
    - File not found errors

15. **test_waveforms_api.py** (new file, ~150 lines)
    - Waveform generation
    - Caching behavior
    - Multi-channel processing
    - Corrupted file handling

---

### 🟢 LOW PRIORITY - Phase 9 (UI Components)

**9 UI components, ~400-500 lines:**

Consider **visual regression testing** (Percy/Chromatic) instead of unit tests.

16-24. UI Component Tests:
    - Button.test.tsx (variant, size, icon, disabled)
    - Card.test.tsx (padding, elevation)
    - Tooltip.test.tsx (hover, shortcuts, positioning)
    - ConfirmDialog.test.tsx (modal, confirm/cancel, variants)
    - EmptyState.test.tsx (message, icon)
    - InfoCard.test.tsx (label/value display)
    - KeyboardHints.test.tsx (shortcut list)
    - Skeleton.test.tsx (loading states)
    - StatusToast.test.tsx (auto-dismiss, queueing)

---

### 🟡 MEDIUM PRIORITY - Phase 10 (E2E Tests)

**E2E Infrastructure & Flows:**

25. Setup Playwright (infrastructure)
26. First-time user flow (~100-150 lines)
27. Power user flow (~100-150 lines)
28. Error scenarios (~100-150 lines)
29. Cross-browser tests (~50-100 lines)

---

## Summary Statistics

| Phase | Status | Items | Lines Remaining |
|-------|--------|-------|-----------------|
| **Phase 5** | 3/8 complete | 5 components | ~600-850 |
| **Phase 6** | Not started | 3 items | ~550-650 |
| **Phase 7** | Not started | 3 suites | ~600-750 |
| **Phase 8** | Not started | 4 files | ~650-800 |
| **Phase 9** | Not started | 9 components | ~400-500 |
| **Phase 10** | Not started | 5 items | ~350-550 |
| **TOTAL** | | **29 items** | **~3,150-4,100** |

---

## Priority Order

### Immediate (Complete Phase 5)
1. DetailWaveform.test.tsx
2. OverviewWaveform.test.tsx
3. RegionOverlays.test.tsx
4. BeatGrid.test.tsx
5. Playhead.test.tsx

### Next (Phase 6)
6. NavigationControls.test.tsx
7. EditingControls.test.tsx
8. fixtures.ts utility

### Then (Phases 7-8, can parallelize)
9-11. Integration tests
12-15. Backend API expansion

### Finally (Phases 9-10, optional)
16-24. UI components
25-29. E2E tests

---

## Testing Patterns Established (Phase 5)

### SVG-Based Rendering
```typescript
it("creates valid SVG paths", () => {
  const { container } = render(<WaveformCanvas />);
  const paths = container.querySelectorAll("path");
  expect(paths[0].getAttribute("d")).toMatch(/^M/);
  expect(paths[0].getAttribute("d")).toMatch(/Z$/);
});
```

### Viewport Filtering
```typescript
it("only renders boundaries within viewport", () => {
  useWaveformStore.setState({ viewportStart: 40, viewportEnd: 120 });
  const { container } = render(<BoundaryMarkers />);
  const markers = container.querySelectorAll('[style*="position: absolute"]');
  expect(markers).toHaveLength(2); // Only 2 within [40, 120]
});
```

### Interactive Elements
```typescript
it("removes boundary on Ctrl+Click", () => {
  const removeBoundarySpy = vi.spyOn(useStructureStore.getState(), "removeBoundary");
  fireEvent.click(markers[1], { ctrlKey: true });
  expect(removeBoundarySpy).toHaveBeenCalledWith(50);
});
```

### Canvas Mocking
```typescript
import { mockCanvasElement } from "@/tests/utils/mockCanvas";

const { canvas, context } = mockCanvasElement();
expect(context.fillRect).toHaveBeenCalled();
```

---

## Test Infrastructure

### Available Utilities
- ✅ `tests/utils/mockCanvas.ts` - Canvas rendering mock
- 🔄 `tests/utils/fixtures.ts` - TODO: Annotation/waveform builders

### Test Organization
```
tests/
├── components/
│   ├── Waveform/     (3/8 tested)
│   ├── Transport/    (1/2 tested)
│   ├── Editing/      (2/3 tested)
│   ├── TrackList/    (1/1 tested)
│   └── UI/           (0/9 tested)
├── hooks/            (3/3 tested ✅)
├── stores/           (6/6 tested ✅)
├── utils/            (5/5 tested ✅)
├── services/         (1/1 tested ✅)
└── integration/      (2 suites)
```

---

## Coverage Goals

### Current
- Stores: 100% ✅
- Hooks: 100% ✅
- Utils: 100% ✅
- Services: 100% ✅
- Components: 29% ⚠️
- Integration: Partial ⚠️

### Target (After All Phases)
- Components: 95% (23/24)
- Integration: 90%
- E2E: Core flows covered

---

*Last updated: December 19, 2025 - Phase 5 Complete*
