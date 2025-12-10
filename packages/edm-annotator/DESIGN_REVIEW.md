# EDM Annotator Design Review & Modernization Plan

## Current Design Analysis

### Strengths
1. **Dark theme foundation**: Solid dark color palette (#0f1419 background, #1E2139 cards)
2. **Clear information hierarchy**: Waveform is prominent, controls are logically grouped
3. **Functional layout**: Two-column grid separates controls from track list effectively
4. **Interactive feedback**: Hover states, transitions, and visual feedback on interactions
5. **Domain-appropriate**: Design suits audio/music production workflows

### Areas for Modernization

#### 1. **Inline Styles → CSS Modules/Tailwind**
**Current**: All styles are inline objects, making the codebase harder to maintain
**Impact**:
- Code duplication (button styles repeated across components)
- No design tokens/variables for consistency
- Difficult to theme or adjust globally

**Recommendation**: Extract to CSS modules or adopt Tailwind CSS for consistency

#### 2. **Typography & Spacing**
**Current**: Good use of Inter font, but spacing could be more systematic
**Observations**:
- Mixed font sizes (11px, 12px, 13px, 14px, 16px, 18px, 28px)
- Inconsistent spacing values (4px, 10px, 12px, 14px, 15px, 16px, 20px, 24px, 30px)

**Recommendation**: Adopt 4px/8px grid system for spacing, establish type scale

#### 3. **Color System**
**Current Colors**:
- Background: `#0F1419`
- Cards: `#1E2139`, `#151828`
- Borders: `#2A2F4C`, `rgba(91, 124, 255, 0.1)`
- Primary: `#5B7CFF` (blue)
- Accent: `#00E6B8` (cyan/teal)
- Warning/Stop: `#FF6B6B` (red)
- Secondary: `#FFB800` (yellow/amber)
- Text: `#FFFFFF`, `#E5E7EB`, `#9CA3AF`, `#6B7280`

**Issues**:
- No semantic color naming
- Hard-coded colors everywhere
- Limited color palette for region labels

**Recommendation**: Create CSS custom properties for design tokens

#### 4. **Component Patterns**
**Observations**:
- Buttons have inconsistent patterns (some use emoji, some use text)
- No reusable Button component with variants
- Info displays have good visual pattern (label + value) but could be component
- Region list items have complex inline hover logic

**Recommendation**: Create design system components (Button, InfoCard, etc.)

#### 5. **Accessibility Issues**
**Problems**:
- No visible focus indicators beyond browser defaults
- Color contrast issues on some grays (#6B7280 on dark background)
- Emoji usage (⏸, ▶, ↺) without aria-labels
- No keyboard navigation indicators
- Select dropdowns rely only on color for state

**Recommendation**: Add focus-visible styles, improve contrast, add ARIA labels

#### 6. **Visual Hierarchy & Polish**
**Opportunities**:
- Add subtle shadows/elevation system (currently using box-shadow inconsistently)
- Improve button iconography (replace text emoji with proper icons)
- Add loading states/skeleton screens
- Improve empty states
- Add micro-interactions (e.g., save button success animation)

#### 7. **Responsive Design**
**Current**: Fixed max-width: 1600px, no mobile considerations
**Issue**: Grid layout (`3fr 2fr`) will break on small screens

**Recommendation**: Add responsive breakpoints, stack layout on mobile

#### 8. **Modern UI Patterns**
**Missing**:
- Keyboard shortcut hints in UI
- Tooltips for controls
- Command palette for power users
- Undo/redo indicators
- Drag-to-reorder regions
- Batch operations on regions

## Modernization Implementation Plan

### Phase 1: Design System Foundation
1. Create CSS custom properties for colors, spacing, typography
2. Establish reusable component library (Button, Card, Badge, etc.)
3. Implement consistent focus-visible styles
4. Add proper icon library (Lucide React or similar)

### Phase 2: Component Refactor
1. Refactor PlaybackControls to use design system
2. Refactor EditingControls with proper button variants
3. Improve RegionList with better interaction patterns
4. Enhance TrackSelector with search/filter

### Phase 3: Enhanced UX
1. Add keyboard shortcut hints overlay
2. Implement tooltips
3. Add loading skeletons
4. Improve empty states
5. Add success/error animations

### Phase 4: Responsive & Accessibility
1. Make layout responsive
2. Improve color contrast to WCAG AA
3. Add comprehensive ARIA labels
4. Test keyboard navigation

### Phase 5: Advanced Features
1. Add region drag-to-reorder
2. Implement batch label operations
3. Add command palette
4. Add undo/redo stack
5. Export/import annotations UI

## Recommended Design Tokens

```css
:root {
  /* Colors - Background */
  --bg-primary: #0F1419;
  --bg-secondary: #1E2139;
  --bg-tertiary: #151828;
  --bg-elevated: #252A45;

  /* Colors - Borders */
  --border-subtle: #2A2F4C;
  --border-focus: #5B7CFF;

  /* Colors - Semantic */
  --color-primary: #5B7CFF;
  --color-accent: #00E6B8;
  --color-danger: #FF6B6B;
  --color-warning: #FFB800;

  /* Colors - Text */
  --text-primary: #FFFFFF;
  --text-secondary: #E5E7EB;
  --text-tertiary: #9CA3AF;
  --text-muted: #6B7280;

  /* Spacing (4px grid) */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 20px;
  --space-6: 24px;
  --space-8: 32px;

  /* Typography */
  --font-size-xs: 11px;
  --font-size-sm: 12px;
  --font-size-base: 14px;
  --font-size-lg: 16px;
  --font-size-xl: 18px;
  --font-size-2xl: 24px;
  --font-size-3xl: 28px;

  /* Radius */
  --radius-sm: 6px;
  --radius-md: 8px;
  --radius-lg: 10px;
  --radius-xl: 14px;

  /* Shadows */
  --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.2);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
  --shadow-lg: 0 8px 16px rgba(0, 0, 0, 0.4);
}
```

## Visual Improvements Summary

### Quick Wins (Low effort, high impact)
1. Replace emoji with icon library
2. Add CSS custom properties
3. Improve button hover states
4. Add focus-visible styles
5. Add tooltips to controls

### Medium Effort
1. Create Button component with variants
2. Extract InfoCard component
3. Improve RegionList interaction
4. Add loading states
5. Make layout responsive

### Long Term
1. Design system documentation
2. Advanced keyboard shortcuts UI
3. Command palette
4. Drag-and-drop regions
5. Comprehensive accessibility audit
