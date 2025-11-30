## 1. Make msaf Required
- [ ] 1.1 Move msaf import to top-level in structure_detector.py
- [ ] 1.2 Remove is_available() method from MSAFDetector
- [ ] 1.3 Remove try/except ImportError guards

## 2. Remove MSAF Runtime Fallback
- [ ] 2.1 Remove MSAF → EnergyDetector fallback in structure.py:130-141
- [ ] 2.2 Let exceptions propagate from MSAFDetector.detect()

## 3. Simplify get_detector()
- [ ] 3.1 Remove "auto" mode fallback logic
- [ ] 3.2 Remove availability checks, return detector directly

## 4. Evaluate EnergyDetector
- [ ] 4.1 Check if EnergyDetector is still used anywhere
- [ ] 4.2 If dead code, remove EnergyDetector class
- [ ] 4.3 If kept for --structure-detector energy flag, document as explicit choice only

## 5. Update Dependencies
- [ ] 5.1 Ensure msaf is in main dependencies (not optional)
- [ ] 5.2 Update README with clear dependency requirements

## 6. Tests
- [ ] 6.1 Update structure detection tests (remove fallback expectations)
- [ ] 6.2 Verify MSAF errors propagate correctly
