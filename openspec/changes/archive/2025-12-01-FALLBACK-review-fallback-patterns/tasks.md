## 1. Make msaf Required
- [x] 1.1 Move msaf import to top-level in structure_detector.py
- [x] 1.2 Remove is_available() method from MSAFDetector
- [x] 1.3 Remove try/except ImportError guards

## 2. Remove MSAF Runtime Fallback
- [x] 2.1 Remove MSAF → EnergyDetector fallback in structure.py:130-141
- [x] 2.2 Let exceptions propagate from MSAFDetector.detect()

## 3. Simplify get_detector()
- [x] 3.1 Remove "auto" mode fallback logic
- [x] 3.2 Remove availability checks, return detector directly

## 4. Evaluate EnergyDetector
- [x] 4.1 Check if EnergyDetector is still used anywhere
- [x] 4.2 If dead code, remove EnergyDetector class
- [x] 4.3 If kept for --structure-detector energy flag, document as explicit choice only

## 5. Update Dependencies
- [x] 5.1 Ensure msaf is in main dependencies (not optional)
- [x] 5.2 Update README with clear dependency requirements

## 6. Tests
- [x] 6.1 Update structure detection tests (remove fallback expectations)
- [x] 6.2 Verify MSAF errors propagate correctly
