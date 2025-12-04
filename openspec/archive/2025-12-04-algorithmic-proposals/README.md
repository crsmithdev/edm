# Algorithmic Proposals Archive

**Archived:** 2025-12-04
**Reason:** Superseded by ML-first approach (MLPIVOT)

## Context

These proposals explored algorithmic improvements to energy detection and structure segmentation. They have been archived due to the pivot to learning-based approaches outlined in MLPIVOT.

## Archived Proposals

- **BEATSYNC** - Beat-synchronized energy analysis
- **ECLUSTER** - Energy envelope clustering
- **ENERGY** - Multi-feature energy detection
- **MULTIENG** - Multi-feature energy
- **REFINE** - Energy boundary refinement
- **TEMPORAL** - Temporal energy context
- **SEGACC** - Segmentation accuracy

## Why Archived

With the MLPIVOT pivot, ML models will learn these patterns instead of hand-coding them:
- Energy patterns across frequency bands → learned by multi-band energy head
- Boundary refinement → learned by boundary-tolerant training
- Temporal context → learned by transformer encoder
- Beat synchronization → integrated into feature extraction

## Reference

The ideas in these proposals informed the design of MLPIVOT:
- Multi-band energy analysis → 3-band energy regression head
- Beat-synchronized features → training data includes beat grids
- Temporal context → shared encoder with neighborhood attention

These proposals remain valuable as documentation of the algorithmic exploration phase.
