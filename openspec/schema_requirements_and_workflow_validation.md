# [SCHEMA] Output Schema Requirements and Workflow Validation

## Vision Statement

Evolve from a system where **validation is required to prevent invalid states** to one where **invalid states are impossible to create**.

Eventually: No validation code needed. Type system guarantees correctness.

## Problem Statement

Currently, the CLI can produce incomplete output depending on which analysis types are requested:
- Requesting `--types structure` doesn't include `downbeat` and `time_signature`
- Different analysis paths can lead to missing critical fields in output
- No guarantee that output contains required metadata for downstream processing
- Type system allows construction of logically impossible states (structure without BPM)

This creates situations where:
1. Users get incomplete data without knowing it
2. Downstream code must handle optional fields that should always exist
3. It's possible to have structure analysis without BPM or beat information
4. Invalid states aren't caught until serialization or use

## Goals

1. **Short-term:** Prevent invalid output through runtime validation
2. **Medium-term:** Make prerequisites explicit and auto-resolved in workflows
3. **Long-term:** Eliminate validation entirely through type-safe discriminated unions where invalid states cannot be constructed

## Analysis Dependencies

Before designing output schemas, understand the analysis dependency graph:

```
bpm (no dependencies)
  ↓
beats (requires: bpm)
  ↓
structure (requires: bpm, downbeat, time_signature from beats)
```

This means:
- BPM can run standalone
- Beats can run alone (but outputs BPM too)
- Structure REQUIRES beats to run first

## Solution Roadmap

### Phase 1: Runtime Validation (Immediate - Correctness Safety Net)

**Goal:** Prevent invalid output states through validation at serialization boundaries.

Add to `TrackAnalysis`:
```python
class ValidationError(EDMError):
    """Output schema validation failed."""
    pass

@dataclass
class TrackAnalysis:
    # ... existing fields ...

    def validate(self) -> None:
        """Validate output completeness based on analysis type."""
        if self.structure is not None:
            if self.bpm is None:
                raise ValidationError(
                    "Structure analysis output requires 'bpm' field"
                )
            if self.downbeat is None:
                raise ValidationError(
                    "Structure analysis output requires 'downbeat' field"
                )
            if self.time_signature is None:
                raise ValidationError(
                    "Structure analysis output requires 'time_signature' field"
                )

    def to_dict(self) -> dict:
        """Convert to dict with validation."""
        self.validate()  # Fail early before serialization
        result: dict = {"file": self.file}
        # ... rest of to_dict ...
        return result
```

**Benefits:**
- Catches invalid states before they reach users
- Clear error messages about missing fields
- Minimal code changes
- Non-breaking change

**Tradeoffs:**
- Runtime validation instead of compile-time
- Easy to bypass if code calls fields directly

### Phase 2: Workflow Dependency Declaration (Short-term - Usability)

**Goal:** Make analysis dependencies explicit and automatically resolve them.

Create analysis type registry:
```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class AnalysisDependencies:
    """Declare what an analysis type requires and provides."""
    requires: set[str]  # Prerequisite analysis types
    provides: set[str]  # Fields this analysis provides

ANALYSIS_TYPES = {
    "bpm": AnalysisDependencies(
        requires=set(),
        provides={"bpm"},
    ),
    "beats": AnalysisDependencies(
        requires={"bpm"},
        provides={"bpm", "downbeat", "time_signature"},
    ),
    "structure": AnalysisDependencies(
        requires={"bpm", "downbeat", "time_signature"},
        provides={"bpm", "downbeat", "time_signature", "structure"},
    ),
}

def resolve_analysis_workflow(
    requested: list[str],
) -> list[str]:
    """Resolve analysis types to execute, auto-adding prerequisites.

    Args:
        requested: User-requested analysis types (e.g., ["structure"])

    Returns:
        Complete workflow (e.g., ["bpm", "beats", "structure"])

    Raises:
        AnalysisError: If prerequisite cannot be satisfied.
    """
    provided = set()
    workflow = []

    for analysis_type in requested:
        deps = ANALYSIS_TYPES.get(analysis_type)
        if not deps:
            raise AnalysisError(f"Unknown analysis type: {analysis_type}")

        missing = deps.requires - provided
        if missing:
            # Auto-add missing prerequisites in dependency order
            for prereq in sorted(missing):  # Ensure deterministic order
                if prereq not in workflow:
                    workflow.append(prereq)

        if analysis_type not in workflow:
            workflow.append(analysis_type)

        provided.update(deps.provides)

    return workflow
```

Use in CLI:
```python
def analyze_command(..., analysis_types: list[str] | None, ...):
    # Resolve analysis workflow
    requested_types = analysis_types or ["structure"]
    workflow = resolve_analysis_workflow(requested_types)

    logger.info(f"Resolved workflow: {workflow}")
    # Input: ["structure"] → Output: ["bpm", "beats", "structure"]

    # Conditionally run based on resolved workflow
    run_bpm = "bpm" in workflow
    run_beats = "beats" in workflow
    run_structure = "structure" in workflow

    # ... rest of analysis ...
```

**Benefits:**
- Users can request "structure" and automatically get BPM+beats
- Dependencies are centralized and versioned
- Can add new analysis types without duplicating logic
- Clear error if prerequisite cannot be satisfied
- Documentation is generated from registry

**Update CLI Help:**
```
--types TYPE    Analysis types to perform: bpm, beats, structure
                Specifying 'structure' automatically includes bpm and beats.
                Dependencies:
                  bpm: no prerequisites
                  beats: requires bpm
                  structure: requires bpm, beats
```

### Phase 3: Type-Safe Discriminated Unions (Medium-term - Type Safety)

**Goal:** Make invalid states impossible to construct. Each analysis type becomes its own class.

Replace flat `TrackAnalysis` with analysis-specific classes:

```python
from pydantic import BaseModel, Field
from typing import Literal, Union

class BPMAnalysis(BaseModel):
    """BPM analysis result."""
    file: str
    duration: float
    bpm: float
    analysis_type: Literal["bpm"] = "bpm"

    # No optional fields - you CANNOT construct this without bpm

class BeatsAnalysis(BaseModel):
    """Beats/downbeat analysis result (includes BPM)."""
    file: str
    duration: float
    bpm: float  # REQUIRED
    downbeat: float  # REQUIRED
    time_signature: str  # REQUIRED
    analysis_type: Literal["beats"] = "beats"

    # Type system enforces all fields present

class StructureAnalysis(BaseModel):
    """Structure analysis result (includes beats + structure)."""
    file: str
    duration: float
    bpm: float  # REQUIRED - impossible to have structure without BPM
    downbeat: float  # REQUIRED
    time_signature: str  # REQUIRED
    structure: list[list]  # REQUIRED
    analysis_type: Literal["structure"] = "structure"

# Union of all analysis types
AnalysisResult = BPMAnalysis | BeatsAnalysis | StructureAnalysis

# Type signature proves it's impossible to have incomplete output
def analyze_file(filepath: Path, analysis_types: list[str]) -> AnalysisResult:
    """Analysis result is guaranteed complete based on analysis_type."""
    workflow = resolve_analysis_workflow(analysis_types)

    # Type checker knows:
    # - If returning StructureAnalysis, ALL fields must be present
    # - Missing a field = compilation error, not runtime error

    if "structure" in workflow:
        return StructureAnalysis(
            file=str(filepath),
            duration=duration,
            bpm=bpm,          # MUST provide
            downbeat=downbeat,  # MUST provide
            time_signature=ts,  # MUST provide
            structure=structure  # MUST provide
        )

    if "beats" in workflow:
        return BeatsAnalysis(
            file=str(filepath),
            duration=duration,
            bpm=bpm,
            downbeat=downbeat,
            time_signature=ts,
        )

    return BPMAnalysis(
        file=str(filepath),
        duration=duration,
        bpm=bpm,
    )
```

**Using Type-Safe Results:**

```python
# Type narrowing - mypy understands this
result = analyze_file("track.mp3", ["structure"])

match result:
    case StructureAnalysis():
        # mypy KNOWS result.structure exists - no None check needed
        process_structure(result.structure)
    case BeatsAnalysis():
        # mypy KNOWS result.downbeat exists
        process_downbeats(result.downbeat)
    case BPMAnalysis():
        # minimal data
        log_bpm(result.bpm)

# Or use isinstance for older Python:
if isinstance(result, StructureAnalysis):
    # Type narrowing - result.structure guaranteed present
    save_structure(result.structure)
```

**Benefits:**
- **Zero validation code needed** - type system prevents construction of invalid states
- **Type checking catches bugs** - mypy errors if missing required fields
- **Self-documenting** - types explicitly show what each analysis includes
- **Compile-time guarantees** - invalid states caught during development
- **No Nones in valid code** - defensive None checks unnecessary

**Tradeoffs:**
- Breaking change to output format (adds `analysis_type` field)
- Clients need to handle union types
- Requires Pydantic or similar for clean validation

### Phase 4: Protocol-Based Type Contracts (Long-term - Flexibility)

**Goal:** For maximum flexibility, use structural typing (Protocols) to define contracts.

```python
from typing import Protocol

class HasBPM(Protocol):
    """Anything with BPM information."""
    bpm: float

class HasBeatGrid(HasBPM, Protocol):
    """Anything with beat/downbeat information."""
    bpm: float
    downbeat: float
    time_signature: str

class HasStructure(HasBeatGrid, Protocol):
    """Anything with track structure."""
    bpm: float
    downbeat: float
    time_signature: str
    structure: list[list]

# Functions declare what they need:
def save_structure_yaml(result: HasStructure):
    """Knows result HAS these fields - no None checks."""
    for section in result.structure:  # Never None
        ...

def process_beats(result: HasBeatGrid):
    """Works with any result that has beat info."""
    grid = BeatGrid(
        first_beat_time=result.downbeat,
        bpm=result.bpm,
        time_signature=tuple(map(int, result.time_signature.split("/")))
    )
    ...
```

**Benefits:**
- Protocols compose naturally (inheritance)
- Structural typing: "if it has the fields, it works"
- Works with any type (discriminated union, flat dataclass, dict, etc.)
- Gradual adoption: retrofit to existing code

**Tradeoffs:**
- Only catches type errors, not runtime None values
- Still need discriminated unions for construction safety

## Implementation Checklist

### Phase 1: Runtime Validation (Immediate)
- [ ] Add `ValidationError` exception to `edm/exceptions.py`
- [ ] Add `TrackAnalysis.validate()` method checking structure requirements
- [ ] Call `validate()` in `TrackAnalysis.to_dict()` before serialization
- [ ] Add unit tests verifying validation errors on incomplete structure
- [ ] ✅ Structure analysis includes downbeat/time_signature (DONE)

### Phase 2: Workflow Dependencies (Short-term)
- [ ] Create `AnalysisDependencies` dataclass in new file `edm/analysis/dependencies.py`
- [ ] Create `ANALYSIS_TYPES` registry with bpm/beats/structure dependencies
- [ ] Implement `resolve_analysis_workflow()` function
- [ ] Integrate into `analyze_command()` to auto-resolve prerequisites
- [ ] Update CLI help text with dependency information
- [ ] Add tests for dependency resolution (auto-adding, error cases)

### Phase 3: Discriminated Unions (Medium-term)
- [ ] Create Pydantic-based `BPMAnalysis`, `BeatsAnalysis`, `StructureAnalysis` classes
- [ ] Add `analysis_type: Literal[...]` discriminator field to each
- [ ] Create `AnalysisResult = Union[BPMAnalysis | BeatsAnalysis | StructureAnalysis]`
- [ ] Update `analyze_file()` function signature to return `AnalysisResult`
- [ ] Update serialization functions to handle union types
- [ ] Migrate old `TrackAnalysis` dataclass usage to new classes
- [ ] Update tests to match new return types
- [ ] Add mypy/type checking validation to test suite

### Phase 4: Protocol Types (Long-term)
- [ ] Define `HasBPM`, `HasBeatGrid`, `HasStructure` Protocols
- [ ] Annotate functions with Protocol requirements (e.g., `def foo(result: HasStructure)`)
- [ ] Update mypy configuration to check Protocol compliance
- [ ] Document Protocol-based function contracts

## Principles

### Make Impossible States Unrepresentable

The type system should prevent construction of invalid states:
```python
# Phase 1 (Current): Can create, validation catches it
analysis = TrackAnalysis(file="x.flac", structure=[...], bpm=None)
analysis.validate()  # Raises error

# Phase 3 (Future): Cannot create invalid state at all
analysis = StructureAnalysis(file="x.flac", structure=[...])  # TypeError: missing bpm
```

### Validate at Boundaries

Perform validation at:
1. **Construction** (Pydantic models)
2. **Serialization** (to_dict, output_yaml)
3. **Function entry** (Protocol types)

Never in the middle of computation.

### Explicit Dependencies

Don't scatter "if structure, then analyze BPM" logic. Declare once:
```python
ANALYSIS_TYPES["structure"].requires = {"bpm", "downbeat", "time_signature"}
```

Use this single source of truth everywhere.

## Success Criteria

1. ✅ Structure analysis always includes BPM, downbeat, time_signature
2. ✅ No API can construct output missing core required fields (Phase 3+)
3. ✅ Invalid states cause compilation errors, not runtime crashes (Phase 3+)
4. ✅ Analysis dependencies are explicit and auto-resolved (Phase 2+)
5. ✅ Error messages guide users on missing prerequisites (Phase 1+)
6. ✅ Type narrowing eliminates defensive None checks (Phase 3+)

## Migration Path

**Users won't need to change code** when upgrading because:

- Phase 1: Behavior unchanged, just adds validation
- Phase 2: Auto-adds prerequisites transparently
- Phase 3: Breaking change to `--types` parameter behavior, but output types are extensible
- Phase 4: Only affects function signatures in our codebase, not users

## Reference: Problem Examples

**Before (Current):**
```bash
$ edm analyze track.mp3 --types structure
# Output missing downbeat, time_signature
```

**After Phase 1:**
```bash
$ edm analyze track.mp3 --types structure
ValidationError: Structure analysis output requires 'downbeat' field
```

**After Phase 2:**
```bash
$ edm analyze track.mp3 --types structure
[Auto-added bpm, beats to analyze chain]
Found 1 file(s) to analyze
# Output includes structure, bpm, downbeat, time_signature
```

**After Phase 3:**
```python
result = analyze_file("track.mp3", ["structure"])
# result is StructureAnalysis (not BPMAnalysis or BeatsAnalysis)
# Type checker KNOWS result.structure exists
# Impossible to have result.bpm == None
```
