# [DATAMGMT] Implementation Tasks

## Phase 1: DVC Integration (Week 1)

- [x] 1.1 Install and configure DVC
  - [x] 1.1.1 Add `dvc>=3.0.0` to pyproject.toml
  - [x] 1.1.2 Run `dvc init` in project root
  - [x] 1.1.3 Configure local remote storage (or S3/GCS)
  - [x] 1.1.4 Add DVC entries to .gitignore

- [x] 1.2 Track data directories
  - [x] 1.2.1 `dvc add data/annotations/`
  - [x] 1.2.2 `dvc add data/generated/` (if keeping)
  - [x] 1.2.3 Commit .dvc files to git

- [x] 1.3 Define new YAML schema
  - [x] 1.3.1 Create `src/edm/data/schema.py` with Pydantic models
  - [x] 1.3.2 Define sections: metadata, audio, structure, energy (optional)
  - [x] 1.3.3 Metadata fields: tier, confidence, source, created, modified, verified_by, notes, flags, validation
  - [x] 1.3.4 Audio fields: file, duration (float seconds), bpm, downbeat, time_signature (array), key
  - [x] 1.3.5 Structure: list of {bar, label, time, confidence}
  - [x] 1.3.6 Energy: overall, by_section (bass/mid/high), at_boundaries
  - [x] 1.3.7 Validator explicitly rejects old format (no annotations: [[bar, label, time]] arrays)

- [x] 1.4 Create schema validator
  - [x] 1.4.1 Implement `validate_annotation(yaml_path)` function
  - [x] 1.4.2 Check all required fields present
  - [x] 1.4.3 Validate confidence scores in [0, 1]
  - [x] 1.4.4 Validate tier in {1, 2, 3}

## Phase 2: Data Management CLI (Week 2)

- [x] 2.1 Create base CLI command
  - [x] 2.1.1 Create `src/cli/commands/data.py`
  - [x] 2.1.2 Add `edm data` command group with Typer
  - [x] 2.1.3 Wire into main CLI

- [x] 2.2 Implement `edm data stats`
  - [x] 2.2.1 Count tracks by tier
  - [x] 2.2.2 Calculate average confidence
  - [x] 2.2.3 Count tracks needing review (flagged)
  - [x] 2.2.4 Display rich table with statistics

- [x] 2.3 Implement `edm data validate`
  - [x] 2.3.1 Load all YAML files in data/annotations/
  - [x] 2.3.2 Validate each against schema (strict, no old format support)
  - [x] 2.3.3 Check heuristics: beat snapping, section length, BPM consistency
  - [x] 2.3.4 Report validation errors with file:line references

- [x] 2.4 Implement `edm data export`
  - [x] 2.4.1 Export to JSON format
  - [x] 2.4.2 Export to PyTorch dataset format
  - [x] 2.4.3 Export confidence weights for training

- [x] 2.5 Implement `edm data tier`
  - [x] 2.5.1 Update tier field in YAML
  - [x] 2.5.2 Support batch updates: `--set 1 track1.yaml track2.yaml`

- [x] 2.6 Implement `edm data flag`
  - [x] 2.6.1 Add/remove flags: needs_review, high_confidence, etc.

- [ ] 2.7 Clean break: delete old annotations
  - [ ] 2.7.1 Git commit current annotations: "archive: old format before DATAMGMT"
  - [ ] 2.7.2 Delete all `data/annotations/*.yaml`
  - [ ] 2.7.3 Ready for fresh import with new format

## Phase 3: Label Studio Integration (Week 3-4, Optional)

- [ ] 3.1 Add Label Studio dependency
  - [ ] 3.1.1 Add to pyproject.toml `[project.optional-dependencies]`
  - [ ] 3.1.2 Create `annotation-ui` extra: `pip install edm[annotation-ui]`

- [ ] 3.2 Create Label Studio adapter
  - [ ] 3.2.1 Create `src/edm/data/label_studio.py`
  - [ ] 3.2.2 Implement `yaml_to_label_studio()` conversion
  - [ ] 3.2.3 Implement `label_studio_to_yaml()` conversion

- [ ] 3.3 Create import/export scripts
  - [ ] 3.3.1 `scripts/import_to_label_studio.py`
  - [ ] 3.3.2 `scripts/export_from_label_studio.py`

- [ ] 3.4 Configure Label Studio project
  - [ ] 3.4.1 Create audio segmentation template
  - [ ] 3.4.2 Add EDM-specific labels: intro, buildup, main, breakdown, outro
  - [ ] 3.4.3 Configure export format to match YAML schema

- [ ] 3.5 Document Label Studio workflow
  - [ ] 3.5.1 Add guide to docs/label-studio-setup.md
  - [ ] 3.5.2 Include screenshots of UI
  - [ ] 3.5.3 Describe import/review/export workflow

## Phase 4: Integration with MLPIVOT

- [x] 4.1 Update Rekordbox importer
  - [x] 4.1.1 Generate metadata section when importing
  - [x] 4.1.2 Set source=rekordbox, tier=2, confidence from validation

- [ ] 4.2 Update training dataset class
  - [ ] 4.2.1 Read metadata from YAML
  - [ ] 4.2.2 Use confidence for sample weighting
  - [ ] 4.2.3 Filter by tier (e.g., only Tier 1 for validation)

- [ ] 4.3 Track model outputs with DVC
  - [ ] 4.3.1 Add dvc pipeline for training
  - [ ] 4.3.2 Track model checkpoints: `dvc add data/models/`
  - [ ] 4.3.3 Link data version to model version

## Phase 5: Documentation & Testing

- [ ] 5.1 Write data management spec
  - [ ] 5.1.1 Create `openspec/specs/data-management/spec.md`
  - [ ] 5.1.2 Document YAML schema with examples
  - [ ] 5.1.3 Document DVC workflow
  - [ ] 5.1.4 Document tier system

- [ ] 5.2 Update development workflow spec
  - [ ] 5.2.1 Add DVC commands to workflow
  - [ ] 5.2.2 Document dataset versioning best practices

- [ ] 5.3 Write tests
  - [ ] 5.3.1 Test schema validation
  - [ ] 5.3.2 Test YAML upgrade migration
  - [ ] 5.3.3 Test export functions

- [ ] 5.4 Update README
  - [ ] 5.4.1 Add DVC setup instructions
  - [ ] 5.4.2 Add data management section

## Phase 6: Verification

- [ ] 6.1 Run full test suite
- [ ] 6.2 Validate all existing annotations with new validator
- [ ] 6.3 Test DVC push/pull workflow
- [ ] 6.4 Verify backward compatibility with old YAML
