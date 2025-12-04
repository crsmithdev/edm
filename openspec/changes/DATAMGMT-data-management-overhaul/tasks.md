# [DATAMGMT] Implementation Tasks

## Phase 1: DVC Integration (Week 1)

- [ ] 1.1 Install and configure DVC
  - [ ] 1.1.1 Add `dvc>=3.0.0` to pyproject.toml
  - [ ] 1.1.2 Run `dvc init` in project root
  - [ ] 1.1.3 Configure local remote storage (or S3/GCS)
  - [ ] 1.1.4 Add DVC entries to .gitignore

- [ ] 1.2 Track data directories
  - [ ] 1.2.1 `dvc add data/annotations/`
  - [ ] 1.2.2 `dvc add data/generated/` (if keeping)
  - [ ] 1.2.3 Commit .dvc files to git

- [ ] 1.3 Define new YAML schema
  - [ ] 1.3.1 Create `src/edm/data/schema.py` with Pydantic models
  - [ ] 1.3.2 Define sections: metadata, audio, structure, energy (optional)
  - [ ] 1.3.3 Metadata fields: tier, confidence, source, created, modified, verified_by, notes, flags, validation
  - [ ] 1.3.4 Audio fields: file, duration (float seconds), bpm, downbeat, time_signature (array), key
  - [ ] 1.3.5 Structure: list of {bar, label, time, confidence}
  - [ ] 1.3.6 Energy: overall, by_section (bass/mid/high), at_boundaries
  - [ ] 1.3.7 Validator explicitly rejects old format (no annotations: [[bar, label, time]] arrays)

- [ ] 1.4 Create schema validator
  - [ ] 1.4.1 Implement `validate_annotation(yaml_path)` function
  - [ ] 1.4.2 Check all required fields present
  - [ ] 1.4.3 Validate confidence scores in [0, 1]
  - [ ] 1.4.4 Validate tier in {1, 2, 3}

## Phase 2: Data Management CLI (Week 2)

- [ ] 2.1 Create base CLI command
  - [ ] 2.1.1 Create `src/cli/commands/data.py`
  - [ ] 2.1.2 Add `edm data` command group with Typer
  - [ ] 2.1.3 Wire into main CLI

- [ ] 2.2 Implement `edm data stats`
  - [ ] 2.2.1 Count tracks by tier
  - [ ] 2.2.2 Calculate average confidence
  - [ ] 2.2.3 Count tracks needing review (flagged)
  - [ ] 2.2.4 Display rich table with statistics

- [ ] 2.3 Implement `edm data validate`
  - [ ] 2.3.1 Load all YAML files in data/annotations/
  - [ ] 2.3.2 Validate each against schema (strict, no old format support)
  - [ ] 2.3.3 Check heuristics: beat snapping, section length, BPM consistency
  - [ ] 2.3.4 Report validation errors with file:line references

- [ ] 2.4 Implement `edm data export`
  - [ ] 2.4.1 Export to JSON format
  - [ ] 2.4.2 Export to PyTorch dataset format
  - [ ] 2.4.3 Export confidence weights for training

- [ ] 2.5 Implement `edm data tier`
  - [ ] 2.5.1 Update tier field in YAML
  - [ ] 2.5.2 Support batch updates: `--set 1 track1.yaml track2.yaml`

- [ ] 2.6 Implement `edm data flag`
  - [ ] 2.6.1 Add/remove flags: needs_review, high_confidence, etc.

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

- [ ] 4.1 Update Rekordbox importer
  - [ ] 4.1.1 Generate metadata section when importing
  - [ ] 4.1.2 Set source=rekordbox, tier=2, confidence from validation

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
