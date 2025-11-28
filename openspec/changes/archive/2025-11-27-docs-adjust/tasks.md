# Tasks: Documentation Alignment

## Configuration Updates

- [x] **Update pyproject.toml Python version to 3.12+**
  - Already set to `requires-python = ">=3.12"`
  - Classifiers already list 3.12+ only
  - Validation: Confirmed

- [x] **Update pyproject.toml test configuration**
  - Already set: target-version = `["py312"]` in tool.black
  - Already set: target-version = `"3.12"` in tool.mypy
  - Validation: Confirmed

## Documentation Updates

- [x] **Update README.md Python requirement**
  - Already says Python 3.12+
  - No madmom references
  - Validation: Confirmed

- [x] **Update docs/python-style.md line length**
  - Already says 100 characters
  - Validation: Confirmed

- [x] **Update docs/cli-reference.md log level default**
  - Updated to show WARNING as default
  - Added comment about use_madmom being legacy name
  - Validation: Confirmed

- [x] **Update docs/architecture.md for beat_this**
  - External services section removed (previous change)
  - Beat_this references already present
  - Validation: Only one madmom reference remains (comparison context)

- [x] **Update docs/development.md for beat_this**
  - Removed Spotify references
  - Updated Python target from 3.9+ to 3.12+
  - Updated reference source line numbers
  - Removed mock_spotify_client fixture reference
  - Validation: Confirmed

- [x] **Update docs/agent-guide.md line references**
  - Removed external services section
  - Updated BPM strategy description
  - Updated line references
  - Validation: Confirmed

## Code Documentation Updates

- [x] **Add clarity to structure.py about placeholder nature**
  - Already has clear placeholder documentation
  - Validation: Docstring mentions "placeholder implementation"

- [x] **Add clarity to beatport.py about unimplemented status**
  - File deleted in remove-external-api-integrations change
  - N/A

- [x] **Add clarity to tunebat.py about unimplemented status**
  - File deleted in remove-external-api-integrations change
  - N/A

- [x] **Update config.py docstring about TOML loading**
  - Already has clear documentation about TOML not being implemented
  - use_madmom comment explains legacy naming
  - Validation: Confirmed

- [x] **Update architecture.md to note unimplemented features**
  - Section already exists listing placeholder features
  - Updated to remove external services (now deleted)
  - Validation: Confirmed

## Validation & Testing

- [x] **Run openspec validation**
  - Proposal validates successfully
  - Validation: Passed

- [x] **Verify documentation consistency**
  - README matches code requirements
  - Style guide matches pyproject.toml
  - CLI reference matches main.py defaults
  - Validation: All checks pass

- [x] **Run full test suite**
  - 106 passed, 5 skipped
  - Validation: All tests pass

## Submission

- [ ] **Create commit and push**
  - Ready for archival
