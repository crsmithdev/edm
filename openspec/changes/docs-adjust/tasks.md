# Tasks: Documentation Alignment

## Configuration Updates

- [ ] **Update pyproject.toml Python version to 3.12+**
  - Change `requires-python = ">=3.10"` to `">=3.12"`
  - Update classifiers to remove 3.10 and 3.11 entries
  - Validate: `openspec validate docs-adjust --strict`
  - Validation: Confirm classifiers only list 3.12+

- [ ] **Update pyproject.toml test configuration**
  - Update target-version in tool.black to `["py312"]`
  - Update target-version in tool.mypy to `"3.12"`
  - Validate all config values are consistent
  - Validation: Ensure no references to 3.10/3.11 remain in config

## Documentation Updates

- [ ] **Update README.md Python requirement**
  - Change "Python 3.12+" section if it says 3.12+ (verify current state)
  - If currently says different version, update to "Python 3.12+"
  - Remove any madmom installation instructions
  - Validation: Run `grep -n "python\|Python\|madmom" README.md`

- [ ] **Update docs/python-style.md line length**
  - Find line mentioning 88 characters
  - Change to 100 characters
  - Update explanation to match pyproject.toml values
  - Validation: `grep -n "88\|100" docs/python-style.md`

- [ ] **Update docs/cli-reference.md log level default**
  - Find --log-level option documentation
  - Change default from INFO to WARNING
  - Update description if needed
  - Validation: `grep -n "log-level\|INFO\|WARNING" docs/cli-reference.md`

- [ ] **Update docs/architecture.md for beat_this**
  - Replace all madmom references with beat_this
  - Remove "Why madmom for BPM?" section or replace with "Why beat_this?"
  - Update BPM Computation section
  - Fix any line number references to functions
  - Validation: `grep -n "madmom" docs/architecture.md` should return 0 results

- [ ] **Update docs/development.md for beat_this**
  - Replace madmom installation instructions with beat_this
  - Update any madmom references
  - Fix any line number references
  - Validation: `grep -n "madmom" docs/development.md` should return 0 results

- [ ] **Update docs/agent-guide.md line references**
  - Verify all function line references are accurate
  - Fix any references off by >1 line
  - Check: analyze_bpm, compute_bpm, and other functions mentioned
  - Validation: Spot-check 3 function references for accuracy

## Code Documentation Updates

- [ ] **Add clarity to structure.py about placeholder nature**
  - Update docstring to note this is placeholder implementation
  - Add comment explaining sections are hardcoded
  - Validation: Read updated docstring and confirm clarity

- [ ] **Add clarity to beatport.py about unimplemented status**
  - Update docstring and comments to note this is not implemented
  - Validation: Confirm function docstring mentions "not implemented"

- [ ] **Add clarity to tunebat.py about unimplemented status**
  - Update docstring and comments to note this is not implemented
  - Validation: Confirm function docstring mentions "not implemented"

- [ ] **Update config.py docstring about TOML loading**
  - Clarify in docstring that TOML configuration is not yet loaded
  - Explain legacy madmom naming vs actual beat_this usage
  - Validation: Read config docstring and confirm clarity

- [ ] **Update architecture.md to note unimplemented features**
  - Add explicit section noting which features are placeholder/unimplemented
  - List: structure analysis, Beatport, TuneBat, TOML config
  - Validation: Confirm section exists and lists all unimplemented features

## Validation & Testing

- [ ] **Run openspec validation**
  - Command: `openspec validate docs-adjust --strict`
  - Validation: All validation checks pass with no errors

- [ ] **Verify documentation consistency**
  - Spot-check: README matches code requirements
  - Spot-check: Style guide matches pyproject.toml
  - Spot-check: CLI reference matches main.py defaults
  - Spot-check: All beat_this references are present where madmom was removed
  - Validation: All spot checks pass

- [ ] **Run full test suite**
  - Command: `uv run pytest`
  - Validation: All tests pass (no functionality changes, so tests should remain green)

## Submission

- [ ] **Create commit and push**
  - Commit all documentation and configuration changes
  - Push to branch and create PR
  - Validation: GitHub Actions passes, documentation builds without errors
