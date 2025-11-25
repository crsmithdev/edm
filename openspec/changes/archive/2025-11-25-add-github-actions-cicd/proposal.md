# Change: Add GitHub Actions CI/CD workflows

## Why

The project currently lacks automated testing, linting, and deployment pipelines. Adding GitHub Actions CI/CD will ensure code quality, catch bugs early, automate releases, and provide confidence in changes through automated validation. This is essential for maintaining code quality as the project grows and accepts contributions.

## What Changes

- Create GitHub Actions workflow for continuous integration (testing, linting, type checking)
- Add workflow for test coverage reporting and tracking
- Create workflow for automated releases and version tagging
- Add workflow for dependency security scanning
- Implement matrix testing across Python versions (3.9, 3.10, 3.11, 3.12)
- Add status badges to README for build status and coverage
- Configure workflows to use uv for fast dependency installation (if migrate-to-uv is implemented)

## Impact

- **Affected specs**: cicd (new capability)
- **Affected code**:
  - `.github/workflows/` (new directory)
  - `README.md` (add status badges)
  - `pyproject.toml` (may need workflow config)
- **Developer experience**: Automated feedback on PRs, confidence in changes, faster reviews
- **Quality assurance**: Every commit tested, consistent code style enforced, security vulnerabilities detected
- **No breaking changes**: Pure additions, no existing functionality changed
