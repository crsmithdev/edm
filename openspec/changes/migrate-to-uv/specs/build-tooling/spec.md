# Build Tooling Specification

## ADDED Requirements

### Requirement: Use uv as Package Installer

The project SHALL use uv as the primary package installer and resolver instead of pip for all dependency management operations.

#### Scenario: Fresh installation
- **WHEN** a developer clones the repository and runs the setup command
- **THEN** uv SHALL download and install all dependencies from `pyproject.toml`
- **AND** the installation SHALL complete in under 30 seconds for a warm cache
- **AND** all transitive dependencies SHALL be resolved correctly

#### Scenario: Installing development dependencies
- **WHEN** a developer needs development tools (testing, linting)
- **THEN** uv SHALL install the `[dev]` optional dependencies
- **AND** all dev tools SHALL be available in the environment

#### Scenario: Adding new dependency
- **WHEN** a developer adds a new package to `pyproject.toml`
- **THEN** uv SHALL resolve and install the package
- **AND** uv SHALL update `uv.lock` with the resolved version
- **AND** the lock file SHALL include all transitive dependencies with exact versions

### Requirement: Automatic Virtual Environment Management

The project SHALL let uv automatically manage virtual environments instead of requiring manual creation and activation.

#### Scenario: First-time setup
- **WHEN** a developer runs `uv sync` for the first time
- **THEN** uv SHALL automatically create a `.venv` directory if it doesn't exist
- **AND** uv SHALL install all dependencies into the virtual environment
- **AND** the developer SHALL NOT need to manually run `python -m venv` or activate the venv

#### Scenario: Running commands in managed environment
- **WHEN** a developer needs to run Python commands or scripts
- **THEN** they SHALL use `uv run <command>` to execute in the managed environment
- **AND** uv SHALL automatically activate the correct environment
- **AND** the command SHALL have access to all installed dependencies

#### Scenario: Using system Python
- **WHEN** uv cannot find a suitable Python version
- **THEN** uv SHALL use the system Python if it meets requirements
- **OR** provide clear instructions to install a compatible Python version

### Requirement: Reproducible Builds with Lockfile

The project SHALL maintain a `uv.lock` file to ensure reproducible builds across different machines and environments.

#### Scenario: Lock file generation
- **WHEN** dependencies are installed or updated
- **THEN** uv SHALL generate or update `uv.lock` with exact resolved versions
- **AND** the lock file SHALL include checksums for integrity verification
- **AND** the lock file SHALL be committed to version control

#### Scenario: Installing from lock file
- **WHEN** a developer runs `uv sync` on a clean checkout
- **THEN** uv SHALL install exact versions from `uv.lock`
- **AND** the installation SHALL be identical across all machines
- **AND** no dependency resolution SHALL be performed (unless lock is outdated)

#### Scenario: Lock file out of date
- **WHEN** `pyproject.toml` changes but `uv.lock` is stale
- **THEN** uv SHALL detect the mismatch
- **AND** prompt the developer to run `uv lock` to update
- **OR** automatically update if `--locked` flag is not used

### Requirement: Python Version Management

The project SHALL specify the required Python version in a `.python-version` file for consistency across environments.

#### Scenario: Python version specification
- **WHEN** the project is checked out
- **THEN** a `.python-version` file SHALL exist in the project root
- **AND** the file SHALL specify the minimum or exact Python version required
- **AND** tools like pyenv SHALL respect this version automatically

#### Scenario: Version compatibility check
- **WHEN** uv installs dependencies
- **THEN** uv SHALL verify the Python version meets requirements from `pyproject.toml`
- **AND** uv SHALL fail with a clear error if the version is incompatible
- **AND** the error message SHALL indicate the required version

### Requirement: Git Dependency Support

The project SHALL support installing dependencies directly from git repositories (specifically madmom) using uv.

#### Scenario: Installing madmom from git
- **WHEN** the project dependencies are installed
- **THEN** uv SHALL install madmom from `git+https://github.com/CPJKU/madmom.git`
- **AND** the installation SHALL build the package from source
- **AND** the build SHALL succeed with Cython and NumPy available

#### Scenario: Lock file with git dependencies
- **WHEN** `uv.lock` is generated
- **THEN** the lock file SHALL include the git commit hash for madmom
- **AND** subsequent installations SHALL use the locked commit
- **AND** the build SHALL be reproducible

### Requirement: Documentation and Migration

The project documentation SHALL provide clear instructions for using uv and migrating from pip-based workflows.

#### Scenario: New developer onboarding
- **WHEN** a new developer reads the README
- **THEN** they SHALL find clear instructions to install uv
- **AND** they SHALL find step-by-step setup instructions using uv
- **AND** the instructions SHALL be simpler than the previous pip workflow

#### Scenario: Existing developer migration
- **WHEN** an existing developer switches to the uv workflow
- **THEN** migration documentation SHALL explain how to transition
- **AND** instructions SHALL include cleaning up old virtualenvs
- **AND** instructions SHALL explain the benefits of the new workflow

#### Scenario: Troubleshooting
- **WHEN** a developer encounters issues with uv
- **THEN** documentation SHALL provide common solutions
- **AND** documentation SHALL explain how to fall back to pip if needed
- **AND** documentation SHALL link to uv's official documentation

### Requirement: Backward Compatibility

The project SHALL maintain the existing `pyproject.toml` structure and remain installable with pip for users who prefer or require it.

#### Scenario: pip installation still works
- **WHEN** a user installs the project with `pip install -e .`
- **THEN** the installation SHALL succeed
- **AND** all dependencies SHALL install correctly
- **AND** the project SHALL function identically to uv installation

#### Scenario: pyproject.toml compatibility
- **WHEN** the project uses uv-specific features
- **THEN** the features SHALL be optional or compatible with pip
- **AND** `pyproject.toml` SHALL remain standards-compliant (PEP 621)
- **AND** no pip users SHALL be forced to switch to uv
