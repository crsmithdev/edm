# core-library Specification Delta

## REMOVED Requirements

### Requirement: Feature Extraction Module
The library SHALL provide a features module (`edm.features`) for extracting audio features used in analysis.

**Reason**: Feature extraction functions were placeholder stubs that returned zeros and were never integrated into the analysis pipeline. Current BPM analysis uses beat_this and librosa directly without a feature extraction abstraction layer. Removing reduces architectural complexity and eliminates misleading API surface.

**Migration**: No migration needed - functions were never functional. Future implementations can add feature extraction when concrete use cases emerge.

#### Scenario: Extract spectral features
- **WHEN** user calls `extract_spectral_features(audio_data, sample_rate)`
- **THEN** returns features including spectral centroid, rolloff, and flux

#### Scenario: Extract temporal features
- **WHEN** user calls `extract_temporal_features(audio_data, sample_rate)`
- **THEN** returns features including RMS energy, zero-crossing rate, and onset strength

### Requirement: Model Management Module
The library SHALL provide a models module (`edm.models`) for loading and managing ML models.

**Reason**: Model management was an abstract BaseModel class and load_model function that always raised ModelNotFoundError. No ML models currently use this abstraction - beat_this handles its own model loading internally. Removing eliminates unused architectural layer and clarifies that model management is not a current system capability.

**Migration**: No migration needed - functions always raised exceptions and were never usable. ModelNotFoundError exception preserved in exceptions module for future use. Future ML model integration can add model management when needed.

#### Scenario: Load pre-trained model
- **WHEN** user calls `load_model(model_name)` with a valid model identifier
- **THEN** loads the model and returns a model instance ready for inference

#### Scenario: Model not found
- **WHEN** user calls `load_model(model_name)` with an invalid identifier
- **THEN** raises a custom `ModelNotFoundError` exception
