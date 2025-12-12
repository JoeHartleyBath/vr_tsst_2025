"""
Private helper modules for physiological feature extraction.

This package contains modular components for:
- Data loading and caching
- Signal cleaning (HR, GSR, eye tracking)
- Feature extraction
- Merging with EEG and subjective data
- Quality control logging

Each module is self-contained and testable.
"""

# Data loading
from .load_data import (
    load_config,
    load_raw_physio_data,
    load_eeg_features,
    load_subjective_ratings,
    validate_loaded_data,
    fix_participant_ids
)

# Module imports will be added as helper functions are implemented
# from .clean_hr_data import clean_hr_pipeline
# from .clean_gsr_data import clean_gsr_pipeline, resample_gsr_to_10hz
# from .clean_eye_data import clean_eye_pipeline
# from .extract_features import extract_all_features
# from .merge_with_eeg import merge_physio_with_eeg
