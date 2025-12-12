# EEG Feature Extraction - Refactored

## Overview

Professional refactoring of EEG feature extraction pipeline with modular architecture, flexible CLI, and robust error handling.

## Key Improvements

### 1. **Modular Architecture**
- Main orchestration script: `extract_eeg_features_refactored.m` (113 lines)
- Helper functions in `private/` folder (automatically available to main script)
- Clean separation of concerns: parsing, validation, processing, merging

### 2. **Flexible CLI Interface**
```matlab
% Default: all 48 participants
extract_eeg_features()

% Specific participants
extract_eeg_features('participants', [1 5 10])

% Force reprocess (ignore resume)
extract_eeg_features('force_reprocess', true)

% Custom output folder
extract_eeg_features('output_folder', 'custom/path')

% Custom config file
extract_eeg_features('config_file', 'my_config.yaml')

% Disable parallel processing
extract_eeg_features('parallel', false)

% Override worker count
extract_eeg_features('num_workers', 4)

% Multiple parameters
extract_eeg_features('participants', 1:10, 'parallel', false, 'force_reprocess', true)
```

### 3. **Robust Error Handling**
- Input validation with clear error messages
- Config validation checks all required fields
- Try-catch blocks with detailed error reporting
- Graceful handling of missing files

### 4. **Better Progress Reporting**
- Clear phase indicators (SETUP, PROCESSING, MERGING, etc.)
- Per-participant status: ✓ success, ⚠ warning, ✗ error
- Timing information (start time, duration, completion time)
- Detailed logging to timestamped file

### 5. **Resume Capability**
- Automatic detection of processed participants
- Skip already-processed work
- Force reprocess option to override

## File Structure

```
scripts/eeg_features/
├── extract_eeg_features_refactored.m    # Main orchestration (113 lines)
├── extract_eeg_features.m               # Legacy script (561 lines)
├── extract_eeg_features_legacy.m        # Original script (1003 lines)
├── check_file_compatibility.m           # Quick format validation
├── test_pipeline_compatibility.m        # Full pipeline test
├── README.md                            # This file
├── REFACTORING_COMPARISON.md            # Before/after comparison
├── private/
│   ├── parse_inputs.m                   # CLI argument parsing
│   ├── validate_config.m                # Config validation
│   ├── setup_environment.m              # Environment setup
│   ├── build_output_schema.m            # CSV header generation
│   ├── determine_work.m                 # Resume logic
│   ├── setup_parallel_pool.m            # Parallel processing setup
│   ├── process_all_participants.m       # Main processing loop
│   ├── merge_temp_files.m               # Result merging
│   ├── cleanup_temp_files.m             # Temp file cleanup
│   ├── normalize_condition_label.m      # Event label mapping
│   ├── compute_features.m               # Feature extraction orchestration
│   ├── calc_psd.m                       # Power spectral density
│   ├── compute_band_power.m             # Band power calculation
│   ├── compute_ratios.m                 # Power ratios
│   ├── compute_sample_entropy.m         # Sample entropy
│   └── compute_spectral_entropy.m       # Spectral entropy
└── legacy/
    └── ... (older versions)
```

## Module Responsibilities

### Input & Validation
- **parse_inputs.m**: Parse CLI arguments with validation
- **validate_config.m**: Validate configuration files
- **setup_environment.m**: Load configs, setup paths, create folders, initialize logging

### Output Schema
- **build_output_schema.m**: Generate CSV column headers based on enabled features

### Work Planning
- **determine_work.m**: Check existing temp files and determine what needs processing

### Parallel Processing
- **setup_parallel_pool.m**: Initialize parallel pool and sync paths

### Main Processing
- **process_all_participants.m**: Main loop (parallel or serial)
  - Handles participant iteration
  - Calls process_single_participant() for each
  - Writes results to temp files

### Feature Computation
- **normalize_condition_label.m**: Map raw event labels to canonical names
- **compute_features.m**: Orchestrate feature extraction
- **calc_psd.m**: Compute power spectral density (Welch's method)
- **compute_band_power.m**: Integrate power in frequency bands
- **compute_ratios.m**: Compute power ratios (asymmetry, etc.)
- **compute_sample_entropy.m**: Sample entropy calculation
- **compute_spectral_entropy.m**: Spectral entropy calculation

### Result Handling
- **merge_temp_files.m**: Merge participant temp files into final CSV
- **cleanup_temp_files.m**: Delete temp files and folder

## Configuration

All parameters externalized to YAML files:

### config/eeg_feature_extraction.yaml
- Frequency bands (Delta, Theta, Alpha, Beta, etc.)
- Regions (FrontalLeft, FrontalRight, etc.)
- Feature flags (band_power, ratios, entropy)
- Toolbox paths
- Parallel settings

### config/conditions.yaml
- Condition definitions
- Aliases for event mapping
- Duration for each condition
- Include/exclude flags

### config/general.yaml
- Input/output paths
- EEG data location
- Cleaned data location
- Events file location

## Usage Examples

### Standard Processing
```matlab
% Process all participants with default settings
extract_eeg_features()
```

### Development/Testing
```matlab
% Test on 3 participants, single-threaded for easier debugging
extract_eeg_features('participants', [1 2 3], 'parallel', false)
```

### Partial Reprocessing
```matlab
% Reprocess specific participants (e.g., after fixing their data)
extract_eeg_features('participants', [5 10 15], 'force_reprocess', true)
```

### Custom Configuration
```matlab
% Use custom config file
extract_eeg_features('config_file', 'config/eeg_extraction_custom.yaml')
```

### Resume After Crash
```matlab
% If processing crashed, just rerun - it will skip completed participants
extract_eeg_features()
% Output: "Found 35 already processed: [1 2 3 ... 35]"
%         "Processing 13 remaining participants..."
```

## Benefits of Refactoring

### Code Quality
- **From 1003 → 113 lines** in main script (89% reduction)
- **16 focused modules** instead of monolithic script
- Each function has single, clear responsibility
- Easy to understand, test, and maintain

### Usability
- **Flexible CLI** - no need to edit code for different runs
- **Self-documenting** - clear help text and examples
- **Robust** - validates inputs and configs before processing

### Reliability
- **Resume capability** - won't lose 24 hours of work if crash occurs
- **Better error messages** - know exactly what went wrong and where
- **Validated configs** - catch issues before starting long runs

### Maintainability
- **Modular** - easy to modify one component without breaking others
- **Testable** - each function can be tested independently
- **Documented** - clear comments and help text

## Migration Guide

### For Users
The new script is backward compatible. If you were running:
```matlab
% Old way (edit participant_numbers in script)
run extract_eeg_features.m
```

Now you can run:
```matlab
% New way (no editing required)
extract_eeg_features('participants', 1:48)
```

### For Developers
When modifying feature extraction:

1. **Adding new feature**: Edit `compute_features.m` and add corresponding module
2. **Changing band definitions**: Edit `config/eeg_feature_extraction.yaml`
3. **Modifying regions**: Edit `config/eeg_feature_extraction.yaml`
4. **Adding validation**: Edit `validate_config.m`
5. **Debugging processing**: Use single-threaded mode with small participant set

## Performance

- **Identical output** to legacy script (validated)
- **Same runtime** (~24 hours for 48 participants with entropy)
- **Resume capability** reduces risk of time loss
- **Parallel safety** - temp file approach prevents race conditions

## Future Enhancements

Potential improvements for future versions:

1. **Progress estimation**: Show estimated time remaining during processing
2. **Checkpointing**: Save intermediate state for even more robust resume
3. **Unit tests**: Test individual modules with synthetic data
4. **Performance profiling**: Identify bottlenecks for optimization
5. **GPU support**: Accelerate entropy calculations if available
6. **Real-time monitoring**: Web dashboard to track progress

## Validation & Testing

Before running the full 24-hour extraction, validate that your cleaned EEG files are compatible:

### Quick Format Check (30 seconds)
```matlab
% Quick check without running full extraction
check_file_compatibility()
```

This validates:
- Cleaned .mat file format
- Filtered .set file availability  
- Data replacement logic
- Uses exact loading code from feature extraction script

### Full Pipeline Test (5-10 minutes)
```matlab
% Comprehensive test with actual feature extraction
test_result = test_pipeline_compatibility()
```

This runs:
1. File existence checks
2. Format validation
3. Full feature extraction on one participant
4. Output validation

**Run these tests after completing the cleaning pipeline to ensure compatibility!**

## Troubleshooting

### "Config file not found"
- Check that you're running from project root or provide full path
- Verify config files exist in `config/` folder

### "No participants to process" (all already done)
- Delete temp files: `rmdir('output/eeg_features/temp', 's')`
- Or use: `extract_eeg_features('force_reprocess', true)`

### "Toolbox path not found"
- Check `config/eeg_feature_extraction.yaml` toolbox paths
- Verify EEGLAB and EntropyHub are installed

### Parallel processing issues
- Try single-threaded: `extract_eeg_features('parallel', false)`
- Check MATLAB Parallel Computing Toolbox is installed
- Reduce workers: `extract_eeg_features('num_workers', 4)`

## Credits

Refactored: December 2025
Original script: VR-TSST study EEG pipeline
Dependencies: EEGLAB, EntropyHub, YAML parser
