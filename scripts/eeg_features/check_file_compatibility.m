function check_file_compatibility()
% CHECK_FILE_COMPATIBILITY Quick check of cleaned EEG file format compatibility
%
% This script performs a lightweight check without running full extraction:
% 1. Loads one cleaned .mat file
% 2. Validates the format matches what feature extraction expects
% 3. Shows the exact loading code that will be used
%
% Run this before doing a full 24-hour extraction to catch format issues early!

    fprintf('=== FILE FORMAT COMPATIBILITY CHECK ===\n\n');
    
    %% Load configs
    config_gen = yaml.loadFile('config/general.yaml');
    cleaned_folder = config_gen.paths.cleaned_eeg;
    filtered_folder = fullfile(config_gen.paths.eeg_data, 'filtered');
    
    %% Find test file
    cleaned_files = dir(fullfile(cleaned_folder, 'P*_cleaned.mat'));
    if isempty(cleaned_files)
        fprintf('✗ No cleaned files found in: %s\n', cleaned_folder);
        fprintf('→ Run the cleaning pipeline first!\n');
        return;
    end
    
    test_file = fullfile(cleaned_folder, cleaned_files(1).name);
    p_num = sscanf(cleaned_files(1).name, 'P%d_cleaned.mat');
    
    fprintf('Testing with file: %s\n', cleaned_files(1).name);
    fprintf('Participant number: %d\n\n', p_num);
    
    %% Test cleaned .mat file loading
    fprintf('--- CLEANED .MAT FILE ---\n');
    fprintf('File: %s\n', test_file);
    
    try
        % This is the exact code the feature extraction script uses
        cleaned_data = load(test_file);
        field_names = fieldnames(cleaned_data);
        
        fprintf('✓ File loaded successfully\n');
        fprintf('  Variables in file: %s\n', strjoin(field_names, ', '));
        
        % Feature extraction uses: cleaned_data.(field_name{1})
        eeg_matrix = cleaned_data.(field_names{1});
        
        fprintf('  Variable name: %s\n', field_names{1});
        fprintf('  Data type: %s\n', class(eeg_matrix));
        fprintf('  Dimensions: %d channels × %d samples\n', size(eeg_matrix, 1), size(eeg_matrix, 2));
        
        % Validate dimensions
        n_chans = size(eeg_matrix, 1);
        if n_chans < 128 || n_chans > 129
            fprintf('  ⚠ WARNING: Expected 128-129 channels, found %d\n', n_chans);
        else
            fprintf('  ✓ Channel count looks good (%d)\n', n_chans);
        end
        
        % Check for NaN/Inf
        n_nan = sum(isnan(eeg_matrix(:)));
        n_inf = sum(isinf(eeg_matrix(:)));
        if n_nan > 0
            fprintf('  ⚠ WARNING: Contains %d NaN values\n', n_nan);
        end
        if n_inf > 0
            fprintf('  ⚠ WARNING: Contains %d Inf values\n', n_inf);
        end
        if n_nan == 0 && n_inf == 0
            fprintf('  ✓ No NaN or Inf values\n');
        end
        
    catch ME
        fprintf('✗ ERROR loading cleaned file:\n');
        fprintf('  %s\n', ME.message);
        return;
    end
    
    %% Test filtered .set file loading (for metadata)
    fprintf('\n--- FILTERED .SET FILE (metadata) ---\n');
    filtered_file = fullfile(filtered_folder, sprintf('P%02d_filtered.set', p_num));
    fprintf('File: %s\n', filtered_file);
    
    if ~isfile(filtered_file)
        fprintf('✗ File not found!\n');
        return;
    end
    
    try
        eeglab nogui;
        EEG = pop_loadset('filename', sprintf('P%02d_filtered.set', p_num), ...
                          'filepath', filtered_folder);
        
        fprintf('✓ File loaded successfully\n');
        fprintf('  Sampling rate: %g Hz\n', EEG.srate);
        fprintf('  Channels: %d\n', EEG.nbchan);
        fprintf('  Samples: %d\n', EEG.pnts);
        fprintf('  Trials: %d\n', EEG.trials);
        
        if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
            fprintf('  ✓ Channel locations present (%d channels)\n', length(EEG.chanlocs));
        else
            fprintf('  ⚠ WARNING: No channel locations\n');
        end
        
    catch ME
        fprintf('✗ ERROR loading filtered .set file:\n');
        fprintf('  %s\n', ME.message);
        return;
    end
    
    %% Test data replacement (what feature extraction does)
    fprintf('\n--- DATA REPLACEMENT TEST ---\n');
    fprintf('This simulates what feature extraction does:\n');
    fprintf('  1. Load filtered .set (for metadata)\n');
    fprintf('  2. Replace EEG.data with cleaned matrix\n');
    fprintf('  3. Remove channel 129 if present\n\n');
    
    try
        % Replace data (feature extraction line 99-100)
        original_size = size(EEG.data);
        EEG.data = eeg_matrix;
        fprintf('  Step 1: Replaced EEG.data\n');
        fprintf('    Original: %s\n', mat2str(original_size));
        fprintf('    New: %s\n', mat2str(size(EEG.data)));
        
        % Remove channel 129 if present (feature extraction lines 102-107)
        if size(EEG.data, 1) >= 129
            EEG.data(129, :, :) = [];
            if length(EEG.chanlocs) >= 129
                EEG.chanlocs(129) = [];
            end
            fprintf('  Step 2: Removed channel 129\n');
        else
            fprintf('  Step 2: No channel 129 to remove\n');
        end
        
        % Update dimensions (feature extraction lines 109-112)
        [EEG.nbchan, EEG.pnts, EEG.trials] = size(EEG.data);
        if ndims(EEG.data) == 2
            EEG.trials = 1;
        end
        fprintf('  Step 3: Updated dimensions\n');
        fprintf('    Channels: %d\n', EEG.nbchan);
        fprintf('    Samples: %d\n', EEG.pnts);
        fprintf('    Trials: %d\n', EEG.trials);
        
        % Validate with eeg_checkset (feature extraction line 113)
        EEG = eeg_checkset(EEG);
        fprintf('  Step 4: Passed eeg_checkset validation\n');
        
        fprintf('\n  ✓ Data replacement successful!\n');
        
    catch ME
        fprintf('\n  ✗ ERROR during data replacement:\n');
        fprintf('    %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('    in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        return;
    end
    
    %% Summary
    fprintf('\n=== COMPATIBILITY CHECK SUMMARY ===\n\n');
    fprintf('✓✓✓ ALL CHECKS PASSED ✓✓✓\n\n');
    fprintf('Your cleaned EEG files are compatible with the refactored feature extraction script!\n');
    fprintf('The exact loading code from the feature extraction script works correctly.\n\n');
    fprintf('Safe to proceed with full extraction:\n');
    fprintf('  extract_eeg_features()  %% All participants\n');
    fprintf('  extract_eeg_features(''participants'', 1:10)  %% Test subset first\n\n');
end
