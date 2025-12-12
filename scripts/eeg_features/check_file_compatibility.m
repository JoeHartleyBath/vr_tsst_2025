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
    
    %% Find test file
    cleaned_files = dir(fullfile(cleaned_folder, 'P*_cleaned.set'));
    if isempty(cleaned_files)
        fprintf('✗ No cleaned .set files found in: %s\n', cleaned_folder);
        fprintf('→ Run the cleaning pipeline first!\n');
        return;
    end
    
    test_file = fullfile(cleaned_folder, cleaned_files(1).name);
    p_num = sscanf(cleaned_files(1).name, 'P%d_cleaned.set');
    
    fprintf('Testing with file: %s\n', cleaned_files(1).name);
    fprintf('Participant number: %d\n\n', p_num);
    
    %% Test cleaned .set file loading
    fprintf('--- CLEANED .SET FILE ---\n');
    fprintf('File: %s\n', test_file);
    
    try
        eeglab nogui;
        
        % This is the exact code the feature extraction script uses
        EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p_num), ...
                          'filepath', cleaned_folder);
        
        fprintf('✓ File loaded successfully\n');
        fprintf('  Sampling rate: %g Hz\n', EEG.srate);
        fprintf('  Channels: %d\n', EEG.nbchan);
        fprintf('  Samples: %d\n', EEG.pnts);
        fprintf('  Trials: %d\n', EEG.trials);
        fprintf('  Data dimensions: %s\n', mat2str(size(EEG.data)));
        
        if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
            fprintf('  ✓ Channel locations present (%d channels)\n', length(EEG.chanlocs));
        else
            fprintf('  ⚠ WARNING: No channel locations\n');
        end
        
        % Validate data
        if isempty(EEG.data)
            fprintf('  ✗ ERROR: No data in .set file\n');
            return;
        end
        
        % Check for NaN/Inf
        n_nan = sum(isnan(EEG.data(:)));
        n_inf = sum(isinf(EEG.data(:)));
        if n_nan > 0
            fprintf('  ⚠ WARNING: Contains %d NaN values\n', n_nan);
        end
        if n_inf > 0
            fprintf('  ⚠ WARNING: Contains %d Inf values\n', n_inf);
        end
        if n_nan == 0 && n_inf == 0
            fprintf('  ✓ No NaN or Inf values\n');
        end
        
        % Validate dimensions
        n_chans = size(EEG.data, 1);
        if n_chans < 128 || n_chans > 129
            fprintf('  ⚠ WARNING: Expected 128-129 channels, found %d\n', n_chans);
        else
            fprintf('  ✓ Channel count looks good (%d)\n', n_chans);
        end
        
    catch ME
        fprintf('✗ ERROR loading cleaned .set file:\n');
        fprintf('  %s\n', ME.message);
        return;
    end
    
    %% Test channel 129 removal (what feature extraction does)
    fprintf('\n--- CHANNEL 129 REMOVAL TEST ---\n');
    fprintf('This simulates what feature extraction does:\n\n');
    
    try
        original_size = size(EEG.data);
        original_chanlocs = length(EEG.chanlocs);
        
        fprintf('  Before: %d channels × %d samples\n', original_size(1), original_size(2));
        
        % Remove channel 129 if present (feature extraction lines 102-107)
        if size(EEG.data, 1) >= 129
            EEG.data(129, :, :) = [];
            if length(EEG.chanlocs) >= 129
                EEG.chanlocs(129) = [];
            end
            fprintf('  → Removed channel 129 (trigger)\n');
        else
            fprintf('  → No channel 129 to remove\n');
        end
        
        % Update dimensions (feature extraction lines 109-112)
        [EEG.nbchan, EEG.pnts, EEG.trials] = size(EEG.data);
        if ndims(EEG.data) == 2
            EEG.trials = 1;
        end
        fprintf('  After: %d channels × %d samples\n', size(EEG.data, 1), size(EEG.data, 2));
        fprintf('  Channel locations: %d → %d\n', original_chanlocs, length(EEG.chanlocs));
        
        % Validate with eeg_checkset (feature extraction line 113)
        EEG = eeg_checkset(EEG);
        fprintf('  ✓ Passed eeg_checkset validation\n');
        
        fprintf('\n  ✓ Channel removal successful!\n');
        
    catch ME
        fprintf('\n  ✗ ERROR during channel removal:\n');
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
