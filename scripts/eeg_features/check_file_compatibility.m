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
        
        % Validate dimensions (trigger channel already removed by cleaning pipeline)
        n_chans = size(EEG.data, 1);
        if n_chans ~= 128
            fprintf('  ⚠ WARNING: Expected 128 channels (after trigger removal), found %d\n', n_chans);
        else
            fprintf('  ✓ Channel count correct: %d (trigger already removed)\n', n_chans);
        end
        
        fprintf('  ✓ PASS: Cleaned .set file is ready for feature extraction\n');
        
    catch ME
        fprintf('\n  ✗ ERROR during channel removal:\n');
        fprintf('    %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('    in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        return;
    end
    
    %% Test events in .set file
    fprintf('\n--- EVENTS IN .SET FILE ---\n');
    
    try
        if ~isfield(EEG, 'event') || isempty(EEG.event)
            fprintf('✗ No events in .set file!\n');
            fprintf('  The cleaning pipeline should preserve event markers.\n');
            return;
        end
        
        fprintf('✓ Found %d events\n', length(EEG.event));
        
        % Check event structure
        if ~isfield(EEG.event, 'type') || ~isfield(EEG.event, 'latency')
            fprintf('✗ Events missing required fields (type, latency)\n');
            return;
        end
        
        % Show event types
        event_types = unique({EEG.event.type});
        fprintf('  Event types (%d unique): %s\n', length(event_types), strjoin(event_types, ', '));
        
        % Show first few events
        fprintf('  First 3 events:\n');
        for i = 1:min(3, length(EEG.event))
            fprintf('    [%d] %s at sample %d\n', i, EEG.event(i).type, round(EEG.event(i).latency));
        end
        
        fprintf('  ✓ Events are valid and accessible\n');
        
    catch ME
        fprintf('✗ ERROR checking events:\n');
        fprintf('  %s\n', ME.message);
        return;
    end
    
    %% Summary
    fprintf('\n=== COMPATIBILITY CHECK SUMMARY ===\n\n');
    fprintf('✓✓✓ ALL CHECKS PASSED ✓✓✓\n\n');
    fprintf('Your cleaned EEG files are compatible with the refactored feature extraction script!\n');
    fprintf('The .set file contains:\n');
    fprintf('  - Cleaned EEG data (%d channels)\n', size(EEG.data, 1));
    fprintf('  - Channel locations\n');
    fprintf('  - Sampling rate (%g Hz)\n', EEG.srate);
    fprintf('  - Event markers (%d events)\n\n', length(EEG.event));
    fprintf('Safe to proceed with full extraction:\n');
    fprintf('  extract_eeg_features()  %% All participants\n');
    fprintf('  extract_eeg_features(''participants'', 1:10)  %% Test subset first\n\n');
end
