function test_result = test_pipeline_compatibility()
% TEST_PIPELINE_COMPATIBILITY Verify cleaning → feature extraction compatibility
%
% This test validates that:
% 1. Cleaned EEG files have the expected format
% 2. Feature extraction can load and process cleaned files
% 3. The refactored script produces valid output
%
% Usage:
%   test_result = test_pipeline_compatibility()
%
% Returns:
%   test_result - struct with test outcomes

    fprintf('=== PIPELINE COMPATIBILITY TEST ===\n');
    fprintf('Testing: Cleaning output → Feature extraction input\n\n');
    
    test_result = struct();
    test_result.all_passed = false;
    test_result.tests = {};
    
    %% Test 1: Check if cleaned files exist
    fprintf('[Test 1] Checking for cleaned EEG files...\n');
    config_gen = yaml.loadFile('config/general.yaml');
    cleaned_folder = config_gen.paths.cleaned_eeg;
    
    if ~isfolder(cleaned_folder)
        fprintf('  ✗ FAIL: Cleaned EEG folder not found: %s\n', cleaned_folder);
        test_result.tests{end+1} = struct('name', 'Cleaned folder exists', 'passed', false);
        return;
    end
    
    cleaned_files = dir(fullfile(cleaned_folder, 'P*_cleaned.set'));
    if isempty(cleaned_files)
        fprintf('  ✗ FAIL: No cleaned .set files found in %s\n', cleaned_folder);
        fprintf('  → Run the cleaning pipeline first!\n');
        test_result.tests{end+1} = struct('name', 'Cleaned files exist', 'passed', false);
        return;
    end
    
    fprintf('  ✓ PASS: Found %d cleaned .set files\n', length(cleaned_files));
    test_result.tests{end+1} = struct('name', 'Cleaned files exist', 'passed', true);
    
    %% Test 2: Validate cleaned file format
    fprintf('\n[Test 2] Validating cleaned .set file format...\n');
    test_file = fullfile(cleaned_folder, cleaned_files(1).name);
    fprintf('  Testing: %s\n', cleaned_files(1).name);
    
    try
        eeglab nogui;
        p_num = sscanf(cleaned_files(1).name, 'P%d_cleaned.set');
        
        EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p_num), ...
                          'filepath', cleaned_folder);
        
        % Extract dimensions
        [n_chans, n_samples] = size(EEG.data);
        fprintf('  ✓ Dimensions: %d channels × %d samples\n', n_chans, n_samples);
        fprintf('  ✓ Sampling rate: %g Hz\n', EEG.srate);
        
        % Check for expected number of channels (128 or 129 with trigger)
        if n_chans < 128 || n_chans > 129
            fprintf('  ✗ FAIL: Expected 128-129 channels, found %d\n', n_chans);
            test_result.tests{end+1} = struct('name', 'Valid channel count', 'passed', false);
            return;
        end
        
        % Check for data
        if isempty(EEG.data)
            fprintf('  ✗ FAIL: No data in .set file\n');
            test_result.tests{end+1} = struct('name', 'Data present', 'passed', false);
            return;
        end
        
        fprintf('  ✓ PASS: Cleaned .set file format is valid\n');
        test_result.tests{end+1} = struct('name', 'Valid file format', 'passed', true);
        
    catch ME
        fprintf('  ✗ FAIL: Error loading file: %s\n', ME.message);
        test_result.tests{end+1} = struct('name', 'Load cleaned file', 'passed', false);
        return;
    end
    
    %% Test 3: Check events files
    fprintf('\n[Test 3] Checking for events files...\n');
    events_folder = config_gen.paths.events;
    
    if ~isfolder(events_folder)
        fprintf('  ✗ FAIL: Events folder not found: %s\n', events_folder);
        test_result.tests{end+1} = struct('name', 'Events folder exists', 'passed', false);
        return;
    end
    
    event_files = dir(fullfile(events_folder, 'P*_events.csv'));
    if isempty(event_files)
        fprintf('  ✗ FAIL: No event files found\n');
        test_result.tests{end+1} = struct('name', 'Event files exist', 'passed', false);
        return;
    end
    
    fprintf('  ✓ PASS: Found %d event files\n', length(event_files));
    test_result.tests{end+1} = struct('name', 'Event files exist', 'passed', true);
    
    %% Test 4: Test feature extraction on one participant
    fprintf('\n[Test 4] Testing feature extraction on one participant...\n');
    
    % Find a participant with all required files
    test_participant = [];
    for i = 1:length(cleaned_files)
        % Extract participant number from filename (e.g., P01_cleaned.set → 1)
        fname = cleaned_files(i).name;
        p_num = sscanf(fname, 'P%d_cleaned.set');
        
        if isempty(p_num), continue; end
        
        % Check all required files exist
        cleaned_exists = isfile(fullfile(cleaned_folder, sprintf('P%02d_cleaned.set', p_num)));
        events_exists = isfile(fullfile(events_folder, sprintf('P%02d_events.csv', p_num)));
        
        if cleaned_exists && events_exists
            test_participant = p_num;
            break;
        end
    end
    
    if isempty(test_participant)
        fprintf('  ✗ FAIL: Could not find participant with all required files\n');
        test_result.tests{end+1} = struct('name', 'Complete participant data', 'passed', false);
        return;
    end
    
    fprintf('  Testing with P%02d...\n', test_participant);
    
    try
        % Create test output folder
        test_output = fullfile('output', 'compatibility_test');
        if ~exist(test_output, 'dir')
            mkdir(test_output);
        end
        
        % Run feature extraction on one participant
        fprintf('  Running: extract_eeg_features(''participants'', %d, ''parallel'', false, ...\n', test_participant);
        fprintf('                                ''output_folder'', ''%s'')\n', test_output);
        
        extract_eeg_features('participants', test_participant, ...
                           'parallel', false, ...
                           'output_folder', test_output);
        
        % Check output
        output_csv = fullfile(test_output, 'eeg_features_aggregated.csv');
        if ~isfile(output_csv)
            fprintf('  ✗ FAIL: Output CSV not created\n');
            test_result.tests{end+1} = struct('name', 'Feature extraction runs', 'passed', false);
            return;
        end
        
        % Read and validate output
        output_data = readtable(output_csv);
        n_rows = height(output_data);
        n_cols = width(output_data);
        
        fprintf('  ✓ Output created: %s\n', output_csv);
        fprintf('  ✓ Output dimensions: %d rows × %d columns\n', n_rows, n_cols);
        
        if n_rows < 1
            fprintf('  ✗ FAIL: No data rows in output\n');
            test_result.tests{end+1} = struct('name', 'Feature extraction produces data', 'passed', false);
            return;
        end
        
        % Check for expected columns
        expected_cols = {'Participant', 'Condition'};
        has_participant = any(strcmp(output_data.Properties.VariableNames, 'Participant'));
        has_condition = any(strcmp(output_data.Properties.VariableNames, 'Condition'));
        
        if ~has_participant || ~has_condition
            fprintf('  ✗ FAIL: Missing required columns\n');
            test_result.tests{end+1} = struct('name', 'Output has required columns', 'passed', false);
            return;
        end
        
        fprintf('  ✓ PASS: Feature extraction completed successfully\n');
        fprintf('  ✓ Extracted %d conditions for P%02d\n', n_rows, test_participant);
        test_result.tests{end+1} = struct('name', 'Feature extraction successful', 'passed', true);
        
    catch ME
        fprintf('  ✗ FAIL: Error during feature extraction:\n');
        fprintf('    %s\n', ME.message);
        if ~isempty(ME.stack)
            fprintf('    in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
        test_result.tests{end+1} = struct('name', 'Feature extraction runs', 'passed', false);
        return;
    end
    
    %% Summary
    fprintf('\n=== TEST SUMMARY ===\n');
    all_passed = all(cellfun(@(t) t.passed, test_result.tests));
    
    for i = 1:length(test_result.tests)
        t = test_result.tests{i};
        if t.passed
            fprintf('  ✓ %s\n', t.name);
        else
            fprintf('  ✗ %s\n', t.name);
        end
    end
    
    fprintf('\n');
    if all_passed
        fprintf('✓✓✓ ALL TESTS PASSED ✓✓✓\n');
        fprintf('The refactored feature extraction script is compatible with cleaned EEG files!\n');
    else
        fprintf('✗✗✗ SOME TESTS FAILED ✗✗✗\n');
        fprintf('Please review the failures above before running full extraction.\n');
    end
    
    test_result.all_passed = all_passed;
    test_result.test_participant = test_participant;
end
