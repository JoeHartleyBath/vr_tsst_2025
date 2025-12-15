%% Test Feature Extraction on P01 (with chanlocs verification)
clearvars; close all; clc;

fprintf('=== TESTING FEATURE EXTRACTION ON P01 ===\n\n');

%% Setup
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

projectRoot = pwd;

%% Step 1: Verify P01_cleaned.set has proper chanlocs
fprintf('STEP 1: Verifying P01_cleaned.set has proper channel labels...\n');

cleaned_path = fullfile(projectRoot, 'output', 'cleaned_eeg', 'P01_cleaned.set');

if ~exist(cleaned_path, 'file')
    error('P01_cleaned.set not found! Run clean_p01_verify.m first.');
end

EEG_check = pop_loadset('filename', 'P01_cleaned.set', 'filepath', fullfile(projectRoot, 'output', 'cleaned_eeg'));

fprintf('  Channels: %d\n', EEG_check.nbchan);
fprintf('  Chanlocs length: %d\n', length(EEG_check.chanlocs));

if length(EEG_check.chanlocs) < EEG_check.nbchan
    error('Chanlocs missing or incomplete! Expected %d, got %d', EEG_check.nbchan, length(EEG_check.chanlocs));
end

fprintf('  First 10 labels: ');
for i = 1:min(10, length(EEG_check.chanlocs))
    fprintf('%s ', EEG_check.chanlocs(i).labels);
end
fprintf('\n');

% Check if labels match YAML region definitions
yaml_expected = {'Z2', 'Z4', 'Z6', 'L1', 'L2', 'L3', 'R1', 'R2', 'R3'};
matches = 0;
for i = 1:min(length(yaml_expected), length(EEG_check.chanlocs))
    if strcmp(EEG_check.chanlocs(i).labels, yaml_expected{i})
        matches = matches + 1;
    end
end

if matches > 5
    fprintf('  ✓ Channel labels match expected ANT Neuro format\n\n');
else
    warning('Channel labels may not match YAML config expectations');
end

clear EEG_check;

%% Step 2: Run Feature Extraction
fprintf('STEP 2: Running feature extraction...\n');
fprintf('  (This should take ~5-10 seconds)\n\n');

% Update script to process only P01
extract_script = fullfile(projectRoot, 'scripts', 'preprocessing', 'eeg', 'feature_extraction', 'extract_eeg_features.m');

try
    run(extract_script);
    fprintf('  ✓ Feature extraction completed!\n\n');
catch ME
    fprintf('  ✗ ERROR during extraction:\n');
    fprintf('    %s\n', ME.message);
    error('Feature extraction failed');
end

%% Step 3: Validate Output CSV
fprintf('STEP 3: Validating feature extraction output...\n');

csv_path = fullfile(projectRoot, 'output', 'aggregated', 'eeg_features.csv');

if ~exist(csv_path, 'file')
    error('Output CSV not found at: %s', csv_path);
end

data = readtable(csv_path);

fprintf('  File: %s\n', csv_path);
fprintf('  Rows: %d\n', height(data));
fprintf('  Columns: %d\n', width(data));
fprintf('  Conditions: %s\n', strjoin(unique(data.Condition), ', '));

%% Step 4: Check for Valid Values (not NaN)
fprintf('\nSTEP 4: Checking feature values...\n');

% Get numeric columns (skip Participant and Condition)
numeric_cols = 3:width(data);
first_row = data(1, numeric_cols);

% Count NaN vs valid values
nan_count = 0;
valid_count = 0;
sample_values = {};

for col = numeric_cols
    val = first_row{1, col - 2};  % Adjust index
    if isnumeric(val)
        if isnan(val)
            nan_count = nan_count + 1;
        else
            valid_count = valid_count + 1;
            if length(sample_values) < 10
                sample_values{end+1} = sprintf('%s = %.4f', data.Properties.VariableNames{col}, val);
            end
        end
    end
end

total_features = nan_count + valid_count;
valid_pct = (valid_count / total_features) * 100;

fprintf('  Total numeric features: %d\n', total_features);
fprintf('  Valid values: %d (%.1f%%)\n', valid_count, valid_pct);
fprintf('  NaN values: %d (%.1f%%)\n', nan_count, (nan_count/total_features)*100);

%% Step 5: Show Sample Values
fprintf('\nSTEP 5: Sample feature values (first condition):\n');
fprintf('  Participant: %d\n', data.Participant(1));
fprintf('  Condition: %s\n', data.Condition{1});
fprintf('\n  Sample features:\n');
for i = 1:min(10, length(sample_values))
    fprintf('    %s\n', sample_values{i});
end

%% Final Assessment
fprintf('\n=== FINAL ASSESSMENT ===\n');

if valid_pct > 90
    fprintf('✓✓✓ SUCCESS! Feature extraction is working perfectly!\n');
    fprintf('    - %d valid values out of %d features (%.1f%%)\n', valid_count, total_features, valid_pct);
    fprintf('    - Channel labels properly matched YAML config\n');
    fprintf('    - Ready to process P02 and P03\n');
    status = 'SUCCESS';
elseif valid_pct > 50
    fprintf('⚠ PARTIAL SUCCESS: Some features computed but issues remain\n');
    fprintf('    - %d valid values, %d NaN (%.1f%% valid)\n', valid_count, nan_count, valid_pct);
    fprintf('    - May need to review region definitions in YAML\n');
    status = 'PARTIAL';
else
    fprintf('✗ FAILURE: Most features are still NaN\n');
    fprintf('    - Only %.1f%% valid values\n', valid_pct);
    fprintf('    - Channel label mismatch likely still present\n');
    status = 'FAILURE';
end

fprintf('\n=== TEST COMPLETE ===\n');

% Save test results
test_results = struct();
test_results.status = status;
test_results.valid_count = valid_count;
test_results.nan_count = nan_count;
test_results.valid_percentage = valid_pct;
test_results.num_rows = height(data);
test_results.num_conditions = length(unique(data.Condition));
test_results.timestamp = datetime('now');

save(fullfile(projectRoot, 'output', 'feature_extraction_test_results.mat'), 'test_results');
fprintf('Test results saved to: output/feature_extraction_test_results.mat\n');
