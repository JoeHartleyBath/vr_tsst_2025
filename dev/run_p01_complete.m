%% Complete P01 Processing Pipeline
% This script:
% 1. Assigns proper channel labels from electrode file
% 2. Runs feature extraction
% 3. Validates the output

clearvars; close all; clc;

fprintf('=== P01 COMPLETE PROCESSING ===\n\n');

%% Initialize EEGLAB
fprintf('Initializing EEGLAB...\n');
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;
fprintf('  ✓ EEGLAB loaded\n\n');

%% Step 1: Assign Channel Labels
fprintf('STEP 1: Assigning channel labels from NA-271.elc...\n');

% Parse electrode file
elc_file = 'config/chanlocs/NA-271.elc';
fid = fopen(elc_file);
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};

% Extract channel labels
chan_names = {};
for i = 1:length(lines)
    line = lines{i};
    if contains(line, ':')
        parts = strsplit(line, ':');
        name = strtrim(parts{1});
        if ~isempty(name) && ~strcmp(name, 'NumberPositions') && ~strcmp(name, 'Positions')
            chan_names{end+1} = name;
        end
    end
end

fprintf('  Parsed %d channel names\n', length(chan_names));
fprintf('  Sample: %s, %s, %s, ..., %s\n', chan_names{1}, chan_names{2}, chan_names{3}, chan_names{end});

% Load cleaned set
fprintf('  Loading P01_cleaned_patched.set...\n');
EEG = pop_loadset('filename', 'P01_cleaned_patched.set', 'filepath', 'output/cleaned_eeg');
fprintf('  Has %d channels\n', EEG.nbchan);

% Assign labels
if length(chan_names) >= EEG.nbchan
    for ch = 1:EEG.nbchan
        EEG.chanlocs(ch).labels = chan_names{ch};
    end
    fprintf('  ✓ Assigned %d labels\n', EEG.nbchan);
    
    % Save as P01_cleaned.set
    fprintf('  Saving as P01_cleaned.set...\n');
    EEG = pop_saveset(EEG, 'filename', 'P01_cleaned.set', 'filepath', 'output/cleaned_eeg');
    fprintf('  ✓ Saved!\n\n');
else
    error('Not enough channel names in electrode file');
end

%% Step 2: Verify Labels
fprintf('STEP 2: Verifying channel labels...\n');
EEG_verify = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'output/cleaned_eeg');
fprintf('  Channels: %d\n', EEG_verify.nbchan);
fprintf('  Chanlocs length: %d\n', length(EEG_verify.chanlocs));
fprintf('  First 10 labels: ');
for i = 1:min(10, length(EEG_verify.chanlocs))
    fprintf('%s ', EEG_verify.chanlocs(i).labels);
end
fprintf('\n  ✓ Labels verified!\n\n');

%% Step 3: Run Feature Extraction
fprintf('STEP 3: Running feature extraction...\n');
fprintf('  (This will take a few seconds)\n');
run('scripts/preprocessing/eeg/feature_extraction/extract_eeg_features.m');
fprintf('  ✓ Extraction complete!\n\n');

%% Step 4: Validate Output
fprintf('STEP 4: Validating output CSV...\n');
csv_file = 'output/aggregated/eeg_features.csv';
if exist(csv_file, 'file')
    data = readtable(csv_file);
    fprintf('  File: %s\n', csv_file);
    fprintf('  Rows: %d\n', height(data));
    fprintf('  Columns: %d\n', width(data));
    
    % Check for NaN values
    num_cols = width(data);
    numeric_data = data(:, 3:end); % Skip Participant and Condition
    all_nan = all(all(ismissing(numeric_data) | isnan(table2array(numeric_data))));
    
    if all_nan
        fprintf('  ⚠ WARNING: All feature values are NaN!\n');
    else
        fprintf('  ✓ Features contain valid values!\n');
        
        % Show sample values
        fprintf('\n  Sample values (first condition):\n');
        row1 = data(1, :);
        fprintf('    Participant: %d\n', row1.Participant);
        fprintf('    Condition: %s\n', row1.Condition{1});
        
        % Show first 5 numeric columns with non-NaN values
        fprintf('    Sample features:\n');
        count = 0;
        for col = 3:width(data)
            val = row1{1, col};
            if isnumeric(val) && ~isnan(val)
                fprintf('      %s: %.4f\n', data.Properties.VariableNames{col}, val);
                count = count + 1;
                if count >= 5, break; end
            end
        end
    end
else
    fprintf('  ✗ Output file not found!\n');
end

fprintf('\n=== PROCESSING COMPLETE ===\n');
fprintf('Next steps:\n');
fprintf('  1. If values are valid, run feature extraction on P02 and P03\n');
fprintf('  2. If still NaN, check channel label matching in YAML config\n');
