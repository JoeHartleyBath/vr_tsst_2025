%% Quick Feature Extraction Test
% Tests feature extraction on synthetic data WITHOUT running full AMICA pipeline
%
% This script:
% - Generates minimal mock EEG data (8 channels, 5 minutes)
% - Tests feature extraction on 2 participants
% - Runs in ~1-2 minutes instead of hours
%
% Usage: Run this script to quickly test your feature extraction code

clear; clc;

%% Setup
addpath(genpath('c:\vr_tsst_2025\scripts\preprocessing\eeg\feature_extraction'));

% Initialize EEGLAB
eeglab nogui;

%% Load configs (use existing config files)
config_feat = load_feature_config();
config_cond = load_conditions_config();
config_gen = load_general_config();

% Override for testing: Use test data directory
test_data_dir = 'c:\vr_tsst_2025\data\test_data\cleaned_eeg';
config_gen.paths.cleaned_eeg = test_data_dir;
config_gen.paths.features = 'c:\vr_tsst_2025\data\test_data\features';

% Create directories
if ~exist(test_data_dir, 'dir')
    mkdir(test_data_dir);
end
if ~exist(config_gen.paths.features, 'dir')
    mkdir(config_gen.paths.features);
end

%% Generate mock data for 2 participants
fprintf('=== GENERATING MOCK DATA ===\n');
test_participants = [1, 2];

for p = test_participants
    create_mock_eeg_data(p, config_cond);
end

%% Disable parallel processing for easier debugging
config_feat.parallel.enabled = false;

%% Run feature extraction
fprintf('\n=== TESTING FEATURE EXTRACTION ===\n');
extract_eeg_features(test_participants, config_feat, config_cond, config_gen);

%% Verify output
fprintf('\n=== VERIFYING OUTPUT ===\n');
output_file = fullfile(config_gen.paths.features, 'eeg_features_all.csv');

if isfile(output_file)
    data = readtable(output_file);
    fprintf('✓ Output file created: %s\n', output_file);
    fprintf('  Rows: %d\n', height(data));
    fprintf('  Columns: %d\n', width(data));
    fprintf('  Participants: %s\n', mat2str(unique(data.ParticipantID)));
    fprintf('  Conditions: %s\n', strjoin(unique(data.Condition), ', '));
else
    fprintf('✗ Output file not found: %s\n', output_file);
end

fprintf('\n=== TEST COMPLETE ===\n');
