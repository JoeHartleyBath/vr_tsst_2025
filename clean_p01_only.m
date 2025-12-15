% Quick script to clean P01 only
clearvars; close all; clc;

% Setup paths
projectRoot = 'C:/vr_tsst_2025';
cd(projectRoot);

% Add toolboxes
addpath(genpath('C:/MATLAB/toolboxes/eeglab2025.1.0'));
addpath(genpath(fullfile(projectRoot, 'scripts', 'utils')));
addpath(fullfile(projectRoot, 'scripts', 'preprocessing', 'eeg', 'cleaning'));

% Initialize EEGLAB
eeglab nogui;

% Define paths
raw_set_path = fullfile(projectRoot, 'output', 'sets', 'P01.set');
output_folder = fullfile(projectRoot, 'output', 'cleaned_eeg');
vis_folder = fullfile(projectRoot, 'output', 'vis', 'P01');
qc_folder = fullfile(projectRoot, 'output', 'qc');

% Create folders if needed
if ~exist(vis_folder, 'dir'), mkdir(vis_folder); end

fprintf('Starting P01 cleaning...\n');
fprintf('Input: %s\n', raw_set_path);
fprintf('Output: %s\n\n', output_folder);

% Run cleaning
try
    [EEG, qc] = clean_eeg(raw_set_path, output_folder, 1, vis_folder, qc_folder, projectRoot);
    fprintf('\nP01 cleaning completed successfully!\n');
    
    % Verify chanlocs
    cleaned_set = fullfile(output_folder, 'P01_cleaned.set');
    EEG_test = pop_loadset('filename', 'P01_cleaned.set', 'filepath', output_folder);
    fprintf('\nVerifying P01_cleaned.set:\n');
    fprintf('  Channels: %d\n', length(EEG_test.chanlocs));
    if length(EEG_test.chanlocs) > 0
        fprintf('  First 5 channel labels:\n');
        for i = 1:min(5, length(EEG_test.chanlocs))
            if isfield(EEG_test.chanlocs, 'labels')
                fprintf('    %d: %s\n', i, EEG_test.chanlocs(i).labels);
            end
        end
    else
        warning('No channel locations found!');
    end
    
catch ME
    fprintf('\nERROR during cleaning: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('  %s at line %d\n', ME.stack(k).name, ME.stack(k).line);
    end
end

fprintf('\nDone.\n');
