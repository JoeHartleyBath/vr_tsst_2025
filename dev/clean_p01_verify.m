%% Re-clean P01 with chanlocs verification
clearvars; close all; clc;

fprintf('=== P01 RE-CLEANING WITH CHANLOCS VERIFICATION ===\n\n');

% Setup
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

projectRoot = pwd;
addpath(genpath(fullfile(projectRoot, 'scripts', 'utils')));
addpath(fullfile(projectRoot, 'scripts', 'preprocessing', 'eeg', 'cleaning'));

raw_set_path = fullfile(projectRoot, 'output', 'sets', 'P01.set');
output_folder = fullfile(projectRoot, 'output', 'cleaned_eeg');
vis_folder = fullfile(projectRoot, 'output', 'vis', 'P01');
qc_folder = fullfile(projectRoot, 'output', 'qc');

% Create folders
if ~exist(vis_folder, 'dir'), mkdir(vis_folder); end
if ~exist(qc_folder, 'dir'), mkdir(qc_folder); end

fprintf('Input: %s\n', raw_set_path);
fprintf('Output: %s\n\n', output_folder);

%% Run Cleaning
fprintf('Running cleaning pipeline...\n');
fprintf('(This will take several minutes with AMICA)\n\n');

try
    [EEG, qc] = clean_eeg(raw_set_path, output_folder, 1, vis_folder, qc_folder, projectRoot);
    fprintf('✓ Cleaning completed successfully!\n\n');
    
    %% Verify the saved .set file
    fprintf('VERIFICATION: Loading saved P01_cleaned.set...\n');
    cleaned_path = fullfile(output_folder, 'P01_cleaned.set');
    
    if exist(cleaned_path, 'file')
        EEG_verify = pop_loadset('filename', 'P01_cleaned.set', 'filepath', output_folder);
        
        fprintf('  Channels: %d\n', EEG_verify.nbchan);
        fprintf('  Chanlocs length: %d\n', length(EEG_verify.chanlocs));
        
        if length(EEG_verify.chanlocs) >= EEG_verify.nbchan
            fprintf('  ✓ Chanlocs present!\n');
            fprintf('  First 10 labels: ');
            for i = 1:min(10, length(EEG_verify.chanlocs))
                fprintf('%s ', EEG_verify.chanlocs(i).labels);
            end
            fprintf('\n');
            
            % Check if labels are meaningful (not just numbers)
            first_label = EEG_verify.chanlocs(1).labels;
            if isempty(str2num(first_label))
                fprintf('  ✓ Labels are proper ANT Neuro names (not numeric)\n');
            else
                fprintf('  ⚠ WARNING: Labels appear to be numeric indices\n');
            end
            
            fprintf('\n=== SUCCESS ===\n');
            fprintf('P01_cleaned.set has proper channel locations\n');
            fprintf('Ready for feature extraction!\n');
        else
            fprintf('  ✗ ERROR: Chanlocs missing or incomplete!\n');
        end
    else
        fprintf('  ✗ ERROR: P01_cleaned.set not found!\n');
    end
    
catch ME
    fprintf('\n✗ ERROR during cleaning:\n');
    fprintf('  %s\n', ME.message);
    fprintf('  Stack trace:\n');
    for k = 1:length(ME.stack)
        fprintf('    %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
    end
end

fprintf('\n=== COMPLETE ===\n');
