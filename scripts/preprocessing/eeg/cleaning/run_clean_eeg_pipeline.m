% RUN_CLEAN_EEG_PIPELINE - End-to-end pipeline runner
%
% This script runs the complete EEG cleaning pipeline for one or more
% participants, from raw .set files to cleaned outputs with QC metrics.
%
% SETUP:
%   1. Ensure raw .set files exist in data/raw/eeg/ (created by xdf_to_set.py)
%   2. Update participant_numbers list below
%   3. Verify paths in config/general.yaml
%   4. Run this script

% Add utilities path
addpath(genpath('scripts/utils'));
addpath(genpath('C:/MATLAB/toolboxes/eeglab'));
addpath(genpath('C:/MATLAB/toolboxes/amica'));

% Initialize EEGLAB in GUI mode (non-batch)
try
    eeglab nogui;
catch
    warning('EEGLAB GUI initialization failed, continuing anyway');
end

% Load config
config = yaml.loadFile('config/general.yaml');

%% Configuration
% Define participant numbers to process
participant_numbers = [1, 2, 3];  % Update this list as needed (now defaults to P01-P03)

% Define paths
raw_eeg_folder = 'output/sets';  % Input comes from Stage 1 (XDF→SET conversion)
output_folder = 'output/cleaned_eeg';
vis_base_folder = 'output/vis';
qc_folder = 'output/qc';

% Ensure output folders exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
if ~exist(vis_base_folder, 'dir')
    mkdir(vis_base_folder);
end
if ~exist(qc_folder, 'dir')
    mkdir(qc_folder);
end

%% Process Each Participant
fprintf('=============================================================\n');
fprintf('EEG CLEANING PIPELINE - BATCH PROCESSING\n');
fprintf('=============================================================\n');
fprintf('Participants to process: %s\n', mat2str(participant_numbers));
fprintf('=============================================================\n\n');

results = struct();
success_count = 0;
failure_count = 0;

for i = 1:length(participant_numbers)
    participant_num = participant_numbers(i);
    
    fprintf('\n');
    fprintf('=============================================================\n');
    fprintf('PROCESSING PARTICIPANT %02d (%d of %d)\n', participant_num, i, length(participant_numbers));
    fprintf('=============================================================\n');
    
    % Define file paths
    raw_set_filename = sprintf('P%02d.set', participant_num);
    raw_set_path = fullfile(raw_eeg_folder, raw_set_filename);
    vis_folder = fullfile(vis_base_folder, sprintf('P%02d', participant_num));
    
    % Check if raw file exists
    if ~exist(raw_set_path, 'file')
        fprintf('ERROR: Raw .set file not found: %s\n', raw_set_path);
        fprintf('Skipping participant %02d\n', participant_num);
        results(i).participant = participant_num;
        results(i).success = false;
        results(i).error = 'Raw .set file not found';
        failure_count = failure_count + 1;
        continue;
    end
    
    fprintf('Input file: %s\n', raw_set_path);
    fprintf('Output folder: %s\n', output_folder);
    fprintf('Visualization folder: %s\n', vis_folder);
    fprintf('QC folder: %s\n\n', qc_folder);
    
    % Run cleaning pipeline
    tic;
    try
        [EEG, qc] = clean_eeg(raw_set_path, ...
                              output_folder, ...
                              participant_num, ...
                              vis_folder, ...
                              qc_folder, ...
                              config);
        
        elapsed = toc;
        
        % Store results
        results(i).participant = participant_num;
        results(i).success = true;
        results(i).elapsed_time = elapsed;
        results(i).nbchan = EEG.nbchan;
        results(i).pnts = EEG.pnts;
        results(i).srate = EEG.srate;
        results(i).n_events = length(EEG.event);
        results(i).qc = qc;
        
        success_count = success_count + 1;
        
        fprintf('\n--- PARTICIPANT %02d COMPLETED SUCCESSFULLY ---\n', participant_num);
        fprintf('Processing time: %.1f seconds (%.1f minutes)\n', elapsed, elapsed/60);
        fprintf('Output channels: %d\n', EEG.nbchan);
        fprintf('Output samples: %d\n', EEG.pnts);
        fprintf('Bad channels: %d\n', qc.nBad);
        fprintf('Samples retained: %.1f%%\n', qc.percSamplesRetained);
        fprintf('ICs removed: %d\n', qc.ICsRemoved);
        
    catch ME
        elapsed = toc;
        
        results(i).participant = participant_num;
        results(i).success = false;
        results(i).elapsed_time = elapsed;
        results(i).error = ME.message;
        
        failure_count = failure_count + 1;
        
        fprintf('\n--- PARTICIPANT %02d FAILED ---\n', participant_num);
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for j = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
        end
        
        % Log error to file
        error_log = fullfile(output_folder, sprintf('P%02d_ERROR.txt', participant_num));
        fid = fopen(error_log, 'w');
        fprintf(fid, 'Error processing P%02d\n', participant_num);
        fprintf(fid, 'Timestamp: %s\n', datestr(now));
        fprintf(fid, 'Error: %s\n\n', ME.message);
        fprintf(fid, 'Stack trace:\n');
        for j = 1:length(ME.stack)
            fprintf(fid, '  %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
        end
        fclose(fid);
    end
end

%% Summary Report
fprintf('\n\n');
fprintf('=============================================================\n');
fprintf('BATCH PROCESSING COMPLETE\n');
fprintf('=============================================================\n');
fprintf('Total participants: %d\n', length(participant_numbers));
fprintf('Successful: %d\n', success_count);
fprintf('Failed: %d\n', failure_count);
fprintf('Success rate: %.1f%%\n', 100 * success_count / length(participant_numbers));
fprintf('=============================================================\n\n');

% Display individual results
fprintf('Individual Results:\n');
fprintf('%-5s %-10s %-10s %-10s %-10s %-15s\n', 'P#', 'Status', 'Time(s)', 'Bad Ch', 'Retain%', 'ICs Removed');
fprintf('---------------------------------------------------------------------\n');
for i = 1:length(results)
    if results(i).success
        fprintf('P%02d   SUCCESS    %8.1f   %8d   %8.1f   %8d\n', ...
            results(i).participant, ...
            results(i).elapsed_time, ...
            results(i).qc.nBad, ...
            results(i).qc.percSamplesRetained, ...
            results(i).qc.ICsRemoved);
    else
        fprintf('P%02d   FAILED     %8.1f   (Error: %s)\n', ...
            results(i).participant, ...
            results(i).elapsed_time, ...
            results(i).error);
    end
end
fprintf('=============================================================\n');

% Save batch results
batch_results_path = fullfile(output_folder, sprintf('batch_results_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(batch_results_path, 'results', 'participant_numbers', 'success_count', 'failure_count');
fprintf('\nBatch results saved: %s\n', batch_results_path);
