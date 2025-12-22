% RUN_CLEAN_EEG_PIPELINE_PARALLEL - Parallel EEG cleaning pipeline runner
%
% This script runs the complete EEG cleaning pipeline for multiple participants
% in parallel using 8 workers with 2 threads each (16 total threads for Ryzen 7 5700X).
%
% SETUP:
%   1. Ensure raw .set files exist in output/sets/ (created by xdf_to_set.py)
%   2. Update participant_numbers list below
%   3. Verify paths in config/general.yaml
%   4. Run this script
%
% Performance:
%   - 8 parallel workers × 2 AMICA threads each = 16 threads utilized (100% CPU)
%   - Est. 48 participants in 2-3 hours (vs 8-12 sequential)

% Load config (project-root absolute path, independent of cwd)
% Explicitly set paths for batch mode compatibility
projectRoot = 'C:\vr_tsst_2025';
cfgPath = fullfile(projectRoot, 'config', 'general.yaml');
if ~exist(cfgPath, 'file')
    error('Could not open file %s. No such file or directory.', cfgPath);
end

% Load YAML config with robust fallbacks
try
    config = yaml.loadFile(cfgPath);
catch
    try
        % Alternate YAML parser API
        config = ReadYaml(cfgPath);
    catch
        warning('YAML parser not available. Proceeding with empty config.');
        config = struct();
    end
end

% Add utilities path
addpath(genpath(fullfile(projectRoot, 'scripts', 'utils')));

% Dynamically locate EEGLAB toolbox and add to path
eeglabAdded = false;
try
    % Common installation root
    eeglabRoot = 'C:/MATLAB/toolboxes';
    if exist(fullfile(eeglabRoot, 'eeglab'), 'dir')
        addpath(genpath(fullfile(eeglabRoot, 'eeglab')));
        eeglabAdded = true;
    else
        d = dir(fullfile(eeglabRoot, 'eeglab*'));
        if ~isempty(d)
            addpath(genpath(fullfile(eeglabRoot, d(1).name)));
            eeglabAdded = true;
        end
    end
catch
    % Continue to verification below
end

% Also attempt to add AMICA if present
try
    amicaRoot = 'C:/MATLAB/toolboxes';
    if exist(fullfile(amicaRoot, 'amica'), 'dir')
        addpath(genpath(fullfile(amicaRoot, 'amica')));
    else
        d2 = dir(fullfile(amicaRoot, 'amica*'));
        if ~isempty(d2)
            addpath(genpath(fullfile(amicaRoot, d2(1).name)));
        end
    end
catch
end

% Verify EEGLAB core function is available
if exist('pop_loadset', 'file') ~= 2
    error(['EEGLAB not found on MATLAB path (missing pop_loadset). ', ...
           'Please install EEGLAB or update paths. ', ...
           'Tip: run install_eeglab_and_r.ps1 or adjust startup.m.']);
end

% Initialize EEGLAB (nogui); continue if GUI fails
try
    eeglab nogui;
catch
    warning('EEGLAB initialization failed, continuing anyway');
end

%% Configuration
% PARALLEL PROCESSING SETTINGS
num_parallel_workers = 8;           % Number of parallel workers (increased from 4)
threads_per_worker = 2;             % AMICA threads per worker (8 × 2 = 16 total)

% Define participant numbers to process
participant_numbers = [1:48];       % Process all 48 participants

% Define paths (use absolute paths relative to project root)
raw_eeg_folder = fullfile(projectRoot, 'output', 'sets');
output_folder = fullfile(projectRoot, 'output', 'cleaned_eeg');
vis_base_folder = fullfile(projectRoot, 'output', 'vis');
qc_folder = fullfile(projectRoot, 'output', 'qc');

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

%% Setup Parallel Processing Pool
fprintf('=============================================================\n');
fprintf('EEG CLEANING PIPELINE - PARALLEL BATCH PROCESSING\n');
fprintf('=============================================================\n');
fprintf('Configuration:\n');
fprintf('  Parallel workers: %d\n', num_parallel_workers);
fprintf('  Threads per worker: %d\n', threads_per_worker);
fprintf('  Total threads: %d (Ryzen 7 5700X: 16 logical threads)\n', num_parallel_workers * threads_per_worker);
fprintf('  Participants: %d\n', length(participant_numbers));
fprintf('=============================================================\n\n');

% Create or get existing parallel pool
poolobj = gcp('nocreate');  % Get existing pool without creating
if isempty(poolobj)
    fprintf('Creating parallel pool with %d workers...\n', num_parallel_workers);
    poolobj = parpool('local', num_parallel_workers, 'IdleTimeout', 120);
    fprintf('Pool created successfully.\n\n');
else
    fprintf('Using existing parallel pool with %d workers.\n\n', poolobj.NumWorkers);
end

%% Initialize Results Structure
results = struct('participant', {}, 'success', {}, 'elapsed_time', {}, ...
                 'error', {}, 'nbchan', {}, 'pnts', {}, 'srate', {}, ...
                 'n_events', {}, 'qc', {});
results_idx = 0;

%% Process Each Participant in Parallel
fprintf('Starting parallel processing of %d participants...\n', length(participant_numbers));
fprintf('Estimated time: 2-3 hours\n\n');

tic;
start_time = datetime('now');

% Pre-allocate results array
results(length(participant_numbers)).participant = 0;

parfor i = 1:length(participant_numbers)
    participant_num = participant_numbers(i);
    
    % Define file paths
    raw_set_filename = sprintf('P%02d.set', participant_num);
    raw_set_path = fullfile(raw_eeg_folder, raw_set_filename);
    vis_folder = fullfile(vis_base_folder, sprintf('P%02d', participant_num));
    
    % Check if raw file exists
    if ~exist(raw_set_path, 'file')
        fprintf('ERROR (P%02d): Raw .set file not found: %s\n', participant_num, raw_set_path);
        results(i).participant = participant_num;
        results(i).success = false;
        results(i).elapsed_time = 0;
        results(i).error = 'Raw .set file not found';
        continue;
    end
    
    % Run cleaning pipeline with thread override
    tic;
    try
        [EEG, qc] = clean_eeg(raw_set_path, ...
                              output_folder, ...
                              participant_num, ...
                              vis_folder, ...
                              qc_folder, ...
                              config, ...
                              threads_per_worker);  % Pass thread count for parallel processing
        
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
        
        fprintf('[%d/%d] P%02d: SUCCESS (%.1f sec, %d bad ch, %.1f%% retained, %d ICs removed)\n', ...
            i, length(participant_numbers), participant_num, elapsed, ...
            qc.nBad, qc.percSamplesRetained, qc.ICsRemoved);
        
    catch ME
        elapsed = toc;
        
        results(i).participant = participant_num;
        results(i).success = false;
        results(i).elapsed_time = elapsed;
        results(i).error = ME.message;
        
        fprintf('[%d/%d] P%02d: FAILED (%.1f sec) - %s\n', ...
            i, length(participant_numbers), participant_num, elapsed, ME.message);
        
        % Log error to file
        error_log = fullfile(output_folder, sprintf('P%02d_ERROR.txt', participant_num));
        try
            fid = fopen(error_log, 'w');
            fprintf(fid, 'Error processing P%02d\n', participant_num);
            fprintf(fid, 'Timestamp: %s\n', datestr(now));
            fprintf(fid, 'Error: %s\n\n', ME.message);
            fprintf(fid, 'Stack trace:\n');
            for j = 1:length(ME.stack)
                fprintf(fid, '  %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
            end
            fclose(fid);
        catch
            % Error logging failed, continue anyway
        end
    end
end

total_elapsed = toc;
end_time = datetime('now');

%% Summary Report
fprintf('\n');
fprintf('=============================================================\n');
fprintf('PARALLEL BATCH PROCESSING COMPLETE\n');
fprintf('=============================================================\n');
fprintf('Total time: %.1f hours (%.1f minutes)\n', total_elapsed/3600, total_elapsed/60);
fprintf('Start time: %s\n', start_time);
fprintf('End time: %s\n', end_time);

% Count successes and failures
success_count = sum([results.success]);
failure_count = length(participant_numbers) - success_count;

fprintf('\nSummary:\n');
fprintf('Total participants: %d\n', length(participant_numbers));
fprintf('Successful: %d\n', success_count);
fprintf('Failed: %d\n', failure_count);
fprintf('Success rate: %.1f%%\n', 100 * success_count / length(participant_numbers));

if success_count > 0
    avg_time = mean([results([results.success]).elapsed_time]);
    fprintf('Average processing time per participant: %.1f seconds\n', avg_time);
end

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
batch_results_path = fullfile(output_folder, sprintf('batch_results_parallel_%s.mat', datestr(now, 'yyyymmdd_HHMMSS')));
save(batch_results_path, 'results', 'participant_numbers', 'success_count', 'failure_count', 'num_parallel_workers', 'threads_per_worker', 'total_elapsed');
fprintf('\nBatch results saved: %s\n', batch_results_path);

% Optional: close pool to free resources
% delete(poolobj);
% fprintf('Parallel pool closed.\n');
