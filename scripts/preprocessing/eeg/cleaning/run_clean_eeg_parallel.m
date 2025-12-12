% RUN_CLEAN_EEG_PARALLEL - Parallel EEG cleaning for multiple participants
%
% Runs clean_eeg.m for multiple participants simultaneously using MATLAB's
% parallel computing toolbox. Each participant runs independently on separate
% workers.
%
% REQUIREMENTS:
%   - 32GB RAM minimum (3 participants × ~8GB each)
%   - Multi-core CPU (Ryzen 7 5700X with 8 cores is ideal)
%   - AMICA must be compiled and in path

% Add utilities path
addpath('scripts/utils');

% Initialize EEGLAB once
[~, ~, ~, ~] = eeglab('nogui');

% Load config
config = yaml.loadFile('config/general.yaml');

%% Configuration
% Define participant numbers to process
participant_numbers = [1, 2, 3];  % Update this list as needed

% Define paths
input_folder = 'output/processed';         % Where raw .set files are
output_folder = 'output/cleaned_eeg';
vis_base_folder = 'output/vis';
qc_folder = 'output/qc';

% Ensure output folders exist
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
if ~exist(vis_base_folder, 'dir'), mkdir(vis_base_folder); end
if ~exist(qc_folder, 'dir'), mkdir(qc_folder); end

%% Setup Parallel Pool
% Use 3 workers (one per participant) - adjust if running more/fewer participants
num_workers = min(3, length(participant_numbers));

fprintf('=============================================================\n');
fprintf('PARALLEL EEG CLEANING PIPELINE\n');
fprintf('=============================================================\n');
fprintf('Participants: %s\n', mat2str(participant_numbers));
fprintf('Workers: %d\n', num_workers);
fprintf('=============================================================\n\n');

% Start parallel pool
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', num_workers);
    fprintf('Parallel pool started with %d workers.\n\n', num_workers);
else
    fprintf('Using existing parallel pool with %d workers.\n\n', pool.NumWorkers);
end

%% Run Parallel Processing
results = cell(length(participant_numbers), 1);
success_count = 0;

parfor i = 1:length(participant_numbers)
    p_num = participant_numbers(i);
    
    fprintf('\n[Worker %d] Starting P%02d...\n', getCurrentTask().ID, p_num);
    
    try
        % Build paths
        raw_set_file = fullfile(input_folder, sprintf('P%02d.set', p_num));
        vis_folder = fullfile(vis_base_folder, sprintf('P%02d', p_num));
        
        % Check if input exists
        if ~exist(raw_set_file, 'file')
            error('Raw .set file not found: %s', raw_set_file);
        end
        
        % Run cleaning pipeline
        [EEG_clean, qc] = clean_eeg(raw_set_file, output_folder, p_num, ...
                                     vis_folder, qc_folder, []);
        
        % Store result
        results{i} = struct('participant', p_num, ...
                           'status', 'SUCCESS', ...
                           'qc', qc);
        
        fprintf('[Worker %d] P%02d COMPLETE ✓\n', getCurrentTask().ID, p_num);
        
    catch ME
        % Store error
        results{i} = struct('participant', p_num, ...
                           'status', 'FAILED', ...
                           'error', ME.message);
        
        fprintf('[Worker %d] P%02d FAILED: %s\n', getCurrentTask().ID, p_num, ME.message);
    end
end

%% Summary Report
fprintf('\n=============================================================\n');
fprintf('PARALLEL CLEANING COMPLETE\n');
fprintf('=============================================================\n');

for i = 1:length(results)
    r = results{i};
    if strcmp(r.status, 'SUCCESS')
        fprintf('✓ P%02d: SUCCESS\n', r.participant);
        success_count = success_count + 1;
    else
        fprintf('✗ P%02d: FAILED - %s\n', r.participant, r.error);
    end
end

fprintf('\nSuccess rate: %d/%d participants\n', success_count, length(participant_numbers));
fprintf('=============================================================\n');

% Save results summary
save(fullfile(output_folder, 'parallel_cleaning_results.mat'), 'results');

% Clean up parallel pool (optional - comment out to keep pool for next stage)
% delete(gcp('nocreate'));
