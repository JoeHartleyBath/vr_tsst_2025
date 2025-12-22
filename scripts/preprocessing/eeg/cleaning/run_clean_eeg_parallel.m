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

% Add utilities path (guard against missing folder)
utilsPath = fullfile(pwd, 'scripts', 'utils');
if exist(utilsPath, 'dir')
    addpath(utilsPath);
end

% Initialize EEGLAB once. If not on path, try project startup
if exist('eeglab', 'file') ~= 2
    try
        % Resolve project root relative to this script and run startup there
        thisDir = fileparts(mfilename('fullpath'));
        projectRoot = fullfile(thisDir, '..', '..', '..', '..');
        projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
        run(fullfile(projectRoot, 'startup.m'));
    catch ME
        fprintf('startup.m failed: %s\n', ME.message);
    end
end
if exist('eeglab', 'file') == 2
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');
else
    fprintf('EEGLAB function not found on path; proceeding assuming startup initialized toolboxes.\n');
end

% Resolve project root from this script's location
thisDir = fileparts(mfilename('fullpath'));
% Script is in scripts/preprocessing/eeg/cleaning, so go up 4 levels to project root
projectRoot = fullfile(thisDir, '..', '..', '..', '..');
projectRoot = char(java.io.File(projectRoot).getCanonicalPath());

fprintf('[init] Script dir: %s\n', thisDir);
fprintf('[init] Project root: %s\n', projectRoot);

% Change to project root so all relative paths work
cd(projectRoot);

% Ensure cleaning and utils folders are on path BEFORE parpool starts
% so workers inherit the path
cleaningFolder = fullfile(projectRoot, 'scripts', 'preprocessing', 'eeg', 'cleaning');
utilsFolder = fullfile(projectRoot, 'scripts', 'utils');
if exist(cleaningFolder, 'dir')
    addpath(cleaningFolder);
    fprintf('[init] Added cleaning folder to path: %s\n', cleaningFolder);
end
if exist(utilsFolder, 'dir')
    addpath(utilsFolder);
    fprintf('[init] Added utils folder to path: %s\n', utilsFolder);
end

% Load config (YAML)
% Use ReadYaml (yamlmatlab) if available, otherwise SimpleYAML
% Resolve path from project root
configPath = fullfile(projectRoot, 'config', 'general.yaml');
config = struct();
if exist('ReadYaml', 'file') == 2
    try
        config = ReadYaml(configPath);
        fprintf('[config] Loaded via ReadYaml (yamlmatlab)\n');
    catch ME
        fprintf('[config] ReadYaml failed: %s; trying SimpleYAML fallback\n', ME.message);
    end
end
if isempty(fieldnames(config)) && exist('SimpleYAML', 'file') == 2
    try
        config = SimpleYAML.readFile(configPath);
        fprintf('[config] Loaded via SimpleYAML (project fallback)\n');
    catch ME
        fprintf('[config] SimpleYAML failed: %s; proceeding without config\n', ME.message);
    end
end
if isempty(fieldnames(config))
    fprintf('[config] No config loaded; using defaults.\n');
end

%% Configuration
% Define participant numbers to process
% Discover participants from output/sets automatically
setFolder = fullfile(projectRoot, 'output', 'sets');
fprintf('[discover] Looking for .set files in: %s\n', setFolder);

if ~exist(setFolder, 'dir')
    fprintf('[discover] ERROR: Folder does not exist: %s\n', setFolder);
    return;
end

setFiles = dir(fullfile(setFolder, 'P*.set'));
fprintf('[discover] Found %d .set files\n', numel(setFiles));

participant_numbers = [];
for k = 1:numel(setFiles)
    nm = setFiles(k).name; % e.g., P01.set
    p = sscanf(nm, 'P%02d.set');
    if ~isempty(p)
        participant_numbers(end+1) = p; %#ok<AGROW>
    end
end
participant_numbers = sort(participant_numbers);

if isempty(participant_numbers)
    fprintf('[discover] ERROR: No participants found in .set files\n');
    return;
end

fprintf('[discover] Extracted %d participant numbers: %s\n', numel(participant_numbers), mat2str(participant_numbers(1:min(5, end))));

% Define paths
% Use converted raw sets location
input_folder = 'output/sets';              % Where raw .set files are
output_folder = 'output/cleaned_eeg';
vis_base_folder = 'output/vis';
qc_folder = 'output/qc';

% Ensure output folders exist
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
if ~exist(vis_base_folder, 'dir'), mkdir(vis_base_folder); end
if ~exist(qc_folder, 'dir'), mkdir(qc_folder); end

%% Setup Parallel Pool
% Balance CPU utilization with memory constraints
% AMICA uses ~8GB per participant; 3 workers = ~24GB max (safer for high CPU load)
num_workers = min(3, length(participant_numbers));

fprintf('=============================================================\n');
fprintf('PARALLEL EEG CLEANING PIPELINE\n');
fprintf('=============================================================\n');
fprintf('Participants: %s\n', mat2str(participant_numbers));
fprintf('Workers: %d\n', num_workers);
fprintf('=============================================================\n\n');

% Start parallel pool if Parallel Computing Toolbox is available
hasPCT = (exist('parpool','file') == 2) && (exist('gcp','file') == 2) && num_workers > 0;
if hasPCT
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local', num_workers);
        fprintf('Parallel pool started with %d workers.\n\n', num_workers);
    else
        fprintf('Using existing parallel pool with %d workers.\n\n', pool.NumWorkers);
    end
else
    fprintf('Parallel Computing Toolbox not available or zero workers requested; running serially.\n\n');
end

%% Run Parallel Processing
results = cell(length(participant_numbers), 1);
success_count = 0;

procFun = @(i) deal(participant_numbers(i));

if hasPCT
parfor i = 1:length(participant_numbers)
    p_num = participant_numbers(i);
    
    fprintf('\n[Worker %d] Starting P%02d...\n', getCurrentTask().ID, p_num);
    
    try
        % Build paths
        raw_set_file = fullfile(input_folder, sprintf('P%02d.set', p_num));
        vis_folder = fullfile(vis_base_folder, sprintf('P%02d', p_num));
        cleaned_file = fullfile(output_folder, sprintf('P%02d_cleaned.set', p_num));
        
        % Check if input exists
        if ~exist(raw_set_file, 'file')
            error('Raw .set file not found: %s', raw_set_file);
        end

        % Skip if already cleaned
        if exist(cleaned_file, 'file')
            fprintf('[Worker %d] P%02d already cleaned, skipping.\n', getCurrentTask().ID, p_num);
            results{i} = struct('participant', p_num, 'status', 'SKIPPED', 'qc', []);
            continue;
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
else
for i = 1:length(participant_numbers)
    p_num = participant_numbers(i);
    fprintf('\n[Serial] Starting P%02d...\n', p_num);
    try
        raw_set_file = fullfile(input_folder, sprintf('P%02d.set', p_num));
        vis_folder = fullfile(vis_base_folder, sprintf('P%02d', p_num));
        if ~exist(raw_set_file, 'file')
            error('Raw .set file not found: %s', raw_set_file);
        end
        [EEG_clean, qc] = clean_eeg(raw_set_file, output_folder, p_num, ...
                                     vis_folder, qc_folder, []);
        results{i} = struct('participant', p_num, 'status', 'SUCCESS', 'qc', qc);
        fprintf('[Serial] P%02d COMPLETE ✓\n', p_num);
    catch ME
        results{i} = struct('participant', p_num, 'status', 'FAILED', 'error', ME.message);
        fprintf('[Serial] P%02d FAILED: %s\n', p_num, ME.message);
    end
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
        if isfield(r, 'error')
            fprintf('✗ P%02d: FAILED - %s\n', r.participant, r.error);
        else
            fprintf('✗ P%02d: %s\n', r.participant, r.status);
        end
    end
end

fprintf('\nSuccess rate: %d/%d participants\n', success_count, length(participant_numbers));
fprintf('=============================================================\n');

% Save results summary
save(fullfile(output_folder, 'parallel_cleaning_results.mat'), 'results');

% Clean up parallel pool (optional - comment out to keep pool for next stage)
if hasPCT
    % delete(gcp('nocreate'));
end
