%% extract_eeg_features.m
% -------------------------------------------------------------------------
% Streamlined EEG feature extraction for VR-TSST study
% 
% Computes aggregated features per condition:
%   • Band power (per region × frequency band)
%   • Power ratios (frontal asymmetry, alpha/beta, theta/beta, etc.)
%   • Entropy (sample & spectral entropy per region)
%
% Configuration-driven: All parameters loaded from YAML files
% No rolling windows, no statistics, no connectivity features
%
% Dependencies:
%   - EEGLAB
%   - EntropyHub or SampEn function
%   - yaml.loadFile (YAML parser)
%   - config/eeg_feature_extraction.yaml
%   - config/conditions.yaml
%   - config/general.yaml
% -------------------------------------------------------------------------

clearvars;
close all;
clc;

%% ========================================================================
%  SETUP: PROJECT ROOT & PATHS
%% ========================================================================

% Compute project root from this script location
% This script is in: scripts/preprocessing/eeg/feature_extraction/
% Need to go up 4 levels to reach project root
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
projectRoot = fullfile(thisDir, '..', '..', '..', '..');
projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
cd(projectRoot);

% Ensure yamlmatlab toolbox is available (for YAML config loading)
yaml_candidate_paths = {
    'C:/MATLAB/toolboxes/yamlmatlab', ...
    'C:/Program Files/MATLAB/R2025b/toolbox/yamlmatlab', ...
    fullfile(projectRoot, 'external', 'yamlmatlab'), ...
    fullfile(projectRoot, 'vendor', 'yamlmatlab')
};
for yc = 1:numel(yaml_candidate_paths)
    if exist(yaml_candidate_paths{yc}, 'dir')
        addpath(genpath(yaml_candidate_paths{yc}));
    end
end

%% ========================================================================
%  CONFIGURATION LOADING
%% ========================================================================

% Make sure yamlmatlab is available
% Check if yaml package namespace is accessible
try
    yaml.ReadYaml; % This will error if function doesn't exist, but package is found
    yaml_available = true;
catch
    yaml_available = exist('yaml.ReadYaml','file') > 0;
end

fprintf('YAML parser available: %d\n', yaml_available);

% Load configurations with robust YAML fallback
loaded = false;

try
    % Always use package namespace form (yaml.ReadYaml)
    config_feat = yaml.ReadYaml('config/eeg_feature_extraction.yaml');
    config_cond = yaml.ReadYaml('config/conditions.yaml');
    config_gen = yaml.ReadYaml('config/general.yaml');
    loaded = true;
    fprintf('YAML configs loaded successfully\n');
catch ME1
    % Fallback: minimal hardcoded config (keeps feature names aligned with downstream code)
    fprintf('Warning: YAML parser failed. Using fallback config.\n');
    config_feat = struct();
    config_feat.frequency_bands = struct('Delta', [1 4], 'Theta', [4 8], 'Alpha', [8 13], 'Beta', [13 30]);
    % Use numeric channel indices for fallback (1-64 ANT Neuro typical layout)
    config_feat.regions = struct( ...
        'FrontalLeft',    {{num2str(1), num2str(2), num2str(3), num2str(4), num2str(5)}}, ...
        'FrontalRight',   {{num2str(60), num2str(61), num2str(62), num2str(63), num2str(64)}}, ...
        'OverallFrontal', {{num2str(1), num2str(2), num2str(3), num2str(4), num2str(5), num2str(60), num2str(61), num2str(62), num2str(63), num2str(64)}}, ...
        'FrontalMidline', {{num2str(30), num2str(31), num2str(32)}});
    config_feat.parallel = struct('enabled', false, 'num_workers', 4);
    config_feat.output = struct('folder', 'aggregated', 'filename', 'eeg_features.csv');
    config_feat.toolbox_paths = struct('eeglab', 'C:/MATLAB/toolboxes/eeglab2025.1.0', ...
                                      'entropy_hub', '', 'utils', 'scripts/utils', ...
                                      'yamlmatlab', 'C:/MATLAB/toolboxes/yamlmatlab');
    config_feat.features = struct('ratios', true, 'entropy', false);
    config_feat.ratios = {'Frontal_Alpha_Asymmetry', 'Alpha_Beta_Ratio', 'Theta_Beta_Ratio', 'RightFrontal_Alpha', 'Theta_Alpha_Ratio'};
    config_feat.entropy_metrics = {'SampleEntropy'};
    
    config_cond = struct();
    config_cond.conditions = struct('Exposure', struct('duration', 180));
    
    config_gen = struct();
    config_gen.paths = struct('output', 'output', 'cleaned_eeg', 'output/cleaned_eeg', 'eeg_data', 'data/raw/eeg', 'events', 'data/raw/events');
    loaded = true;
end

if ~loaded
    error('Failed to load configuration files.');
end

% Make all relative paths absolute (relative to project root)
if isfield(config_gen, 'paths')
    if isfield(config_gen.paths, 'cleaned_eeg') && ~isempty(config_gen.paths.cleaned_eeg)
        % Check if path is relative (doesn't start with drive letter or /)
        if ~(length(config_gen.paths.cleaned_eeg) > 1 && config_gen.paths.cleaned_eeg(2) == ':') && ...
           ~(config_gen.paths.cleaned_eeg(1) == '/')
            config_gen.paths.cleaned_eeg = fullfile(projectRoot, config_gen.paths.cleaned_eeg);
        end
    end
    if isfield(config_gen.paths, 'output') && ~isempty(config_gen.paths.output)
        if ~(length(config_gen.paths.output) > 1 && config_gen.paths.output(2) == ':') && ...
           ~(config_gen.paths.output(1) == '/')
            config_gen.paths.output = fullfile(projectRoot, config_gen.paths.output);
        end
    end
    if isfield(config_gen.paths, 'eeg_data') && ~isempty(config_gen.paths.eeg_data)
        if ~(length(config_gen.paths.eeg_data) > 1 && config_gen.paths.eeg_data(2) == ':') && ...
           ~(config_gen.paths.eeg_data(1) == '/')
            config_gen.paths.eeg_data = fullfile(projectRoot, config_gen.paths.eeg_data);
        end
    end
    if isfield(config_gen.paths, 'events') && ~isempty(config_gen.paths.events)
        if ~(length(config_gen.paths.events) > 1 && config_gen.paths.events(2) == ':') && ...
           ~(config_gen.paths.events(1) == '/')
            config_gen.paths.events = fullfile(projectRoot, config_gen.paths.events);
        end
    end
end

fprintf('Paths configured:\n');
fprintf('  Project root: %s\n', projectRoot);
fprintf('  Cleaned EEG: %s\n', config_gen.paths.cleaned_eeg);
fprintf('  Output: %s\n', config_gen.paths.output);

% Extract settings
frequency_bands = config_feat.frequency_bands;
regions = normalize_regions(config_feat.regions);
num_workers = config_feat.parallel.num_workers;
parallel_enabled = config_feat.parallel.enabled;

% Paths
output_folder = fullfile(config_gen.paths.output, config_feat.output.folder);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
output_csv = fullfile(output_folder, config_feat.output.filename);

% Temp folder for intermediate results (enables resume)
temp_folder = fullfile(output_folder, 'temp');
if ~exist(temp_folder, 'dir'), mkdir(temp_folder); end
% Create log file for progress tracking
log_file = fullfile(output_folder, sprintf('extraction_log_%s.txt', datestr(now, 'yyyymmdd_HHMMSS')));
diary(log_file);
fprintf('=== EEG Feature Extraction Started at %s ===\n', datestr(now));
fprintf('Log file: %s\n', log_file);

% Initialize EEGLAB explicitly (required for batch mode)
fprintf('Initializing EEGLAB from: %s\n', config_feat.toolbox_paths.eeglab);
addpath(config_feat.toolbox_paths.eeglab);
addpath(fullfile(config_feat.toolbox_paths.eeglab, 'functions'));
addpath(fullfile(config_feat.toolbox_paths.eeglab, 'functions', 'popfunc'));
addpath(fullfile(config_feat.toolbox_paths.eeglab, 'functions', 'adminfunc'));
addpath(fullfile(config_feat.toolbox_paths.eeglab, 'functions', 'guifunc'));
addpath(fullfile(config_feat.toolbox_paths.eeglab, 'functions', 'sigprocfunc'));

% Initialize EEGLAB structure
fprintf('Calling eeg_getversion to initialize EEGLAB globals...\n');
try
    eeg_getversion;
    global ALLEEG EEG CURRENTSET ALLCOM;
    ALLEEG = [];
    EEG = [];
    CURRENTSET = 0;
    ALLCOM = {};
    fprintf('EEGLAB initialized successfully\n');
catch ME
    warning('EEGLAB initialization warning: %s', ME.message);
end

% Add other toolbox paths
if ~isempty(config_feat.toolbox_paths.entropy_hub) && exist(config_feat.toolbox_paths.entropy_hub, 'dir')
    addpath(genpath(config_feat.toolbox_paths.entropy_hub));
end
if ~isempty(config_feat.toolbox_paths.utils) && exist(config_feat.toolbox_paths.utils, 'dir')
    addpath(genpath(config_feat.toolbox_paths.utils));
end

% Get task conditions from config (all conditions with duration field)
task_conditions = fieldnames(config_cond.conditions);
% Filter to conditions that have duration (analysis-ready conditions)
task_conditions = task_conditions(cellfun(@(c) isfield(config_cond.conditions.(c), 'duration'), task_conditions));

% Get condition durations
condition_durations = containers.Map();
for i = 1:length(task_conditions)
    cond = task_conditions{i};
    condition_durations(cond) = config_cond.conditions.(cond).duration;
end

%% ========================================================================
%  BUILD COLUMN HEADER
%% ========================================================================

fprintf('Building output schema...\n');

cols = {'Participant', 'Condition'};

% Band power columns: Region_Band_Power
region_names = fieldnames(regions);
band_names = fieldnames(frequency_bands);

for ri = 1:length(region_names)
    for bi = 1:length(band_names)
        cols{end+1} = sprintf('%s_%s_Power', region_names{ri}, band_names{bi}); %#ok<SAGROW>
    end
end

% Ratio columns
if config_feat.features.ratios
    for i = 1:length(config_feat.ratios)
        cols{end+1} = config_feat.ratios{i}; %#ok<SAGROW>
    end
end

% Entropy columns: Region_EntropyType
if config_feat.features.entropy
    for ri = 1:length(region_names)
        for ei = 1:length(config_feat.entropy_metrics)
            cols{end+1} = sprintf('%s_%s', region_names{ri}, config_feat.entropy_metrics{ei}); %#ok<SAGROW>
        end
    end
end

fprintf('  Total columns: %d\n', length(cols));
fprintf('    Metadata: 2\n');
fprintf('    Band power: %d\n', length(region_names) * length(band_names));
if config_feat.features.ratios
    fprintf('    Ratios: %d\n', length(config_feat.ratios));
end
if config_feat.features.entropy
    fprintf('    Entropy: %d\n', length(region_names) * length(config_feat.entropy_metrics));
end

%% ========================================================================
%  WRITE HEADER
%% ========================================================================

fid = fopen(output_csv, 'w');
if fid == -1, error('Could not open %s for writing.', output_csv); end
fprintf(fid, '%s\n', strjoin(cols, ','));
fclose(fid);
fprintf('Header written -> %s\n', output_csv);

%% ========================================================================
%  PARALLEL PROCESSING SETUP
%% ========================================================================

% Force serial mode (parfor problematic in batch MATLAB)
parallel_enabled = false;

if parallel_enabled
    pool = gcp('nocreate');
    if ~isempty(pool), delete(pool); end
    parpool('local', num_workers);
    fprintf('Parallel pool started with %d workers\n', num_workers);
    
    % Sync paths to workers
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.eeglab '''));']);
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.entropy_hub '''));']);
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.utils '''));']);
    pctRunOnAll eeglab nogui;
end

%% ========================================================================
%  MAIN PROCESSING LOOP
%% ========================================================================

participant_numbers = [1]; % TEST: Just P01 for now

% Check for already-processed participants (resume capability)
processed = [];
for p = participant_numbers
    temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
    if isfile(temp_file)
        processed(end+1) = p; %#ok<SAGROW>
    end
end

if ~isempty(processed)
    fprintf('Found %d already-processed participants: %s\n', ...
        length(processed), mat2str(processed));
    fprintf('Skipping these. Delete temp files to reprocess.\n');
    participant_numbers = setdiff(participant_numbers, processed);
end

if isempty(participant_numbers)
    fprintf('All participants already processed. Merging results...\n');
else
    fprintf('Processing %d remaining participants...\n', length(participant_numbers));
end

% Copy config to local variables for parfor
freq_bands_local = frequency_bands;
regions_local = regions;
conds_local = task_conditions;
durations_local = condition_durations;
cols_local = cols;
config_cond_local = config_cond;

fprintf('\n=== PROCESSING %d PARTICIPANTS ===\n', length(participant_numbers));

% Track overall timing
tic;
start_time = datetime('now');
fprintf('Start time: %s\n', datestr(start_time));

for p = participant_numbers  % Changed from parfor to for (batch mode compatibility)
    try
        fprintf('[P%02d] Starting...\n', p);
        
        % Use the main cleaned set (has proper chanlocs after verification fix)
        base_name = sprintf('P%02d_cleaned.set', p);
        set_to_load = base_name;
        cleaned_set = fullfile(config_gen.paths.cleaned_eeg, set_to_load);
        if ~isfile(cleaned_set)
            warning('[P%02d] Cleaned .set file not found: %s', p, cleaned_set);
            continue;
        end

        % Load cleaned EEG data
        EEG = pop_loadset('filename', set_to_load, ...
                          'filepath', config_gen.paths.cleaned_eeg);
        
        if isempty(EEG.data)
            warning('[P%02d] Failed to load EEG data from .set file', p);
            continue;
        end
        
        % Remove channel 129 if present (trigger channel)
        if size(EEG.data, 1) >= 129
            EEG.data(129, :, :) = [];
            if length(EEG.chanlocs) >= 129
                EEG.chanlocs(129) = [];
            end
        end
        
        % Update dimensions
        [EEG.nbchan, EEG.pnts, EEG.trials] = size(EEG.data);
        if ndims(EEG.data) == 2, EEG.trials = 1; end
        EEG = eeg_checkset(EEG);
        
        % Load events from EEG struct (already embedded in .set file)
        if isempty(EEG.event)
            warning('[P%02d] No events found in EEG structure', p);
            continue;
        end
        
        % Convert EEG.event to table format with error handling
        try
            fprintf('[P%02d] Processing %d events\n', p, length(EEG.event));
            eventTypes = cell(length(EEG.event), 1);
            eventLatencies = zeros(length(EEG.event), 1);
            for ei = 1:length(EEG.event)
                try
                    eventTypes{ei} = EEG.event(ei).type;
                    eventLatencies(ei) = EEG.event(ei).latency;
                catch
                    eventTypes{ei} = 'unknown';
                    eventLatencies(ei) = 0;
                end
            end
            eventTable = table(eventTypes, eventLatencies, 'VariableNames', {'type', 'latency'});
            eventTable = sortrows(eventTable, 'latency');
            fprintf('[P%02d] Event table created with %d rows\n', p, height(eventTable));
        catch ME
            warning('[P%02d] Error converting events: %s', p, ME.message);
            fprintf('[P%02d] Stack: %s\n', p, ME.stack(1).name);
            continue;
        end
        
        % Get channel labels (or use indices if labels missing/empty)
        try
            fprintf('[P%02d] Getting channel labels...\n', p);
            if isempty(EEG.chanlocs) || length(EEG.chanlocs) == 0
                % No chanlocs at all - use numeric indices
                fprintf('[P%02d] No chanlocs structure - using numeric channel indices\n', p);
                chan_labels = arrayfun(@(x) num2str(x), 1:EEG.nbchan, 'UniformOutput', false);
            else
                % Chanlocs exist - try to get labels
                chan_labels = cell(1, EEG.nbchan);
                for ch = 1:EEG.nbchan
                    if ch <= length(EEG.chanlocs)
                        % Try both 'labels' and 'label' fields
                        if isfield(EEG.chanlocs, 'labels') && ~isempty(EEG.chanlocs(ch).labels)
                            chan_labels{ch} = EEG.chanlocs(ch).labels;
                        elseif isfield(EEG.chanlocs, 'label') && ~isempty(EEG.chanlocs(ch).label)
                            chan_labels{ch} = EEG.chanlocs(ch).label;
                        else
                            % Chanloc exists but no label - use index
                            chan_labels{ch} = num2str(ch);
                        end
                    else
                        % Channel beyond chanlocs array - use index
                        chan_labels{ch} = num2str(ch);
                    end
                end
            end
            fprintf('[P%02d] Found %d channel labels\n', p, length(chan_labels));
            if length(chan_labels) > 0
                fprintf('[P%02d] Sample labels: %s ... %s\n', p, chan_labels{1}, chan_labels{end});
            end
        catch ME
            warning('[P%02d] Error getting channel labels: %s', p, ME.message);
            % Fallback to numeric indices
            chan_labels = arrayfun(@(x) num2str(x), 1:EEG.nbchan, 'UniformOutput', false);
        end
        
        % Process each condition
        rows_this_participant = {};
        seen_conditions = {};
        
        fprintf('[P%02d] Starting condition loop...\n', p);
        
        for i = 1:height(eventTable)
            try
                raw_cond = eventTable.type{i};
                % Convert to string if numeric
                if isnumeric(raw_cond)
                    raw_cond = num2str(raw_cond);
                end
                cond = normalize_condition_label(raw_cond, config_cond_local);
                
                % Ensure cond is a string or char
                if ~ischar(cond) && ~isstring(cond)
                    continue;
                end
                
                % Skip if not in analysis set or already processed
                if isempty(cond) || ~ismember(cond, conds_local) || ismember(cond, seen_conditions)
                    continue;
                end
                seen_conditions{end+1} = cond; %#ok<SAGROW>
                
                % Get timing
                lat = round(eventTable.latency(i));
                
                % Check if condition exists in duration map
                if ~isKey(durations_local, cond)
                    warning('[P%02d] Condition "%s" not found in durations map', p, cond);
                    continue;
                end
                duration = durations_local(cond);
                
                t0 = max(1, lat);
                t1 = min(EEG.pnts, t0 + duration * EEG.srate - 1);
                
                if t1 <= t0
                    warning('[P%02d] Invalid time range for %s', p, cond);
                    continue;
                end
                
                % Extract window
                window_data = EEG.data(:, t0:t1);
                
                % Compute features (notch filtering already done in cleaning pipeline)
                feats = compute_features(window_data, EEG.srate, ...
                    freq_bands_local, regions_local, chan_labels, config_feat);
                
                % Build row
                row = {p, cond};
                
                % Band power
                for ri = 1:length(fieldnames(regions_local))
                    for bi = 1:length(fieldnames(freq_bands_local))
                        row{end+1} = feats.band_power{ri, bi}; %#ok<SAGROW>
                    end
                end
                
                % Ratios
                if config_feat.features.ratios
                    for ri = 1:length(config_feat.ratios)
                        ratio_name = config_feat.ratios{ri};
                        row{end+1} = feats.ratios.(ratio_name); %#ok<SAGROW>
                    end
                end
                
                % Entropy
                if config_feat.features.entropy
                    for ri = 1:length(fieldnames(regions_local))
                        for ei = 1:length(config_feat.entropy_metrics)
                            row{end+1} = feats.entropy{ri, ei}; %#ok<SAGROW>
                        end
                    end
                end
                
                rows_this_participant{end+1} = row; %#ok<SAGROW>
                
            catch ME
                warning('[P%02d] Error processing event %d (%s): %s', p, i, raw_cond, ME.message);
            end
        end
        
        % Write rows to temp file for this participant
        if ~isempty(rows_this_participant)
            temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
            fid = fopen(temp_file, 'w');
            for r = 1:length(rows_this_participant)
                fprintf(fid, '%s\n', strjoin(cellfun(@num2str, rows_this_participant{r}, 'UniformOutput', false), ','));
            end
            fclose(fid);
            fprintf('[P%02d] Wrote %d condition(s) to temp file\n', p, length(rows_this_participant));
        end
        
    catch ME
        warning('[P%02d] Error: %s', p, ME.message);
    end
end

% Report processing time
elapsed = toc;
fprintf('\nParallel processing completed in %.1f minutes (%.1f hours)\n', elapsed/60, elapsed/3600);

fprintf('\n=== MERGING RESULTS ===\n');

% Merge all temp files into final output
all_participants = 1:48; % Full participant list
fid_out = fopen(output_csv, 'a');
merged_count = 0;

for p = all_participants
    temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
    if isfile(temp_file)
        % Read and append temp file contents
        temp_data = fileread(temp_file);
        if ~isempty(temp_data)
            fprintf(fid_out, '%s', temp_data);
            merged_count = merged_count + 1;
        end
    else
        warning('[P%02d] Temp file not found - participant not processed', p);
    end
end

fclose(fid_out);

fprintf('\n=== EXTRACTION COMPLETE ===\n');
fprintf('Merged %d participant files\n', merged_count);
fprintf('Output: %s\n', output_csv);

% Clean up temp folder (optional - comment out to keep for debugging)
fprintf('\nCleaning up temp files...\n');
for p = all_participants
    temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
    if isfile(temp_file)
        delete(temp_file);
    end
end
rmdir(temp_folder);
fprintf('Cleanup complete.\n');

% Close log
fprintf('\n=== Extraction finished at %s ===\n', datestr(now));
diary off;

%% ========================================================================
%  HELPER FUNCTIONS
%% ========================================================================

function cond = normalize_condition_label(raw_label, config_cond)
    % Normalize event label to canonical condition name
    cond = '';
    if isempty(raw_label), return; end
    if isstring(raw_label), raw_label = char(raw_label); end
    raw_label = strtrim(raw_label);
    
    % Check each condition in config
    cond_names = fieldnames(config_cond.conditions);
    for i = 1:length(cond_names)
        cond_name = cond_names{i};
        cond_info = config_cond.conditions.(cond_name);
        
        % Check aliases
        if isfield(cond_info, 'aliases')
            for j = 1:length(cond_info.aliases)
                if contains(raw_label, cond_info.aliases{j}, 'IgnoreCase', true)
                    cond = cond_name;
                    return;
                end
            end
        end
        
        % Check exact match
        if strcmpi(raw_label, cond_name)
            cond = cond_name;
            return;
        end
    end
end

function regions_out = normalize_regions(regions_in)
    % Ensure region channel lists are cell arrays of strings (not numeric indices)
    regions_out = regions_in;
    region_names = fieldnames(regions_in);
    for i = 1:length(region_names)
        name = region_names{i};
        chans = regions_in.(name);
        if isnumeric(chans)
            % Convert numeric indices to strings for ismember comparisons
            regions_out.(name) = cellfun(@(x) num2str(x), num2cell(chans), 'UniformOutput', false);
        elseif iscell(chans)
            regions_out.(name) = chans;
        else
            regions_out.(name) = {chans};
        end
    end
end

function feats = compute_features(data, srate, frequency_bands, regions, chan_labels, config)
    % Compute all features for a data window
    % Note: Assumes data is already cleaned (notch filters applied in cleaning pipeline)
    
    feats = struct();
    
    % Compute PSD
    [psd, freqs] = calc_psd(data, srate);
    
    % Band power per region
    region_names = fieldnames(regions);
    band_names = fieldnames(frequency_bands);
    
    feats.band_power = cell(length(region_names), length(band_names));
    
    for ri = 1:length(region_names)
        % Get channel mask for this region
        region_chans = regions.(region_names{ri});
        chan_mask = ismember(chan_labels, region_chans);
        
        for bi = 1:length(band_names)
            band_range = frequency_bands.(band_names{bi});
            
            % Compute band power
            bp = compute_band_power(psd, freqs, chan_mask, band_range);
            feats.band_power{ri, bi} = bp;
        end
    end
    
    % Power ratios
    if config.features.ratios
        feats.ratios = compute_ratios(feats.band_power, region_names, band_names);
    end
    
    % Entropy
    if config.features.entropy
        feats.entropy = cell(length(region_names), length(config.entropy_metrics));
        for ri = 1:length(region_names)
            region_chans = regions.(region_names{ri});
            chan_mask = ismember(chan_labels, region_chans);
            region_data = data(chan_mask, :);
            
            if ~isempty(region_data)
                for ei = 1:length(config.entropy_metrics)
                    metric = config.entropy_metrics{ei};
                    if strcmp(metric, 'SampleEntropy')
                        feats.entropy{ri, ei} = compute_sample_entropy(region_data);
                    elseif strcmp(metric, 'SpectralEntropy')
                        feats.entropy{ri, ei} = compute_spectral_entropy(psd(chan_mask, :), freqs);
                    end
                end
            else
                for ei = 1:length(config.entropy_metrics)
                    feats.entropy{ri, ei} = NaN;
                end
            end
        end
    end
end

function [psd, freqs] = calc_psd(data, srate)
    % Calculate power spectral density using Welch's method
    window_length = min(2 * srate, size(data, 2));
    overlap = round(window_length / 2);
    nfft = 2^nextpow2(window_length);
    
    [psd, freqs] = pwelch(data', hamming(window_length), overlap, nfft, srate);
    psd = psd'; % Transpose to [channels × freqs]
end

function bp = compute_band_power(psd, freqs, chan_mask, band_range)
    % Compute band power for a region
    
    if ~any(chan_mask)
        bp = NaN;
        return;
    end
    
    % Frequency mask
    freq_mask = freqs >= band_range(1) & freqs <= band_range(2);
    
    if ~any(freq_mask)
        bp = NaN;
        return;
    end
    
    % Integrate power (trapezoidal)
    region_psd = psd(chan_mask, :);
    mean_psd = mean(region_psd, 1);
    bp = trapz(freqs(freq_mask), mean_psd(freq_mask));
    
    % Log transform
    bp = log10(max(bp, 1e-10));
end

function ratios = compute_ratios(band_power, region_names, band_names)
    % Compute power ratios from pre-computed band powers
    
    ratios = struct();
    
    % Helper to get band power by name safely (returns NaN if missing)
    function v = get_bp(region, band)
        ri = find(strcmp(region_names, region), 1);
        bi = find(strcmp(band_names, band), 1);
        if isempty(ri) || isempty(bi)
            v = NaN;
            return;
        end
        v = band_power{ri, bi};
        if isempty(v)
            v = NaN;
        end
    end
    
    % Frontal_Alpha_Asymmetry: log(RightFrontal_Alpha) - log(LeftFrontal_Alpha)
    left_alpha = get_bp('FrontalLeft', 'Alpha');
    right_alpha = get_bp('FrontalRight', 'Alpha');
    ratios.Frontal_Alpha_Asymmetry = right_alpha - left_alpha;
    
    % Alpha_Beta_Ratio: OverallFrontal_Alpha / OverallFrontal_Beta
    frontal_alpha = get_bp('OverallFrontal', 'Alpha');
    frontal_beta = get_bp('OverallFrontal', 'Beta');
    ratios.Alpha_Beta_Ratio = frontal_alpha - frontal_beta;
    
    % Theta_Beta_Ratio: FrontalMidline_Theta / FrontalMidline_Beta
    fm_theta = get_bp('FrontalMidline', 'Theta');
    fm_beta = get_bp('FrontalMidline', 'Beta');
    ratios.Theta_Beta_Ratio = fm_theta - fm_beta;
    
    % RightFrontal_Alpha
    ratios.RightFrontal_Alpha = right_alpha;
    
    % Theta_Alpha_Ratio: FrontalMidline_Theta / FrontalMidline_Alpha
    fm_alpha = get_bp('FrontalMidline', 'Alpha');
    ratios.Theta_Alpha_Ratio = fm_theta - fm_alpha;
end

function se = compute_sample_entropy(data)
    % Compute sample entropy for multi-channel data
    % Average entropy across channels
    
    try
        m = 2; % Embedding dimension
        r = 0.2 * std(data(:)); % Tolerance
        
        entropies = zeros(size(data, 1), 1);
        for ch = 1:size(data, 1)
            sig = data(ch, :);
            sig = (sig - mean(sig)) / std(sig); % Normalize
            entropies(ch) = SampEn(m, r, sig);
        end
        se = mean(entropies(~isnan(entropies) & ~isinf(entropies)));
        if isempty(se), se = NaN; end
    catch
        se = NaN;
    end
end

function spec_ent = compute_spectral_entropy(psd, freqs)
    % Compute spectral entropy
    % Average across channels
    
    try
        entropies = zeros(size(psd, 1), 1);
        for ch = 1:size(psd, 1)
            p = psd(ch, :);
            p = p / sum(p); % Normalize to probability
            p(p <= 0) = eps; % Avoid log(0)
            entropies(ch) = -sum(p .* log2(p));
        end
        spec_ent = mean(entropies(~isnan(entropies) & ~isinf(entropies)));
        if isempty(spec_ent), spec_ent = NaN; end
    catch
        spec_ent = NaN;
    end
end
