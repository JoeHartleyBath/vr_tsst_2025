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
%  CONFIGURATION LOADING
%% ========================================================================

% Load configurations
config_feat = yaml.loadFile('config/eeg_feature_extraction.yaml');
config_cond = yaml.loadFile('config/conditions.yaml');
config_gen = yaml.loadFile('config/general.yaml');

% Extract settings
frequency_bands = config_feat.frequency_bands;
regions = config_feat.regions;
num_workers = config_feat.parallel.num_workers;
parallel_enabled = config_feat.parallel.enabled;

% Paths
output_folder = fullfile(config_gen.paths.output, config_feat.output.folder);
if ~exist(output_folder, 'dir'), mkdir(output_folder); end
output_csv = fullfile(output_folder, config_feat.output.filename);

% Add toolbox paths
addpath(genpath(config_feat.toolbox_paths.eeglab));
addpath(genpath(config_feat.toolbox_paths.entropy_hub));
addpath(genpath(config_feat.toolbox_paths.utils));

% Get task conditions from config
task_conditions = fieldnames(config_cond.conditions);
task_conditions = task_conditions(cellfun(@(c) config_cond.conditions.(c).include_in_analysis, task_conditions));

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

participant_numbers = 1:48; % Adjust based on your dataset

% Copy config to local variables for parfor
freq_bands_local = frequency_bands;
regions_local = regions;
conds_local = task_conditions;
durations_local = condition_durations;
cols_local = cols;
config_cond_local = config_cond;

fprintf('\n=== PROCESSING %d PARTICIPANTS ===\n', length(participant_numbers));

parfor p = participant_numbers
    try
        eeglab nogui; % Initialize EEGLAB on worker
        
        fprintf('[P%02d] Starting...\n', p);
        
        % Load cleaned EEG data
        cleaned_file = fullfile(config_gen.paths.cleaned_eeg, sprintf('P%d_cleaned.mat', p));
        if ~isfile(cleaned_file)
            warning('[P%02d] Cleaned file not found: %s', p, cleaned_file);
            continue;
        end
        
        % Load filtered .set for metadata
        filtered_set = fullfile(config_gen.paths.eeg_data, 'filtered', sprintf('P%02d_filtered.set', p));
        if ~isfile(filtered_set)
            warning('[P%02d] Filtered .set not found: %s', p, filtered_set);
            continue;
        end
        
        EEG = pop_loadset('filename', sprintf('P%02d_filtered.set', p), ...
                          'filepath', fullfile(config_gen.paths.eeg_data, 'filtered'));
        
        % Load cleaned data matrix
        cleaned_data = load(cleaned_file);
        field_name = fieldnames(cleaned_data);
        EEG.data = cleaned_data.(field_name{1});
        
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
        
        % Load events
        events_file = fullfile(config_gen.paths.events, sprintf('P%02d_events.csv', p));
        if ~isfile(events_file)
            warning('[P%02d] Events file not found: %s', p, events_file);
            continue;
        end
        eventTable = readtable(events_file);
        eventTable = sortrows(eventTable, 'latency');
        
        % Get channel labels
        chan_labels = {EEG.chanlocs.labels};
        
        % Process each condition
        rows_this_participant = {};
        seen_conditions = {};
        
        for i = 1:height(eventTable)
            raw_cond = eventTable.type{i};
            cond = normalize_condition_label(raw_cond, config_cond_local);
            
            % Skip if not in analysis set or already processed
            if isempty(cond) || ~ismember(cond, conds_local) || ismember(cond, seen_conditions)
                continue;
            end
            seen_conditions{end+1} = cond; %#ok<SAGROW>
            
            % Get timing
            lat = round(eventTable.latency(i));
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
        end
        
        % Write rows for this participant
        if ~isempty(rows_this_participant)
            fid = fopen(output_csv, 'a');
            for r = 1:length(rows_this_participant)
                fprintf(fid, '%s\n', strjoin(cellfun(@num2str, rows_this_participant{r}, 'UniformOutput', false), ','));
            end
            fclose(fid);
            fprintf('[P%02d] Wrote %d condition(s)\n', p, length(rows_this_participant));
        end
        
    catch ME
        warning('[P%02d] Error: %s', p, ME.message);
    end
end

fprintf('\n=== EXTRACTION COMPLETE ===\n');
fprintf('Output: %s\n', output_csv);

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
    
    % Helper to get band power by name
    get_bp = @(region, band) band_power{strcmp(region_names, region), strcmp(band_names, band)};
    
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
