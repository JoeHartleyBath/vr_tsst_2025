function process_all_participants(participants, temp_folder, config_feat, config_cond, config_gen, header_cols)
% PROCESS_ALL_PARTICIPANTS Process multiple participants in parallel or serial
%
% Inputs:
%   participants  - Vector of participant numbers to process
%   temp_folder   - Path to temp folder for intermediate results
%   config_feat   - Feature extraction configuration
%   config_cond   - Conditions configuration
%   config_gen    - General configuration
%   header_cols   - Cell array of column names

    if isempty(participants)
        fprintf('No participants to process.\n\n');
        return;
    end
    
    fprintf('=== PROCESSING %d PARTICIPANTS ===\n', length(participants));
    fprintf('Start time: %s\n', datestr(now));
    
    % Prepare local copies for parfor
    freq_bands_local = config_feat.frequency_bands;
    regions_local = config_feat.regions;
    config_cond_local = config_cond;
    config_gen_local = config_gen;
    config_feat_local = config_feat;
    
    % Get task conditions
    task_conditions = fieldnames(config_cond.conditions);
    task_conditions = task_conditions(cellfun(@(c) config_cond.conditions.(c).include_in_analysis, ...
                                              task_conditions));
    
    % Get condition durations
    condition_durations = containers.Map();
    for i = 1:length(task_conditions)
        cond = task_conditions{i};
        condition_durations(cond) = config_cond.conditions.(cond).duration;
    end
    
    % Track timing
    tic;
    
    % Process participants
    if config_feat.parallel.enabled
        parfor p = participants
            process_single_participant(p, temp_folder, freq_bands_local, regions_local, ...
                                      config_feat_local, config_cond_local, config_gen_local, ...
                                      task_conditions, condition_durations);
        end
    else
        for p = participants
            process_single_participant(p, temp_folder, freq_bands_local, regions_local, ...
                                      config_feat_local, config_cond_local, config_gen_local, ...
                                      task_conditions, condition_durations);
        end
    end
    
    % Report completion
    elapsed = toc;
    fprintf('\n=== PROCESSING COMPLETE ===\n');
    fprintf('Processed %d participants in %.1f minutes (%.1f hours)\n', ...
            length(participants), elapsed/60, elapsed/3600);
    fprintf('Completion time: %s\n\n', datestr(now));
end


function process_single_participant(p, temp_folder, freq_bands, regions, ...
                                   config_feat, config_cond, config_gen, ...
                                   task_conditions, condition_durations)
% PROCESS_SINGLE_PARTICIPANT Extract features for one participant
%
% This function is called by parfor or regular for loop.
% Writes results to temp file.

    try
        % Initialize EEGLAB on worker
        eeglab nogui;
        
        fprintf('[P%02d] Starting...\n', p);
        
        % ===== Load data =====
        % Load cleaned EEG .set file (contains both data and metadata)
        cleaned_set = fullfile(config_gen.paths.cleaned_eeg, sprintf('P%02d_cleaned.set', p));
        if ~isfile(cleaned_set)
            warning('[P%02d] Cleaned .set file not found: %s', p, cleaned_set);
            return;
        end
        
        EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), ...
                          'filepath', config_gen.paths.cleaned_eeg);
        
        % Validate loaded data
        if isempty(EEG.data)
            warning('[P%02d] Loaded .set file has no data', p);
            return;
        end
        
        % Update dimensions (trigger channel already removed by cleaning pipeline)
        [EEG.nbchan, EEG.pnts, EEG.trials] = size(EEG.data);
        if ndims(EEG.data) == 2
            EEG.trials = 1;
        end
        EEG = eeg_checkset(EEG);
        
        % Check events exist in .set file
        if ~isfield(EEG, 'event') || isempty(EEG.event)
            warning('[P%02d] No events found in .set file', p);
            return;
        end
        
        fprintf('[P%02d] Found %d events\n', p, length(EEG.event));
        
        % Get channel labels
        chan_labels = {EEG.chanlocs.labels};
        
        % ===== Extract features for each condition =====
        rows_this_participant = {};
        seen_conditions = {};
        
        for i = 1:length(EEG.event)
            raw_cond = EEG.event(i).type;
            cond = normalize_condition_label(raw_cond, config_cond);
            
            % Skip if not in analysis set or already processed
            if isempty(cond) || ~ismember(cond, task_conditions) || ismember(cond, seen_conditions)
                continue;
            end
            seen_conditions{end+1} = cond; %#ok<AGROW>
            
            % Get timing
            lat = round(EEG.event(i).latency);
            duration = condition_durations(cond);
            t0 = max(1, lat);
            t1 = min(EEG.pnts, t0 + duration * EEG.srate - 1);
            
            if t1 <= t0
                warning('[P%02d] Invalid time range for %s', p, cond);
                continue;
            end
            
            % Extract window
            window_data = EEG.data(:, t0:t1);
            
            % Compute features
            feats = compute_features(window_data, EEG.srate, ...
                freq_bands, regions, chan_labels, config_feat);
            
            % Build row
            row = format_feature_row(p, cond, feats, freq_bands, regions, config_feat);
            rows_this_participant{end+1} = row; %#ok<AGROW>
        end
        
        % ===== Write to temp file =====
        if ~isempty(rows_this_participant)
            temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
            fid = fopen(temp_file, 'w');
            if fid == -1
                error('[P%02d] Could not write to temp file: %s', p, temp_file);
            end
            
            for r = 1:length(rows_this_participant)
                fprintf(fid, '%s\n', strjoin(cellfun(@num2str, rows_this_participant{r}, ...
                                                     'UniformOutput', false), ','));
            end
            fclose(fid);
            
            fprintf('[P%02d] ✓ Wrote %d condition(s)\n', p, length(rows_this_participant));
        else
            fprintf('[P%02d] ⚠ No conditions extracted\n', p);
        end
        
    catch ME
        fprintf('[P%02d] ✗ ERROR: %s\n', p, ME.message);
        fprintf('         %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    end
end


function row = format_feature_row(p, cond, feats, freq_bands, regions, config)
% FORMAT_FEATURE_ROW Convert features struct to CSV row
    
    row = {p, cond};
    
    region_names = fieldnames(regions);
    band_names = fieldnames(freq_bands);
    
    % Band power
    if config.features.band_power
        for ri = 1:length(region_names)
            for bi = 1:length(band_names)
                row{end+1} = feats.band_power{ri, bi}; %#ok<AGROW>
            end
        end
    end
    
    % Ratios
    if config.features.ratios
        for ri = 1:length(config.ratios)
            ratio_name = config.ratios{ri};
            row{end+1} = feats.ratios.(ratio_name); %#ok<AGROW>
        end
    end
    
    % Entropy
    if config.features.entropy
        for ri = 1:length(region_names)
            for ei = 1:length(config.entropy_metrics)
                row{end+1} = feats.entropy{ri, ei}; %#ok<AGROW>
            end
        end
    end
end
