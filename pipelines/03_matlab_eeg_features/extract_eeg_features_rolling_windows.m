function extract_eeg_features_rolling_windows(varargin)
% EXTRACT_EEG_FEATURES_ROLLING_WINDOWS
% Extracts band power features using 10s rolling windows with 50% overlap.
% Outputs raw features (no baseline correction) for downstream processing.
%
% Usage:
%   extract_eeg_features_rolling_windows()                          % All participants
%   extract_eeg_features_rolling_windows('participants', [1 5 10])  % Specific participants
%   extract_eeg_features_rolling_windows('force_reprocess', true)   % Ignore cache
%   extract_eeg_features_rolling_windows('num_workers', 8)          % Parallel with 8 workers
%
% Output: output/aggregated/eeg_features_rolling_windows.csv

    %% ====================================================================
    %  SETUP
    %% ====================================================================
    
    params = parse_inputs(varargin{:});
    [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = setup_environment(params);
    
    % Override output path for rolling windows
    output_csv = fullfile('output', 'aggregated', 'eeg_features_rolling_windows.csv');
    
    % Override temp folder for rolling windows
    temp_folder = fullfile(output_folder, 'temp_rolling');
    if ~exist(temp_folder, 'dir')
        mkdir(temp_folder);
    end

    % Ensure cleaned_eeg path is set
    if ~isfield(config_gen.paths, 'cleaned_eeg') || isempty(config_gen.paths.cleaned_eeg)
        config_gen.paths.cleaned_eeg = 'C:/vr_tsst_2025/output/cleaned_eeg';
    end

    % Only extract band power features (no ratios, no entropy)
    config_feat.features.band_power = true;
    config_feat.features.ratios = false;
    config_feat.features.entropy = false;

    % Filter to task conditions only
    allowed_conditions = {'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', 'HighStress_LowCog_Task', ...
                         'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', 'LowStress_LowCog_Task'};
    all_conditions = fieldnames(config_cond.conditions);
    conditions_to_remove = setdiff(all_conditions, allowed_conditions);
    config_cond.conditions = rmfield(config_cond.conditions, conditions_to_remove);
    
    %% ====================================================================
    %  BUILD OUTPUT SCHEMA
    %% ====================================================================
    
    % Rolling window schema: pid, event_label, window_idx, features
    header_cols = [{'pid','event_label','window_idx'}];
    region_names = fieldnames(config_feat.regions);
    band_names = fieldnames(config_feat.frequency_bands);
    for ri = 1:length(region_names)
        for bi = 1:length(band_names)
            header_cols{end+1} = sprintf('%s_%s', region_names{ri}, band_names{bi});
        end
    end

    % Write header to output CSV
    fid = fopen(output_csv, 'w');
    if fid == -1
        error('Could not create output file: %s', output_csv);
    end
    fprintf(fid, '%s\n', strjoin(header_cols, ','));
    fclose(fid);
    fprintf('Header written to: %s\n\n', output_csv);
    
    %% ====================================================================
    %  DETERMINE WORK (RESUME LOGIC)
    %% ====================================================================
    
    participants_to_process = determine_work(params.participants, temp_folder, params.force_reprocess);
    
    %% ====================================================================
    %  PARALLEL PROCESSING SETUP
    %% ====================================================================
    
    setup_parallel_pool(config_feat);
    
    %% ====================================================================
    %  PROCESS PARTICIPANTS
    %% ====================================================================
    
    fprintf('=== PROCESSING %d PARTICIPANTS ===\n', length(participants_to_process));
    fprintf('Start time: %s\n', datestr(now));
    fprintf('Window size: 10s, Overlap: 50%% (5s stride)\n\n');
    
    % Prepare local copies for parfor
    freq_bands_local = config_feat.frequency_bands;
    regions_local = config_feat.regions;
    config_cond_local = config_cond;
    config_gen_local = config_gen;
    config_feat_local = config_feat;
    task_conditions = allowed_conditions;
    
    % Track timing
    tic;
    
    % Process participants (parallel or serial)
    if config_feat.parallel.enabled
        parfor p = participants_to_process
            process_single_participant_rolling(p, temp_folder, freq_bands_local, regions_local, ...
                                              config_feat_local, config_cond_local, config_gen_local, ...
                                              task_conditions, header_cols);
        end
    else
        for p = participants_to_process
            process_single_participant_rolling(p, temp_folder, freq_bands_local, regions_local, ...
                                              config_feat_local, config_cond_local, config_gen_local, ...
                                              task_conditions, header_cols);
        end
    end
    
    % Report completion
    elapsed = toc;
    fprintf('\n=== PROCESSING COMPLETE ===\n');
    fprintf('Processed %d participants in %.1f minutes (%.1f hours)\n', ...
            length(participants_to_process), elapsed/60, elapsed/3600);
    fprintf('Completion time: %s\n\n', datestr(now));
    
    %% ====================================================================
    %  MERGE RESULTS
    %% ====================================================================
    
    merged_count = merge_temp_files(params.participants, temp_folder, output_csv);
    
    %% ====================================================================
    %  CLEANUP
    %% ====================================================================
    
    keep_temp = false; % Set to true for debugging
    cleanup_temp_files(params.participants, temp_folder, keep_temp);
    
    %% ====================================================================
    %  SUMMARY
    %% ====================================================================
    
    fprintf('\n=== EXTRACTION COMPLETE ===\n');
    fprintf('  Processed: %d participants\n', length(participants_to_process));
    fprintf('  Merged: %d participants\n', merged_count);
    fprintf('  Output: %s\n', output_csv);
    fprintf('  Completion time: %s\n', datestr(now));
    
    fprintf('\n✓ Rolling window feature extraction complete!\n');
    fprintf('  Features: Raw band power (no baseline correction)\n');
    fprintf('  Baseline adjustment: Applied in downstream preprocessing\n');
end


function process_single_participant_rolling(p, temp_folder, freq_bands, regions, ...
                                           config_feat, config_cond, config_gen, ...
                                           task_conditions, header_cols)
% PROCESS_SINGLE_PARTICIPANT_ROLLING Extract rolling window features for one participant
%
% This function is called by parfor or regular for loop.
% Writes results to temp file for later merging.

    try
        % Initialize EEGLAB on worker
        eeglab nogui;
        
        fprintf('[P%02d] Starting rolling window extraction...\n', p);
        
        % ===== Load data =====
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
        
        EEG = eeg_checkset(EEG);
        
        if ~isfield(EEG, 'event') || isempty(EEG.event)
            warning('[P%02d] No events found', p);
            return;
        end
        
        chan_labels = {EEG.chanlocs.labels};
        region_names = fieldnames(regions);
        band_names = fieldnames(freq_bands);
        
        % ===== Extract features with rolling windows =====
        results = {};
        total_windows = 0;
        
        for i = 1:length(EEG.event)
            raw_cond = EEG.event(i).type;
            cond = normalize_condition_label(raw_cond, config_cond);
            
            % Skip non-task conditions
            if isempty(cond) || ~ismember(cond, task_conditions)
                continue;
            end
            
            % Get event timing
            lat = round(EEG.event(i).latency);
            duration = config_cond.conditions.(cond).duration;
            t0 = max(1, lat);
            t1 = min(EEG.pnts, t0 + duration * EEG.srate - 1);
            
            if t1 <= t0
                warning('[P%02d] Invalid time range for %s', p, cond);
                continue;
            end
            
            % Define rolling window parameters
            window_len = 10 * EEG.srate;  % 10 seconds
            step = round(window_len * 0.5); % 50% overlap (5s stride)
            
            % Extract features from each window
            win_idx = 1;
            for win_start = t0:step:(t1-window_len+1)
                win_end = win_start + window_len - 1;
                if win_end > t1
                    break;
                end
                
                % Extract window data
                window_data = EEG.data(:, win_start:win_end);
                
                % Compute features for this window
                feats = compute_features(window_data, EEG.srate, freq_bands, regions, chan_labels, config_feat);
                
                % Build result row: [pid, event_label, window_idx, features...]
                row = cell(1, length(header_cols));
                row{1} = p;
                row{2} = cond;
                row{3} = win_idx;
                
                % Add band power features
                col_idx = 4;
                for ri = 1:length(region_names)
                    for bi = 1:length(band_names)
                        row{col_idx} = feats.band_power{ri,bi};
                        col_idx = col_idx + 1;
                    end
                end
                
                results{end+1} = row;
                win_idx = win_idx + 1;
                total_windows = total_windows + 1;
            end
        end
        
        % ===== Save to temp file =====
        if ~isempty(results)
            temp_file = fullfile(temp_folder, sprintf('P%02d.mat', p));
            save(temp_file, 'results', 'header_cols');
            fprintf('[P%02d] Saved %d windows to temp file\n', p, total_windows);
        else
            warning('[P%02d] No windows extracted', p);
        end
        
    catch ME
        warning('[P%02d] ERROR: %s\n%s', p, ME.message, getReport(ME));
    end
end
