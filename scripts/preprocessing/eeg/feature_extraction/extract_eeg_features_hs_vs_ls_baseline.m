function extract_eeg_features_rolling_windows(varargin)
% EXTRACT_EEG_FEATURES_ROLLING_WINDOWS
% Extracts band power features for all allowed conditions using 4s windows with 80% overlap, and applies precondition baseline subtraction.
%
% Output: output/aggregated/eeg_features_rolling_windows.csv

    %% SETUP
    params = parse_inputs(varargin{:});
    [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = setup_environment(params);
    config_gen.paths.cleaned_eeg = 'C:/vr_tsst_2025/output/cleaned_eeg';

    config_feat.features.band_power = true;
    config_feat.features.ratios = false;
    config_feat.features.entropy = false;

    allowed_conditions = {'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', 'HighStress_LowCog_Task', ...
                         'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', 'LowStress_LowCog_Task'};
    config_cond.conditions = rmfield(config_cond.conditions, setdiff(fieldnames(config_cond.conditions), allowed_conditions));

    % Build output schema: pid, label, window_idx, features
    header_cols = [{'pid','event_label','window_idx'}];
    region_names = fieldnames(config_feat.regions);
    band_names = fieldnames(config_feat.frequency_bands);
    for ri = 1:length(region_names)
        for bi = 1:length(band_names)
            header_cols{end+1} = sprintf('%s_%s', region_names{ri}, band_names{bi});
        end
    end

    output_csv_hs = fullfile('output', 'aggregated', 'eeg_features_rolling_windows.csv');
    fid = fopen(output_csv_hs, 'w');
    if fid == -1, error('Could not create output file: %s', output_csv_hs); end
    fprintf(fid, '%s\n', strjoin(header_cols, ','));
    fclose(fid);

    participants_to_process = determine_work(params.participants, temp_folder, params.force_reprocess);
    setup_parallel_pool(config_feat);

    for p = participants_to_process
        try
            eeglab nogui;
            cleaned_set = fullfile(config_gen.paths.cleaned_eeg, sprintf('P%02d_cleaned.set', p));
            fprintf('[DEBUG] Checking for cleaned set file: %s\n', cleaned_set);
            if ~isfile(cleaned_set)
                warning('[P%02d] Cleaned .set file not found: %s', p, cleaned_set);
                continue;
            end
            EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), 'filepath', config_gen.paths.cleaned_eeg);
            if isempty(EEG.data), warning('[P%02d] No data', p); continue; end
            EEG = eeg_checkset(EEG);
            if ~isfield(EEG, 'event') || isempty(EEG.event), warning('[P%02d] No events', p); continue; end
            chan_labels = {EEG.chanlocs.labels};

            % Identify all baseline (Forest) scenes for this participant
            baseline_events = {};
            for i = 1:length(EEG.event)
                raw_cond = EEG.event(i).type;
                cond = normalize_condition_label(raw_cond, config_cond);
                if contains(cond, 'Forest', 'IgnoreCase', true)
                    baseline_events{end+1} = struct('idx', i, 'cond', cond, 'latency', EEG.event(i).latency);
                end
            end

            for i = 1:length(EEG.event)
                raw_cond = EEG.event(i).type;
                cond = normalize_condition_label(raw_cond, config_cond);
                if isempty(cond) || ~ismember(cond, allowed_conditions), continue; end
                % Use event label (condition name) instead of numeric label
                event_label = cond;
                lat = round(EEG.event(i).latency);
                duration = config_cond.conditions.(cond).duration;
                t0 = max(1, lat);
                t1 = min(EEG.pnts, t0 + duration * EEG.srate - 1);
                if t1 <= t0, warning('[P%02d] Invalid time range for %s', p, cond); continue; end
                window_len = 15 * EEG.srate;
                step = round(window_len * 0.5); % 80% overlap

                % --- Find the correct baseline (Forest) event for this condition ---
                baseline_idx = -1;
                baseline_latency = -1;
                for b = 1:length(baseline_events)
                    if baseline_events{b}.latency < lat
                        if baseline_events{b}.latency > baseline_latency
                            baseline_latency = baseline_events{b}.latency;
                            baseline_idx = b;
                        end
                    end
                end
                if baseline_idx == -1
                    warning('[P%02d] No baseline found for %s', p, cond); continue;
                end
                base_lat = round(baseline_events{baseline_idx}.latency);
                base_dur = 60; % Assume 60s for baseline (adjust if needed)
                base_t0 = max(1, base_lat);
                base_t1 = min(EEG.pnts, base_t0 + base_dur * EEG.srate - 1);
                if base_t1 <= base_t0, warning('[P%02d] Invalid baseline time range', p); continue; end

                % Compute baseline features (average over all windows in baseline)
                base_feats_all = [];
                for base_win_start = base_t0:step:(base_t1-window_len+1)
                    base_win_end = base_win_start + window_len - 1;
                    if base_win_end > base_t1, break; end
                    base_window_data = EEG.data(:, base_win_start:base_win_end);
                    base_feats = compute_features(base_window_data, EEG.srate, config_feat.frequency_bands, config_feat.regions, chan_labels, config_feat);
                    base_vec = [];
                    for ri = 1:length(region_names)
                        for bi = 1:length(band_names)
                            base_vec(end+1) = base_feats.band_power{ri,bi};
                        end
                    end
                    base_feats_all = [base_feats_all; base_vec];
                end
                if isempty(base_feats_all)
                    warning('[P%02d] No baseline windows for %s', p, cond); continue;
                end
                base_mean = mean(base_feats_all, 1);

                % --- Condition windows ---
                win_idx = 1;
                for win_start = t0:step:(t1-window_len+1)
                    win_end = win_start + window_len - 1;
                    if win_end > t1, break; end
                    window_data = EEG.data(:, win_start:win_end);
                    feats = compute_features(window_data, EEG.srate, config_feat.frequency_bands, config_feat.regions, chan_labels, config_feat);
                    row = {p, event_label, win_idx};
                    cond_vec = [];
                    for ri = 1:length(region_names)
                        for bi = 1:length(band_names)
                            cond_vec(end+1) = feats.band_power{ri,bi};
                        end
                    end
                    % Subtract baseline mean from condition features
                    corrected_vec = cond_vec - base_mean;
                    for v = 1:length(corrected_vec)
                        row{end+1} = corrected_vec(v);
                    end
                    fid = fopen(output_csv_hs, 'a');
                    if fid == -1, error('[P%02d] Could not write to output file', p); end
                    fprintf(fid, '%s\n', strjoin(cellfun(@num2str, row, 'UniformOutput', false), ','));
                    fclose(fid);
                    win_idx = win_idx + 1;
                end
            end
        catch ME
            fprintf('[P%02d] ERROR: %s\n', p, ME.message);
        end
    end
    fprintf('\n✓ HS vs LS band power feature extraction with baseline correction complete!\n');
end
