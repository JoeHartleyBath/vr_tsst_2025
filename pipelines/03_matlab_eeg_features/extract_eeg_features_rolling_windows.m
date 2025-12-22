function extract_eeg_features_rolling_windows(varargin)
% EXTRACT_EEG_FEATURES_ROLLING_WINDOWS
% Extracts band power features for all allowed conditions using 10s windows with 50% overlap, and applies precondition baseline subtraction.
%
% Output: output/aggregated/eeg_features_rolling_windows.csv

    %% SETUP
    params = parse_inputs(varargin{:});
    [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = setup_environment(params);
    config_gen.paths.cleaned_eeg = 'C:/vr_tsst_2025/output/cleaned_eeg';

    config_feat.features.band_power = true;
    config_feat.features.ratios = false;
    config_feat.features.entropy = false;

    % Include both task conditions AND baseline conditions (Forest1-4)
    allowed_conditions = {'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', 'HighStress_LowCog_Task', ...
                         'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', 'LowStress_LowCog_Task', ...
                         'Forest1', 'Forest2', 'Forest3', 'Forest4'};
    config_cond.conditions = rmfield(config_cond.conditions, setdiff(fieldnames(config_cond.conditions), allowed_conditions));

    % Build output schema: pid, event_label, window_idx, window_start, window_end, features
    header_cols = [{'pid','event_label','window_idx','window_start','window_end'}];
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
                window_len = 10 * EEG.srate;
                step = round(window_len * 0.5); % 50% overlap

                % --- Extract features from condition windows (raw features, no baseline correction) ---
                win_idx = 1;
                for win_start = t0:step:(t1-window_len+1)
                    win_end = win_start + window_len - 1;
                    if win_end > t1, break; end
                    % Calculate timestamps in seconds (matching physio time reference)
                    window_start_sec = (win_start - 1) / EEG.srate;
                    window_end_sec = win_end / EEG.srate;
                    window_data = EEG.data(:, win_start:win_end);
                    feats = compute_features(window_data, EEG.srate, config_feat.frequency_bands, config_feat.regions, chan_labels, config_feat);
                    row = {p, event_label, win_idx, window_start_sec, window_end_sec};
                    cond_vec = [];
                    for ri = 1:length(region_names)
                        for bi = 1:length(band_names)
                            cond_vec(end+1) = feats.band_power{ri,bi};
                        end
                    end
                    % Output raw features (baseline correction applied in downstream preprocessing)
                    for v = 1:length(cond_vec)
                        row{end+1} = cond_vec(v);
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
    fprintf('\n✓ Rolling window band power feature extraction complete! (Raw features - baseline correction applied in downstream preprocessing)\n');
end
