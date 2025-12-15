function extract_eeg_features_highstress(varargin)
% EXTRACT_EEG_FEATURES_HIGHSTRESS
% Extracts only band power features for high stress, high/low workload conditions
% using 10s windows with 50% overlap. Prepares data for ML best practices.
%
% Usage:
%   extract_eeg_features_highstress('participants', [1 2 3])
%   extract_eeg_features_highstress('output_folder', 'output/path')
%   ... (see extract_eeg_features for more)
%
% Only the following conditions are included:
%   - HighStress_HighCog1022_Task
%   - HighStress_HighCog2043_Task
%   - HighStress_LowCog_Task
%
% Only band power features are computed.
% Features are computed in 10s windows with 50% overlap.
% Output is formatted for ML (one row per window, with participant, condition, window_start, window_end).

    %% SETUP
    params = parse_inputs(varargin{:});
    [config_feat, config_cond, config_gen, output_folder, temp_folder, output_csv] = setup_environment(params);

    % Restrict to only band power features
    config_feat.features.band_power = true;
    config_feat.features.ratios = false;
    config_feat.features.entropy = false;

    % Only keep high stress, high/low workload conditions
    allowed_conditions = {'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', 'HighStress_LowCog_Task'};
    config_cond.conditions = rmfield(config_cond.conditions, setdiff(fieldnames(config_cond.conditions), allowed_conditions));

    % Build output schema: only pid, label, features
    header_cols = [{'pid','label'}];
    region_names = fieldnames(config_feat.regions);
    band_names = fieldnames(config_feat.frequency_bands);
    for ri = 1:length(region_names)
        for bi = 1:length(band_names)
            header_cols{end+1} = sprintf('%s_%s', region_names{ri}, band_names{bi});
        end
    end

    % Output file: avoid clash with main feature extraction
    output_csv_hs = fullfile(output_folder, 'eeg_features_highstress.csv');
    fid = fopen(output_csv_hs, 'w');
    if fid == -1, error('Could not create output file: %s', output_csv_hs); end
    fprintf(fid, '%s\n', strjoin(header_cols, ','));
    fclose(fid);

    participants_to_process = determine_work(params.participants, temp_folder, params.force_reprocess);
    setup_parallel_pool(config_feat);

    % Process participants
    for p = participants_to_process
        try
            eeglab nogui;
            cleaned_set = fullfile(config_gen.paths.cleaned_eeg, sprintf('P%02d_cleaned.set', p));
            if ~isfile(cleaned_set), warning('[P%02d] Cleaned .set file not found', p); continue; end
            EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), 'filepath', config_gen.paths.cleaned_eeg);
            if isempty(EEG.data), warning('[P%02d] No data', p); continue; end
            [~, ~, ~] = size(EEG.data);
            EEG = eeg_checkset(EEG);
            if ~isfield(EEG, 'event') || isempty(EEG.event), warning('[P%02d] No events', p); continue; end
            chan_labels = {EEG.chanlocs.labels};
            for i = 1:length(EEG.event)
                raw_cond = EEG.event(i).type;
                cond = normalize_condition_label(raw_cond, config_cond);
                if isempty(cond) || ~ismember(cond, allowed_conditions), continue; end
                % Label: 1 for HighStress_LowCog_Task, 2 for others
                if strcmp(cond, 'HighStress_LowCog_Task')
                    label = 1;
                else
                    label = 2;
                end
                lat = round(EEG.event(i).latency);
                duration = config_cond.conditions.(cond).duration;
                t0 = max(1, lat);
                t1 = min(EEG.pnts, t0 + duration * EEG.srate - 1);
                if t1 <= t0, warning('[P%02d] Invalid time range for %s', p, cond); continue; end
                window_len = 10 * EEG.srate;
                step = window_len / 2;
                for win_start = t0:step:(t1-window_len+1)
                    win_end = win_start + window_len - 1;
                    if win_end > t1, break; end
                    window_data = EEG.data(:, win_start:win_end);
                    feats = compute_features(window_data, EEG.srate, config_feat.frequency_bands, config_feat.regions, chan_labels, config_feat);
                    row = {p, label};
                    for ri = 1:length(region_names)
                        for bi = 1:length(band_names)
                            row{end+1} = feats.band_power{ri,bi};
                        end
                    end
                    % Write directly to output file (append)
                    fid = fopen(output_csv_hs, 'a');
                    if fid == -1, error('[P%02d] Could not write to output file', p); end
                    fprintf(fid, '%s\n', strjoin(cellfun(@num2str, row, 'UniformOutput', false), ','));
                    fclose(fid);
                end
            end
        catch ME
            fprintf('[P%02d] ERROR: %s\n', p, ME.message);
        end
    end
    fprintf('\n✓ High-stress band power feature extraction complete!\n');
end
