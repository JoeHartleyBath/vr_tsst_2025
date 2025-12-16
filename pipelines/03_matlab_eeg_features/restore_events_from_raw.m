function restore_events_from_raw(participants)
% RESTORE_EVENTS_FROM_RAW Reattach raw onset events to cleaned EEG after ASR cuts
% Keeps AMICA/ICA results; uses boundary durations to map raw latencies into
% cleaned timeline. Drops events that fall inside removed segments.
%
% Usage:
%   restore_events_from_raw([])            %% auto-detect participants in cleaned dir
%   restore_events_from_raw([5 26 37])     %% run specific IDs
%
% Paths are read from config/general.yaml (cleaned_eeg, raw sets in output/sets).

    % Load config for paths
    config = yaml.ReadYaml(fullfile('config', 'general.yaml'));
    cleaned_dir = config.paths.cleaned_eeg;
    raw_dir = fullfile('output', 'sets');

    if nargin < 1 || isempty(participants)
        % Discover participants from cleaned files
        files = dir(fullfile(cleaned_dir, 'P??_cleaned.set'));
        participants = [];
        for k = 1:numel(files)
            p = sscanf(files(k).name, 'P%02d_cleaned.set');
            if ~isempty(p)
                participants(end+1) = p; %#ok<AGROW>
            end
        end
        participants = sort(participants);
    end

    fprintf('Restoring events for %d participants...\n', numel(participants));

    for idx = 1:numel(participants)
        p = participants(idx);
        raw_path = fullfile(raw_dir, sprintf('P%02d.set', p));
        clean_path = fullfile(cleaned_dir, sprintf('P%02d_cleaned.set', p));

        if ~isfile(raw_path)
            fprintf('[P%02d] Raw set missing, skipping: %s\n', p, raw_path);
            continue;
        end
        if ~isfile(clean_path)
            fprintf('[P%02d] Cleaned set missing, skipping: %s\n', p, clean_path);
            continue;
        end

        fprintf('[P%02d] Loading raw and cleaned sets...\n', p);
        EEG_raw = pop_loadset('filename', sprintf('P%02d.set', p), 'filepath', raw_dir);
        EEG_clean = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), 'filepath', cleaned_dir);

        % Extract boundary events with durations
        boundary_idx = strcmpi({EEG_clean.event.type}, 'boundary');
        boundaries = EEG_clean.event(boundary_idx);

        % Build removed intervals in raw sample space using boundary durations
        intervals = [];
        durations = [];
        cumulative_removed = 0;

        if ~isempty(boundaries)
            % Sort boundaries by latency just in case
            [~, order] = sort([boundaries.latency]);
            boundaries = boundaries(order);

            for b = 1:numel(boundaries)
                dur = 0;
                if isfield(boundaries(b), 'duration') && ~isempty(boundaries(b).duration)
                    dur = double(boundaries(b).duration);
                end
                raw_start = double(boundaries(b).latency) + cumulative_removed;
                raw_end = raw_start + dur - 1;
                if dur > 0
                    intervals(end+1, :) = [raw_start, raw_end]; %#ok<AGROW>
                    durations(end+1) = dur; %#ok<AGROW>
                    cumulative_removed = cumulative_removed + dur;
                end
            end
        end

        % Restore events from raw, skipping those inside removed spans
        restored = EEG_raw.event;
        keep_mask = true(1, numel(restored));
        for e = 1:numel(restored)
            raw_lat = double(restored(e).latency);
            % Inside removed interval?
            if ~isempty(intervals)
                in_removed = any(raw_lat >= intervals(:,1) & raw_lat <= intervals(:,2));
                if in_removed
                    keep_mask(e) = false;
                    continue;
                end
            end
            % Shift by samples removed before this latency
            removed_before = 0;
            if ~isempty(intervals)
                removed_before = sum(durations(raw_lat > intervals(:,2)));
            end
            restored(e).latency = raw_lat - removed_before;
        end
        restored = restored(keep_mask);

        % Combine restored events with boundary events and sort
        combined = [restored, boundaries];
        if isempty(combined)
            fprintf('[P%02d] No events to write.\n', p);
            continue;
        end
        [~, order_all] = sort([combined.latency]);
        combined = combined(order_all);

        EEG_clean.event = combined;
        EEG_clean = eeg_checkset(EEG_clean);

        out_name = sprintf('P%02d_cleaned_events.set', p);
        pop_saveset(EEG_clean, 'filename', out_name, 'filepath', cleaned_dir);
        fprintf('[P%02d] Restored %d events (kept %d boundaries) -> %s\n', ...
            p, numel(restored), numel(boundaries), out_name);
    end

    fprintf('Done.\n');
end
