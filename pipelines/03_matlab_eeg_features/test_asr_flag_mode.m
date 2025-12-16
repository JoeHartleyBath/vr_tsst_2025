% TEST_ASR_FLAG_MODE - Quick validation of ASR flag-only strategy
% Tests impact on events, boundaries, and data quality for selected participants
%
% Usage:
%   test_asr_flag_mode([1, 2, 26])  % Test specific participants
%   test_asr_flag_mode([])          % Test P01, P02, P26 (default)

function test_asr_flag_mode(participant_ids)
    
    if nargin < 1 || isempty(participant_ids)
        % Default test set: P01 (25 Hz notch), P02 (noisy), P26 (later session)
        participant_ids = [1, 2, 26];
    end
    
    fprintf('=============================================================\n');
    fprintf('ASR FLAG-ONLY MODE VALIDATION TEST\n');
    fprintf('=============================================================\n');
    fprintf('Testing %d participants: %s\n', length(participant_ids), mat2str(participant_ids));
    fprintf('=============================================================\n\n');
    
    % Setup paths
    project_root = pwd;
    raw_dir = fullfile(project_root, 'output', 'sets');
    clean_dir = fullfile(project_root, 'output', 'cleaned_eeg');
    backup_dir = fullfile(project_root, 'output', 'cleaned_eeg_backup');
    vis_base = fullfile(project_root, 'output', 'vis');
    qc_dir = fullfile(project_root, 'output', 'qc');
    
    % Add cleaning scripts to path
    addpath(fullfile(project_root, 'scripts', 'preprocessing', 'eeg', 'cleaning'));
    
    % Initialize EEGLAB
    eeglab nogui;
    
    % Store results
    results = struct('participant', {}, 'raw_events', {}, 'new_events', {}, ...
                     'raw_boundaries', {}, 'new_boundaries', {}, ...
                     'pnts_preserved', {}, 'asr_flagged_pct', {}, ...
                     'weights_saved', {}, 'iclabel_saved', {}, 'mask_saved', {});
    
    for i = 1:length(participant_ids)
        p = participant_ids(i);
        
        fprintf('\n=== P%02d ===\n', p);
        
        % Check raw file exists
        raw_file = fullfile(raw_dir, sprintf('P%02d.set', p));
        if ~exist(raw_file, 'file')
            fprintf('  SKIP: Raw file not found: %s\n', raw_file);
            continue;
        end
        
        % Backup existing cleaned files if they exist
        clean_file = fullfile(clean_dir, sprintf('P%02d_cleaned.set', p));
        if exist(clean_file, 'file')
            fprintf('  Backing up existing cleaned files...\n');
            if ~exist(backup_dir, 'dir')
                mkdir(backup_dir);
            end
            
            % Move existing files to backup
            files_to_backup = {
                fullfile(clean_dir, sprintf('P%02d_cleaned.set', p))
                fullfile(clean_dir, sprintf('P%02d_cleaned.fdt', p))
                fullfile(clean_dir, sprintf('P%02d_cleaned.mat', p))
                fullfile(clean_dir, sprintf('P%02d_processing_log.txt', p))
                fullfile(qc_dir, sprintf('QC_P%02d.txt', p))
                fullfile(qc_dir, sprintf('P%02d_qc.mat', p))
            };
            
            for f = 1:length(files_to_backup)
                if exist(files_to_backup{f}, 'file')
                    [~, name, ext] = fileparts(files_to_backup{f});
                    copyfile(files_to_backup{f}, fullfile(backup_dir, [name ext]));
                    delete(files_to_backup{f});
                end
            end
        end
        
        % Load raw to count events
        fprintf('  Loading raw set...\n');
        EEG_raw = pop_loadset('filename', sprintf('P%02d.set', p), 'filepath', raw_dir);
        raw_event_types = cellfun(@num2str, {EEG_raw.event.type}, 'UniformOutput', false);
        raw_boundaries = sum(strcmp(raw_event_types, 'boundary'));
        raw_pnts = EEG_raw.pnts;
        
        fprintf('  Raw: %d events, %d boundaries, %d timepoints\n', ...
                length(EEG_raw.event), raw_boundaries, raw_pnts);
        
        % Run cleaning with new ASR flag-only mode
        fprintf('  Running cleaning pipeline (ASR flag-only)...\n');
        vis_folder = fullfile(vis_base, sprintf('P%02d', p));
        
        try
            [EEG_clean, qc] = clean_eeg(raw_file, clean_dir, p, vis_folder, qc_dir, []);
            
            % Analyze cleaned results
            fprintf('  Cleaning complete. Analyzing...\n');
            
            % Reload cleaned file
            EEG_clean = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), 'filepath', clean_dir);
            
            clean_event_types = cellfun(@num2str, {EEG_clean.event.type}, 'UniformOutput', false);
            clean_boundaries = sum(strcmp(clean_event_types, 'boundary'));
            clean_pnts = EEG_clean.pnts;
            
            % Check ASR mask
            asr_flagged_pct = 0;
            if isfield(EEG_clean.etc, 'clean_sample_mask')
                asr_flagged_pct = 100 * mean(~EEG_clean.etc.clean_sample_mask);
            end
            
            % Check if files were saved
            weights_file = fullfile(project_root, 'output', 'ica_weights', sprintf('P%02d_amica_weights.mat', p));
            iclabel_file = fullfile(project_root, 'output', 'ica_weights', sprintf('P%02d_iclabel_snapshot.mat', p));
            mask_file = fullfile(qc_dir, 'asr_masks', sprintf('P%02d_clean_sample_mask.mat', p));
            
            weights_saved = exist(weights_file, 'file') > 0;
            iclabel_saved = exist(iclabel_file, 'file') > 0;
            mask_saved = exist(mask_file, 'file') > 0;
            
            % Store results
            results(end+1) = struct(...
                'participant', p, ...
                'raw_events', length(raw_event_types), ...
                'new_events', length(clean_event_types), ...
                'raw_boundaries', raw_boundaries, ...
                'new_boundaries', clean_boundaries, ...
                'pnts_preserved', (clean_pnts == raw_pnts), ...
                'asr_flagged_pct', asr_flagged_pct, ...
                'weights_saved', weights_saved, ...
                'iclabel_saved', iclabel_saved, ...
                'mask_saved', mask_saved); %#ok<AGROW>
            
            % Print summary
            fprintf('  New: %d events, %d boundaries, %d timepoints\n', ...
                    length(clean_event_types), clean_boundaries, clean_pnts);
            fprintf('  Timepoints preserved: %s\n', bool2str(clean_pnts == raw_pnts));
            fprintf('  ASR flagged: %.2f%%\n', asr_flagged_pct);
            fprintf('  AMICA weights saved: %s\n', bool2str(weights_saved));
            fprintf('  ICLabel saved: %s\n', bool2str(iclabel_saved));
            fprintf('  ASR mask saved: %s\n', bool2str(mask_saved));
            
            % Event comparison
            task_events = setdiff(unique(raw_event_types), 'boundary');
            task_events_clean = setdiff(unique(clean_event_types), 'boundary');
            lost_events = setdiff(task_events, task_events_clean);
            
            if ~isempty(lost_events)
                fprintf('  WARNING: Lost events: %s\n', strjoin(lost_events, ', '));
            else
                fprintf('  ✓ All task events preserved\n');
            end
            
            fprintf('  ✓ SUCCESS\n');
            
        catch ME
            fprintf('  ✗ FAILED: %s\n', ME.message);
            fprintf('    %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
        end
    end
    
    % Print summary table
    fprintf('\n=============================================================\n');
    fprintf('VALIDATION SUMMARY\n');
    fprintf('=============================================================\n');
    fprintf('ID  | Events | Bounds | PntsPres | ASRFlag | Weights | ICLabel | Mask\n');
    fprintf('----+--------+--------+----------+---------+---------+---------+------\n');
    
    for i = 1:length(results)
        r = results(i);
        fprintf('P%02d | %3d→%3d | %2d→%2d | %s     | %5.1f%%  | %s    | %s    | %s\n', ...
                r.participant, r.raw_events, r.new_events, ...
                r.raw_boundaries, r.new_boundaries, ...
                bool2str(r.pnts_preserved), r.asr_flagged_pct, ...
                bool2str(r.weights_saved), bool2str(r.iclabel_saved), bool2str(r.mask_saved));
    end
    
    fprintf('=============================================================\n');
    fprintf('COMPLETE\n');
    fprintf('=============================================================\n\n');
end

function s = bool2str(b)
    if b
        s = 'YES';
    else
        s = 'NO ';
    end
end
