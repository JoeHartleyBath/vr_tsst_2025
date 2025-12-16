% QUICK_TEST_ASR_FLAG - Fast test of ASR flag-only mode without full AMICA
% Tests ASR behavior, mask saving, and event preservation in ~2-3 minutes
%
% This test:
% - Uses existing raw P01.set data
% - Runs only through ASR step (skips AMICA/ICA)
% - Validates flag-only behavior and mask saving
% - Compares event counts before/after
%
% Usage: quick_test_asr_flag

function quick_test_asr_flag()
    
    fprintf('=============================================================\n');
    fprintf('QUICK ASR FLAG-ONLY TEST (No AMICA)\n');
    fprintf('=============================================================\n\n');
    
    % Setup
    cd('C:\vr_tsst_2025');
    addpath(genpath('scripts'));
    eeglab nogui;
    
    % Test participant
    p = 1;
    
    % Paths
    raw_file = fullfile('output', 'sets', sprintf('P%02d.set', p));
    test_output = fullfile('output', 'test_cleaning');
    qc_dir = fullfile(test_output, 'qc');
    
    % Create test output directories
    if ~exist(test_output, 'dir'), mkdir(test_output); end
    if ~exist(qc_dir, 'dir'), mkdir(qc_dir); end
    if ~exist(fullfile(qc_dir, 'asr_masks'), 'dir')
        mkdir(fullfile(qc_dir, 'asr_masks'));
    end
    
    % Load raw data
    fprintf('[1/5] Loading raw P%02d...\n', p);
    if ~exist(raw_file, 'file')
        error('Raw file not found: %s', raw_file);
    end
    
    EEG_raw = pop_loadset('filename', sprintf('P%02d.set', p), 'filepath', 'output/sets');
    
    % Count events before
    raw_event_types = cellfun(@num2str, {EEG_raw.event.type}, 'UniformOutput', false);
    raw_boundaries = sum(strcmp(raw_event_types, 'boundary'));
    raw_pnts = EEG_raw.pnts;
    
    fprintf('  Raw: %d events, %d boundaries, %d timepoints\n\n', ...
            length(EEG_raw.event), raw_boundaries, raw_pnts);
    
    % Run basic preprocessing (filters + resample)
    fprintf('[2/5] Running basic preprocessing...\n');
    EEG = EEG_raw;
    
    % Assign channel locations
    chanlocs_file = fullfile('config', 'chanlocs', 'NA-271.elc');
    EEG.chanlocs = readlocs(chanlocs_file);
    EEG = eeg_checkset(EEG);
    
    % Resample to 125 Hz
    if EEG.srate ~= 125
        EEG = pop_resample(EEG, 125);
    end
    
    % Band-pass filter 1-49 Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
    
    % Remove 50 Hz line noise
    EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
    
    % Apply 25 Hz notch for P01
    if p <= 7
        EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
    end
    
    fprintf('  Done. %d channels, %d timepoints\n\n', EEG.nbchan, EEG.pnts);
    
    % Store original chanlocs
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    
    % Run ASR with flag-only mode (NEW BEHAVIOR)
    fprintf('[3/5] Running ASR (flag-only mode)...\n');
    fprintf('  Parameters: BurstCriterion=50, WindowCriterion=off, BurstRejection=off\n');
    
    try
        [EEG, com] = clean_artifacts(EEG, ...
            'FlatlineCriterion',  5, ...
            'ChannelCriterion',   0.60, ...
            'LineNoiseCriterion', 4, ...
            'BurstCriterion',     50, ...
            'BurstRejection',     'off', ...
            'WindowCriterion',    'off');
        
        fprintf('  ✓ ASR completed\n');
        asr_success = true;
    catch ME
        fprintf('  ✗ ASR failed with BurstRejection parameter\n');
        fprintf('  Trying alternative: WindowCriterion=Inf\n');
        
        % Fallback: use high threshold instead
        [EEG, com] = clean_artifacts(EEG, ...
            'FlatlineCriterion',  5, ...
            'ChannelCriterion',   0.60, ...
            'LineNoiseCriterion', 4, ...
            'BurstCriterion',     50, ...
            'WindowCriterion',    Inf);
        
        fprintf('  ✓ ASR completed (fallback method)\n');
        asr_success = true;
    end
    
    % Check sample mask
    if ~isfield(EEG.etc, 'clean_sample_mask') || isempty(EEG.etc.clean_sample_mask)
        warning('clean_sample_mask not found - assuming all samples retained.');
        EEG.etc.clean_sample_mask = true(1, EEG.pnts);
    end
    
    asr_mask = EEG.etc.clean_sample_mask;
    perc_flagged = 100 * mean(~asr_mask);
    
    fprintf('  ASR flagged: %.2f%% samples\n\n', perc_flagged);
    
    % Save mask
    fprintf('[4/5] Saving ASR mask...\n');
    mask_file = fullfile(qc_dir, 'asr_masks', sprintf('P%02d_clean_sample_mask.mat', p));
    save(mask_file, 'asr_mask');
    fprintf('  ✓ Saved: %s\n\n', mask_file);
    
    % Check events after ASR
    fprintf('[5/5] Validating results...\n');
    
    clean_event_types = cellfun(@num2str, {EEG.event.type}, 'UniformOutput', false);
    clean_boundaries = sum(strcmp(clean_event_types, 'boundary'));
    clean_pnts = EEG.pnts;
    
    % Calculate differences
    events_lost = length(raw_event_types) - length(clean_event_types);
    boundaries_added = clean_boundaries - raw_boundaries;
    pnts_lost = raw_pnts - clean_pnts;
    
    % Print results table
    fprintf('\n');
    fprintf('=============================================================\n');
    fprintf('RESULTS SUMMARY\n');
    fprintf('=============================================================\n');
    fprintf('Metric                 | Raw    | Clean  | Diff\n');
    fprintf('-----------------------+--------+--------+---------\n');
    fprintf('Events                 | %6d | %6d | %+6d\n', length(raw_event_types), length(clean_event_types), events_lost);
    fprintf('Boundaries             | %6d | %6d | %+6d\n', raw_boundaries, clean_boundaries, boundaries_added);
    fprintf('Timepoints             | %6d | %6d | %+6d\n', raw_pnts, clean_pnts, pnts_lost);
    fprintf('ASR Flagged            |      - | %5.1f%% |       -\n', perc_flagged);
    fprintf('=============================================================\n\n');
    
    % Validation checks
    fprintf('VALIDATION CHECKS:\n');
    
    pass_count = 0;
    total_checks = 4;
    
    % Check 1: No timepoints lost
    if pnts_lost == 0
        fprintf('  ✓ Timepoints preserved (flag-only mode working)\n');
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ Timepoints lost: %d (flag-only mode may not be working)\n', pnts_lost);
    end
    
    % Check 2: No additional boundaries
    if boundaries_added == 0
        fprintf('  ✓ No boundaries added (no window rejection)\n');
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ Boundaries added: %d (window rejection occurred)\n', boundaries_added);
    end
    
    % Check 3: Task events preserved
    task_events_raw = setdiff(unique(raw_event_types), 'boundary');
    task_events_clean = setdiff(unique(clean_event_types), 'boundary');
    lost_task_events = setdiff(task_events_raw, task_events_clean);
    
    if isempty(lost_task_events)
        fprintf('  ✓ All task events preserved\n');
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ Lost task events: %s\n', strjoin(lost_task_events, ', '));
    end
    
    % Check 4: Mask saved
    if exist(mask_file, 'file')
        fprintf('  ✓ ASR mask saved\n');
        pass_count = pass_count + 1;
    else
        fprintf('  ✗ ASR mask not saved\n');
    end
    
    fprintf('\n');
    fprintf('=============================================================\n');
    fprintf('TEST RESULT: %d/%d checks passed\n', pass_count, total_checks);
    
    if pass_count == total_checks
        fprintf('STATUS: ✓ ALL TESTS PASSED\n');
    else
        fprintf('STATUS: ✗ SOME TESTS FAILED\n');
    end
    
    fprintf('=============================================================\n');
    fprintf('\nTest completed in test_cleaning/ directory\n');
    fprintf('No changes made to original cleaned_eeg/ files\n\n');
end
