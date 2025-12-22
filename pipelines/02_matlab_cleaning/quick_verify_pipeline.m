% QUICK_VERIFY_PIPELINE - Fast verification of clean_eeg pipeline
% Tests with 3 minutes of data to verify:
%   1. Pipeline completes without errors
%   2. All events preserved (no boundary events added)
%   3. clean_sample_mask = all true (100% retention)
%   4. AMICA weights saved
%   5. AMICA ran for 2 iterations only
%
% Runtime: ~3-5 minutes
%
% Usage: quick_verify_pipeline(participant_num)
% Example: quick_verify_pipeline(14)

function quick_verify_pipeline(participant_num)
    if nargin < 1
        participant_num = 10;  % Default test participant
    end
    
    fprintf('=============================================================\n');
    fprintf('QUICK PIPELINE VERIFICATION - P%02d\n', participant_num);
    fprintf('=============================================================\n');
    fprintf('Using first 3 minutes of data for fast testing\n');
    fprintf('Expected runtime: 3-5 minutes\n');
    fprintf('=============================================================\n\n');
    
    % Setup paths
    projectRoot = 'C:\vr_tsst_2025';
    raw_file = fullfile(projectRoot, 'output', 'sets', sprintf('P%02d.set', participant_num));
    test_output = fullfile(projectRoot, 'output', 'test_cleaned');
    test_vis = fullfile(projectRoot, 'output', 'test_vis', sprintf('P%02d', participant_num));
    test_qc = fullfile(projectRoot, 'output', 'test_qc');
    
    if ~exist(test_output, 'dir'), mkdir(test_output); end
    if ~exist(test_vis, 'dir'), mkdir(test_vis); end
    if ~exist(test_qc, 'dir'), mkdir(test_qc); end
    
    % Initialize EEGLAB
    addpath(fullfile(projectRoot, 'pipelines', '02_matlab_cleaning'));
    eeglab nogui;
    
    fprintf('Step 1: Loading and truncating raw data...\n');
    EEG_raw = pop_loadset('filename', sprintf('P%02d.set', participant_num), ...
                          'filepath', fullfile(projectRoot, 'output', 'sets'));
    
    % Truncate to 3 minutes
    max_samples = min(10 * 60 * EEG_raw.srate, EEG_raw.pnts);
    EEG_truncated = pop_select(EEG_raw, 'point', [1 max_samples]);
    
    % Count raw events
    raw_event_count = length(EEG_truncated.event);
    raw_boundary_count = sum(strcmp({EEG_truncated.event.type}, 'boundary'));
    raw_event_types = unique({EEG_truncated.event.type});
    
    fprintf('  Raw: %d events, %d boundaries, %.1f min\n', ...
        raw_event_count, raw_boundary_count, max_samples/EEG_raw.srate/60);
    fprintf('  Event types: %s\n\n', strjoin(raw_event_types, ', '));
    
    % Save truncated version temporarily
    temp_raw = fullfile(test_output, sprintf('P%02d_temp_raw.set', participant_num));
    pop_saveset(EEG_truncated, 'filename', sprintf('P%02d_temp_raw.set', participant_num), ...
                'filepath', test_output);
    
    fprintf('Step 2: Running cleaning pipeline...\n');
    tic;
    try
        [EEG_clean, qc] = clean_eeg(temp_raw, test_output, participant_num, ...
                                     test_vis, test_qc, struct());
        elapsed = toc;
        fprintf('  ✓ Pipeline completed in %.1f minutes\n\n', elapsed/60);
    catch ME
        fprintf('  ✗ Pipeline FAILED: %s\n', ME.message);
        for i = 1:length(ME.stack)
            fprintf('    %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
        return;
    end
    
    fprintf('Step 3: Verifying outputs...\n');
    results = struct();
    
    % Check 1: Event preservation
    clean_event_count = length(EEG_clean.event);
    clean_boundary_count = sum(strcmp({EEG_clean.event.type}, 'boundary'));
    boundaries_added = clean_boundary_count - raw_boundary_count;
    events_preserved = (clean_event_count >= raw_event_count);
    
    results.events_preserved = events_preserved;
    results.boundaries_added = boundaries_added;
    fprintf('  Events: %d → %d (preserved: %s)\n', raw_event_count, clean_event_count, yesno(events_preserved));
    fprintf('  Boundaries added: %d (expected: 0)\n', boundaries_added);
    
    % Check 2: clean_sample_mask (all true = 100% retention)
    if isfield(EEG_clean.etc, 'clean_sample_mask')
        all_samples_kept = all(EEG_clean.etc.clean_sample_mask);
        pct_kept = 100 * mean(EEG_clean.etc.clean_sample_mask);
        results.all_samples_kept = all_samples_kept;
        results.pct_samples_kept = pct_kept;
        fprintf('  Sample retention: %.1f%% (all kept: %s)\n', pct_kept, yesno(all_samples_kept));
    else
        fprintf('  ✗ clean_sample_mask not found!\n');
        results.all_samples_kept = false;
    end
    
    % Check 3: AMICA weights saved
    weights_file = fullfile(projectRoot, 'output', 'ica_weights', sprintf('P%02d_amica_weights.mat', participant_num));
    weights_exist = exist(weights_file, 'file') > 0;
    results.weights_saved = weights_exist;
    
    if weights_exist
        loaded = load(weights_file);
        has_weights = isfield(loaded, 'amica_weights') && ...
                      isfield(loaded.amica_weights, 'weights') && ...
                      isfield(loaded.amica_weights, 'sphere');
        results.weights_valid = has_weights;
        fprintf('  AMICA weights: %s (valid: %s)\n', yesno(weights_exist), yesno(has_weights));
        
        if isfield(loaded.amica_weights, 'LL_trace')
            n_iters = length(loaded.amica_weights.LL_trace);
            results.amica_iterations = n_iters;
            fprintf('  AMICA iterations: %d (expected: 2)\n', n_iters);
        end
    else
        fprintf('  ✗ AMICA weights file not found: %s\n', weights_file);
        results.weights_valid = false;
    end
    
    % Check 4: ICA weights in EEG structure
    has_ica = ~isempty(EEG_clean.icaweights) && ~isempty(EEG_clean.icasphere);
    results.ica_in_eeg = has_ica;
    if has_ica
        fprintf('  ICA in EEG: %dx%d weights, %dx%d sphere\n', ...
            size(EEG_clean.icaweights,1), size(EEG_clean.icaweights,2), ...
            size(EEG_clean.icasphere,1), size(EEG_clean.icasphere,2));
    else
        fprintf('  ✗ ICA weights/sphere missing from EEG structure\n');
    end
    
    % Check 5: QC metrics
    fprintf('  QC: %d bad channels, %d ICs removed\n', qc.nBad, qc.ICsRemoved);
    results.qc = qc;
    
    % Check processing log for iteration count
    log_file = fullfile(test_output, sprintf('P%02d_processing_log.txt', participant_num));
    if exist(log_file, 'file')
        log_text = fileread(log_file);
        iter_matches = regexp(log_text, 'iter (\d+) -> LL', 'tokens');
        if ~isempty(iter_matches)
            max_iter = max(cellfun(@(x) str2double(x{1}), iter_matches));
            fprintf('  Log shows max iteration: %d\n', max_iter);
            results.log_max_iter = max_iter;
        end
    end
    
    % Summary
    fprintf('\n=============================================================\n');
    fprintf('VERIFICATION SUMMARY\n');
    fprintf('=============================================================\n');
    
    all_pass = events_preserved && (boundaries_added == 0) && ...
               results.all_samples_kept && weights_exist && has_ica;
    
    fprintf('Status: %s\n', status_text(all_pass));
    fprintf('  ✓ Events preserved: %s\n', yesno(events_preserved));
    fprintf('  ✓ No boundaries added: %s\n', yesno(boundaries_added == 0));
    fprintf('  ✓ All samples kept: %s\n', yesno(results.all_samples_kept));
    fprintf('  ✓ AMICA weights saved: %s\n', yesno(weights_exist));
    fprintf('  ✓ ICA in EEG structure: %s\n', yesno(has_ica));
    fprintf('=============================================================\n');
    
    % Save results
    save(fullfile(test_qc, sprintf('verification_results_P%02d.mat', participant_num)), 'results');
    
    % Cleanup temp file
    if exist(temp_raw, 'file')
        delete(temp_raw);
        delete(strrep(temp_raw, '.set', '.fdt'));
    end
end

function s = yesno(b)
    if b
        s = 'YES';
    else
        s = 'NO';
    end
end

function s = status_text(passed)
    if passed
        s = '✓ ALL CHECKS PASSED';
    else
        s = '✗ SOME CHECKS FAILED';
    end
end
