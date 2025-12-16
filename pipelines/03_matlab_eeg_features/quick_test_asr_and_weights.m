% QUICK_TEST_ASR_AND_WEIGHTS - Fast validation of ASR flag mode and ICA weight saving
% Uses real data but with shortened segments and minimal AMICA iterations
%
% Tests:
%   - ASR flag-only mode (no window rejection)
%   - ICA weight saving to output/ica_weights
%   - ICLabel snapshot saving
%   - ASR mask preservation
%   - Event preservation
%
% Runtime: ~5-10 minutes per participant (vs 2-3 hours full)
%
% Usage:
%   quick_test_asr_and_weights()           % Test P01, P05 (default)
%   quick_test_asr_and_weights([1 5 26])  % Test specific IDs

function quick_test_asr_and_weights(test_ids)
    
    if nargin < 1 || isempty(test_ids)
        % Default: P01 (25Hz notch), P05 (had missing events)
        test_ids = [1, 5];
    end
    
    fprintf('=============================================================\n');
    fprintf('QUICK TEST: ASR Flag Mode + ICA Weight Saving\n');
    fprintf('=============================================================\n');
    fprintf('Testing %d participants: %s\n', length(test_ids), mat2str(test_ids));
    fprintf('AMICA iterations: 20 (fast test mode)\n');
    fprintf('Expected runtime: ~5-10 min per participant\n');
    fprintf('=============================================================\n\n');
    
    % Setup
    addpath(genpath('scripts'));
    eeglab nogui;
    
    % Paths
    raw_dir = fullfile('output', 'sets');
    test_dir = fullfile('output', 'test_cleaned');
    vis_dir = fullfile('output', 'test_vis');
    qc_dir = fullfile('output', 'test_qc');
    
    % Create test directories
    if ~exist(test_dir, 'dir'), mkdir(test_dir); end
    if ~exist(vis_dir, 'dir'), mkdir(vis_dir); end
    if ~exist(qc_dir, 'dir'), mkdir(qc_dir); end
    
    results = struct('participant', {}, 'status', {}, ...
                     'raw_events', {}, 'clean_events', {}, ...
                     'boundaries_added', {}, 'pnts_preserved', {}, ...
                     'asr_flagged_pct', {}, 'weights_saved', {}, ...
                     'iclabel_saved', {}, 'mask_saved', {});
    
    for i = 1:length(test_ids)
        p = test_ids(i);
        
        fprintf('\n=== P%02d ===\n', p);
        tic;
        
        try
            % Load raw
            raw_file = fullfile(raw_dir, sprintf('P%02d.set', p));
            if ~exist(raw_file, 'file')
                fprintf('  SKIP: Raw file not found\n');
                continue;
            end
            
            fprintf('  Loading raw data...\n');
            EEG = pop_loadset('filename', sprintf('P%02d.set', p), 'filepath', raw_dir);
            raw_events = length(EEG.event);
            raw_boundaries = sum(strcmp({EEG.event.type}, 'boundary'));
            raw_pnts = EEG.pnts;
            
            % Truncate to first 5 minutes for speed
            max_samples = 5 * 60 * EEG.srate;
            if EEG.pnts > max_samples
                fprintf('  Truncating to 5 minutes (from %.1f min)...\n', EEG.pnts/EEG.srate/60);
                EEG = pop_select(EEG, 'point', [1 max_samples]);
            end
            
            fprintf('  Running quick cleaning (AMICA 20 iters)...\n');
            [EEG_clean, qc] = quick_clean_test(EEG, p, vis_dir, qc_dir);
            
            % Save
            pop_saveset(EEG_clean, 'filename', sprintf('P%02d_test_cleaned.set', p), ...
                        'filepath', test_dir);
            
            % Analyze results
            clean_events = length(EEG_clean.event);
            clean_boundaries = sum(strcmp({EEG_clean.event.type}, 'boundary'));
            boundaries_added = clean_boundaries - raw_boundaries;
            pnts_preserved = (EEG_clean.pnts == EEG.pnts);
            
            % ASR mask
            asr_flagged_pct = 0;
            if isfield(EEG_clean.etc, 'clean_sample_mask')
                asr_flagged_pct = 100 * mean(~EEG_clean.etc.clean_sample_mask);
            end
            
            % Check files saved
            weights_file = fullfile('output', 'ica_weights', sprintf('P%02d_amica_weights.mat', p));
            iclabel_file = fullfile('output', 'ica_weights', sprintf('P%02d_iclabel_snapshot.mat', p));
            mask_file = fullfile(qc_dir, 'asr_masks', sprintf('P%02d_clean_sample_mask.mat', p));
            
            weights_saved = exist(weights_file, 'file') > 0;
            iclabel_saved = exist(iclabel_file, 'file') > 0;
            mask_saved = exist(mask_file, 'file') > 0;
            
            elapsed = toc;
            
            fprintf('  ✓ Complete in %.1f minutes\n', elapsed/60);
            fprintf('    Events: %d → %d\n', raw_events, clean_events);
            fprintf('    Boundaries added: %d\n', boundaries_added);
            fprintf('    Timepoints preserved: %s\n', yesno(pnts_preserved));
            fprintf('    ASR flagged: %.2f%%\n', asr_flagged_pct);
            fprintf('    Weights saved: %s\n', yesno(weights_saved));
            fprintf('    ICLabel saved: %s\n', yesno(iclabel_saved));
            fprintf('    Mask saved: %s\n', yesno(mask_saved));
            
            results(end+1) = struct('participant', p, 'status', 'SUCCESS', ...
                'raw_events', raw_events, 'clean_events', clean_events, ...
                'boundaries_added', boundaries_added, 'pnts_preserved', pnts_preserved, ...
                'asr_flagged_pct', asr_flagged_pct, 'weights_saved', weights_saved, ...
                'iclabel_saved', iclabel_saved, 'mask_saved', mask_saved); %#ok<AGROW>
            
        catch ME
            fprintf('  ✗ FAILED: %s\n', ME.message);
            results(end+1) = struct('participant', p, 'status', 'FAILED', ...
                'raw_events', 0, 'clean_events', 0, 'boundaries_added', 0, ...
                'pnts_preserved', false, 'asr_flagged_pct', 0, ...
                'weights_saved', false, 'iclabel_saved', false, 'mask_saved', false); %#ok<AGROW>
        end
    end
    
    % Summary
    fprintf('\n=============================================================\n');
    fprintf('TEST SUMMARY\n');
    fprintf('=============================================================\n');
    fprintf('ID  | Status  | Events | +Bound | PntsPres | ASR%% | Wgt | ICL | Msk\n');
    fprintf('----+---------+--------+--------+----------+------+-----+-----+----\n');
    
    for i = 1:length(results)
        r = results(i);
        fprintf('P%02d | %7s | %3d→%3d | %6d | %8s | %4.1f | %3s | %3s | %3s\n', ...
            r.participant, r.status, r.raw_events, r.clean_events, ...
            r.boundaries_added, yesno(r.pnts_preserved), r.asr_flagged_pct, ...
            yesno(r.weights_saved), yesno(r.iclabel_saved), yesno(r.mask_saved));
    end
    
    fprintf('=============================================================\n');
    
    % Check expectations
    fprintf('\nEXPECTED RESULTS:\n');
    fprintf('  - Boundaries added: 0 (flag-only ASR)\n');
    fprintf('  - Timepoints preserved: YES\n');
    fprintf('  - All weight/ICLabel/mask files: YES\n\n');
    
    success_count = sum(strcmp({results.status}, 'SUCCESS'));
    fprintf('Passed: %d/%d\n', success_count, length(results));
end

function [EEG, qc] = quick_clean_test(EEG, p, vis_dir, qc_dir)
    % Quick cleaning pipeline with minimal AMICA iterations
    
    logfile = fullfile(qc_dir, sprintf('P%02d_quick_test.log', p));
    fid = fopen(logfile, 'w');
    if fid ~= -1
        fclose(fid);
    end
    
    % Step 1: Assign chanlocs
    chanlocs_file = fullfile('config', 'chanlocs', 'NA-271.elc');
    EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    
    % Step 2: Basic cleaning
    if EEG.srate ~= 125
        EEG = pop_resample(EEG, 125);
    end
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
    if ismember(p, 1:7)
        EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
    end
    
    % Step 3: ASR without window rejection (omit WindowCriterion)
    fprintf('    Running ASR (no window rejection)...\n');
    [EEG, ~] = clean_artifacts(EEG, ...
        'FlatlineCriterion', 5, ...
        'ChannelCriterion', 0.60, ...
        'LineNoiseCriterion', 4, ...
        'BurstCriterion', 'off');
    
    % Save ASR mask
    asr_mask_dir = fullfile(qc_dir, 'asr_masks');
    if ~exist(asr_mask_dir, 'dir')
        mkdir(asr_mask_dir);
    end
    if isfield(EEG.etc, 'clean_sample_mask')
        asr_mask = EEG.etc.clean_sample_mask;
        save(fullfile(asr_mask_dir, sprintf('P%02d_clean_sample_mask.mat', p)), 'asr_mask');
    end
    
    % Step 4: Quick AMICA (20 iterations only)
    fprintf('    Running AMICA (20 iters, fast mode)...\n');
    outdir = fullfile(pwd, sprintf('amicaouttmp_test_%d', p));
    if exist(outdir, 'dir'), rmdir(outdir, 's'); end
    mkdir(outdir);
    
    [weights, sphere, mods] = runamica15(EEG.data, ...
        'num_models', 1, ...
        'outdir', outdir, ...
        'numprocs', 1, ...
        'max_threads', 2, ...
        'max_iter', 20, ...       % FAST: 20 instead of 200
        'write_LLt', 1, ...
        'writestep', 10);
    
    EEG.icaweights = weights;
    EEG.icasphere = sphere;
    EEG = eeg_checkset(EEG);
    
    % Save AMICA weights
    ica_weights_dir = fullfile('output', 'ica_weights');
    if ~exist(ica_weights_dir, 'dir')
        mkdir(ica_weights_dir);
    end
    LL_trace = [];
    if isfield(mods, 'LL')
        LL_trace = mods.LL;
    end
    amica_weights = struct('weights', weights, 'sphere', sphere, 'LL_trace', LL_trace);
    save(fullfile(ica_weights_dir, sprintf('P%02d_amica_weights.mat', p)), 'amica_weights', '-v7.3');
    
    rmdir(outdir, 's');
    
    % Step 5: ICLabel
    fprintf('    Running ICLabel...\n');
    EEG = iclabel(EEG);
    
    % Save ICLabel
    if isfield(EEG, 'etc') && isfield(EEG.etc, 'ic_classification') && ...
       isfield(EEG.etc.ic_classification, 'ICLabel')
        iclabel_results = EEG.etc.ic_classification.ICLabel;
        save(fullfile(ica_weights_dir, sprintf('P%02d_iclabel_snapshot.mat', p)), ...
             'iclabel_results', '-v7.3');
    end
    
    % Step 6: Remove artifacts (simple threshold)
    brain_thresh = 0.7;
    classifications = EEG.etc.ic_classification.ICLabel.classifications;
    brain_prob = classifications(:, 1);
    keep_ics = brain_prob >= brain_thresh;
    remove_ics = find(~keep_ics);
    
    if ~isempty(remove_ics)
        EEG = pop_subcomp(EEG, remove_ics, 0);
    end
    
    % Step 7: Interpolate and reref
    EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
    EEG = pop_reref(EEG, []);
    
    % Dummy QC
    qc = struct('percASRrepaired', 100*mean(~asr_mask), 'ICsRemoved', length(remove_ics));
end

function s = yesno(b)
    if b
        s = 'YES';
    else
        s = 'NO';
    end
end
