function [EEG, qc] = clean_eeg(raw_set_path, output_folder, participant_num, vis_folder, qc_folder, config, max_threads_override)
    % CLEAN_EEG - Streamlined EEG cleaning pipeline
    %
    % Applies basic and advanced cleaning to a raw .set file produced by
    % the Python xdf_to_set pipeline. Saves cleaned .mat and .set files
    % with QC metrics.
    %
    % INPUTS:
    %   raw_set_path   - Full path to raw .set file (from xdf_to_set.py)
    %   output_folder  - Folder to save cleaned .mat and .set files
    %   participant_num - Participant number (e.g., 1 for P01)
    %   vis_folder     - Folder to save visualization PNGs
    %   qc_folder      - Folder to save QC metrics
    %   config         - YAML config struct (optional, loads default if not provided)
    %
    % OUTPUTS:
    %   EEG - Cleaned EEGLAB structure
    %   qc  - Quality control metrics struct
    %
    % EXAMPLE:
    %   [EEG, qc] = clean_eeg('data/raw/eeg/P01_raw.set', ...
    %                         'output/cleaned_eeg', 1, ...
    %                         'output/vis/P01', 'output/qc', []);
    
    %% Setup and Validation
    % Add project scripts to path
    projectRoot = 'C:\vr_tsst_2025';
    addpath(fullfile(projectRoot, 'scripts', 'preprocessing', 'eeg', 'cleaning'));
    addpath(fullfile(projectRoot, 'scripts', 'utils'));
    
    if nargin < 7 || isempty(max_threads_override)
        max_threads_override = 8;  % Default to 8 threads
    end
    
    if nargin < 6 || isempty(config)
        % Load config with fallback support (ReadYaml → SimpleYAML → skip)
        configPath = fullfile(projectRoot, 'config', 'general.yaml');
        config = struct();
        
        if exist('ReadYaml', 'file') == 2
            try
                config = ReadYaml(configPath);
            catch
                % Fall through to SimpleYAML
            end
        end
        if isempty(fieldnames(config)) && exist('SimpleYAML', 'file') == 2
            try
                config = SimpleYAML.readFile(configPath);
            catch
                % Continue with empty config
            end
        end
    end
    
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    if ~exist(vis_folder, 'dir')
        mkdir(vis_folder);
    end
    if ~exist(qc_folder, 'dir')
        mkdir(qc_folder);
    end
    
    logfile = fullfile(output_folder, sprintf('P%02d_processing_log.txt', participant_num));
    
    %% Step 1: Load Raw Data
    log_message(logfile, '=== Step 1: Load Raw Data ===');
    
    [filepath, filename, ext] = fileparts(raw_set_path);
    EEG = pop_loadset('filename', [filename, ext], 'filepath', filepath);
    
    if isempty(EEG.data)
        error('Loaded dataset contains no data.');
    end
    
    % Fix data types for MATLAB compatibility (in case Python saved as integers)
    EEG.xmin = double(EEG.xmin);
    EEG.xmax = double(EEG.xmax);
    EEG.srate = double(EEG.srate);
    EEG.pnts = double(EEG.pnts);
    EEG.nbchan = double(EEG.nbchan);
    EEG.trials = double(EEG.trials);
    
    log_message(logfile, sprintf('Loaded: %d channels, %d samples, %.1f Hz', ...
        EEG.nbchan, EEG.pnts, EEG.srate));

    % Log amplitude stats after load
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after load: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_01_raw_loaded.png', participant_num));
    % catch
    % end
    
    %% Step 2: Assign Channel Locations
    log_message(logfile, '=== Step 2: Assign Channel Locations ===');

    % Assign ANT Neuro 128-channel equidistant layout (resolve via project root)
    chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
    EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);

    if isempty(EEG.chanlocs) || isempty(EEG.chanlocs(1).X)
        warning('Failed to assign channel locations.');
    else
        log_message(logfile, 'Channel locations assigned.');
    end

    % Persist original chanlocs after template assignment for later interpolation/saving
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_02_chanlocs.png', participant_num));
    % catch
    % end
    
    %% Step 3: Basic Cleaning (Filtering)
    log_message(logfile, '=== Step 3: Basic Cleaning ===');
    
    % Resample to 125 Hz if needed
    if EEG.srate ~= 125
        EEG = pop_resample(EEG, 125);
        log_message(logfile, 'Resampled to 125 Hz.');
    end
    
    % Band-pass filter 1-49 Hz
    EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
    log_message(logfile, 'Applied 1-49 Hz band-pass filter.');
    
    % Remove 50 Hz line noise with notch filter
    EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
    log_message(logfile, 'Applied 50 Hz notch filter.');
    
    % Apply 25 Hz notch for participants 1-7
    if ismember(participant_num, 1:7)
        EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
        log_message(logfile, 'Applied 25 Hz notch filter.');
    end

    % Log amplitude stats after basic cleaning
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after basic cleaning: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_03_basic_clean.png', participant_num));
    % catch
    % end
    
    %% Step 4: Detect Bad Channels Only (No Data Removal)
    log_message(logfile, '=== Step 4: Bad Channel Detection (for interpolation) ===');

    log_message(logfile, 'Skipping channel detection - all channels will be preserved.');
    
    % Detect bad channels WITHOUT removing any samples/timepoints
    try
        [EEG, ~] = clean_artifacts(EEG, ...
            'FlatlineCriterion', 5, ...
            'ChannelCriterion', 0.60, ...
            'LineNoiseCriterion', 4, ...
            'BurstCriterion', 'off', ...      % No burst detection/repair
            'WindowCriterion', 'off');         % No window rejection
        log_message(logfile, 'Bad channel detection completed (no samples removed).');
    catch ME
        log_message(logfile, sprintf('Channel detection failed: %s. Proceeding with all channels.', ME.message));
        EEG.etc.clean_channel_mask = true(1, EEG.nbchan);
    end

    % Ensure all samples are marked as kept (no data removal)
    if ~isfield(EEG.etc, 'clean_sample_mask') || isempty(EEG.etc.clean_sample_mask)
        EEG.etc.clean_sample_mask = true(1, EEG.pnts);
    end
    
    log_message(logfile, sprintf('All %d channels and %d timepoints preserved (100%%) for AMICA and event recovery.', EEG.nbchan, EEG.pnts));

    % Log stats (data unchanged)
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats pre-AMICA: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Identify bad channels (none, all will be kept)
    mask = EEG.etc.clean_channel_mask;
    origLocs = EEG.etc.orig_chanlocs;
    
    if length(origLocs) ~= length(mask)
        error('Mismatch between orig_chanlocs (%d) and clean_channel_mask (%d).', ...
            length(origLocs), length(mask));
    end
    
    badIdx = ~mask;
    badLabels = {origLocs(badIdx).labels}';
    
    log_message(logfile, sprintf('%d bad channels identified: %s', ...
        sum(badIdx), strjoin(badLabels, ', ')));
    
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_04_pre_amica.png', participant_num));
    % catch
    %     % Visualization failed, continue anyway
    % end
    
    %% Step 5: Run AMICA (ICA)
    log_message(logfile, '=== Step 5: Run AMICA ===');
    
    [EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile, max_threads_override);
    
    % Log AMICA likelihood trace
    if ~isempty(LL_trace)
        log_message(logfile, '--- AMICA Log-Likelihood Trace (every 10 iters) ---');
        for idx = 1:length(LL_trace)
            if mod(idx, 10) == 0
                log_message(logfile, sprintf('  iter %d -> LL = %.4f', idx, LL_trace(idx)));
            end
        end
        log_message(logfile, '--- End AMICA Trace ---');
    end
    
    % Save AMICA weights to separate file
    ica_weights_dir = fullfile(projectRoot, 'output', 'ica_weights');
    if ~exist(ica_weights_dir, 'dir')
        mkdir(ica_weights_dir);
    end
    amica_weights_path = fullfile(ica_weights_dir, sprintf('P%02d_amica_weights.mat', participant_num));
    amica_weights = struct('weights', EEG.icaweights, 'sphere', EEG.icasphere, 'LL_trace', LL_trace);
    save(amica_weights_path, 'amica_weights', '-v7.3');
    log_message(logfile, sprintf('AMICA weights saved to: %s', amica_weights_path));

    % Log stats after AMICA unmixing applied
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after AMICA: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_05_after_amica.png', participant_num));
    % catch
    % end
    
    %% Step 6: Apply ICLabel and Remove Artifacts
    log_message(logfile, '=== Step 6: ICLabel and Artifact Removal ===');
    
    EEG = iclabel(EEG);
    % Save ICLabel results for post-hoc inspection (use correct field)
    try
        output_dir = fullfile(pwd, '../../../../output/ica_weights');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        iclabel_snapshot_path = fullfile(output_dir, sprintf('P%02d_iclabel_snapshot.mat', participant_num));
        if isfield(EEG, 'etc') && isfield(EEG.etc, 'ic_classification') && isfield(EEG.etc.ic_classification, 'ICLabel')
            iclabel_results = EEG.etc.ic_classification.ICLabel;
            save(iclabel_snapshot_path, 'iclabel_results', '-v7.3');
            log_message(logfile, sprintf('ICLabel results snapshot saved: %s', iclabel_snapshot_path));
        else
            warning('[ICLabel Save] ICLabel results not found in EEG.etc.ic_classification.ICLabel');
        end
    catch ME
        warning('[ICLabel Save] Failed to save ICLabel snapshot: %s', ME.message);
    end
    log_message(logfile, sprintf('ICLabel applied. %d components classified.', size(EEG.icaweights, 1)));
    
    EEG = flag_and_remove_artifacts(EEG, logfile);

    % Log stats after IC removal step
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after IC removal: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Visualization disabled for performance (5-10 sec each)
    % try
    %     save_visualization(EEG, vis_folder, sprintf('P%02d_06_after_artifact_removal.png', participant_num));
    % catch
    % end
    
    %% Step 7: Interpolate Bad Channels and Re-reference
    log_message(logfile, '=== Step 7: Interpolate and Re-reference ===');
    
    EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
    log_message(logfile, 'Bad channels interpolated.');
    
    EEG = pop_reref(EEG, []);
    log_message(logfile, sprintf('Data re-referenced to average. Mean: %.4f', mean(EEG.data(:))));

    % Log stats after interpolation + reref
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after reref: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Keep final visualization for quality check
    try
        save_visualization(EEG, vis_folder, sprintf('P%02d_07_final_clean.png', participant_num));
    catch
    end
    
    %% Step 8: Compute QC Metrics
    log_message(logfile, '=== Step 8: QC Metrics ===');
    
    % Get bad channel labels from clean_channel_mask
    if isfield(EEG.etc, 'clean_channel_mask') && isfield(EEG.etc, 'orig_chanlocs')
        mask = EEG.etc.clean_channel_mask;
        origLocs = EEG.etc.orig_chanlocs;
        badIdx = ~mask;
        badLabels = {origLocs(badIdx).labels}';
    else
        badLabels = {};
    end
    
    qc = compute_qc_metrics(EEG, badLabels, logfile);
    
    % Save QC metrics
    save(fullfile(qc_folder, sprintf('P%02d_qc.mat', participant_num)), 'qc');
    
    % Write QC text report
    write_qc_report(qc_folder, participant_num, qc);
    
    log_message(logfile, 'QC metrics saved.');
    
    %% Step 9: Save Cleaned Data
    log_message(logfile, '=== Step 9: Save Cleaned Data ===');
    
    % Save as .mat (data matrix only)
    cleaned_mat_path = fullfile(output_folder, sprintf('P%02d_cleaned.mat', participant_num));
    cleaned_EEG = double(EEG.data);
    stats = [min(cleaned_EEG(:)), max(cleaned_EEG(:)), mean(cleaned_EEG(:)), std(cleaned_EEG(:))];
    log_message(logfile, sprintf('Stats at save: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    save(cleaned_mat_path, 'cleaned_EEG', '-v7.3');
    log_message(logfile, sprintf('Cleaned EEG matrix saved: %s', cleaned_mat_path));
    
    % Ensure chanlocs are present before saving
    if (isempty(EEG.chanlocs) || length(EEG.chanlocs) == 0 || isempty(EEG.chanlocs(1).X)) && isfield(EEG, 'etc') && isfield(EEG.etc, 'orig_chanlocs')
        if ~isempty(EEG.etc.orig_chanlocs)
            log_message(logfile, 'WARNING: chanlocs empty, restoring from orig_chanlocs');
            EEG.chanlocs = EEG.etc.orig_chanlocs;
        end
    end

    % Verify chanlocs before saving
    if isempty(EEG.chanlocs) || length(EEG.chanlocs) < EEG.nbchan
        log_message(logfile, sprintf('ERROR: chanlocs missing or incomplete! nbchan=%d, chanlocs length=%d', EEG.nbchan, length(EEG.chanlocs)));
        error('Cannot save .set file without proper channel locations');
    else
        log_message(logfile, sprintf('Verified: %d chanlocs present (labels: %s, %s, %s, ...)', ...
            length(EEG.chanlocs), EEG.chanlocs(1).labels, EEG.chanlocs(2).labels, EEG.chanlocs(3).labels));
    end

    % Save as .set (full EEGLAB structure)
    cleaned_set_path = fullfile(output_folder, sprintf('P%02d_cleaned.set', participant_num));
    EEG.setname = sprintf('P%02d_cleaned', participant_num);
    pop_saveset(EEG, 'filename', sprintf('P%02d_cleaned.set', participant_num), ...
        'filepath', output_folder);
    log_message(logfile, sprintf('Cleaned EEG .set saved: %s', cleaned_set_path));
    
    log_message(logfile, '=== CLEANING PIPELINE COMPLETE ===');
end


%% ========== HELPER FUNCTIONS ==========

function [EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile, max_threads)
    % Run AMICA with log-likelihood tracking
    % Supports parallel processing by accepting thread count as parameter
    
    num_models   = 1;
    numprocs     = 1;
    % max_threads passed in as parameter (for parallel processing support)
    max_iter     = 200;
    writeStep    = 10;
    
    outdir = fullfile(pwd, sprintf('amicaouttmp_%d', participant_num));
    
    % Clean up old directory
    if exist(outdir, 'dir')
        try
            rmdir(outdir, 's');
        catch ME
            warning('Could not remove existing AMICA directory: %s', ME.message);
        end
    end
    mkdir(outdir);
    
    % Run AMICA
    [weights, sphere, mods] = runamica15(EEG.data, ...
        'num_models',   num_models, ...
        'outdir',       outdir, ...
        'numprocs',     numprocs, ...
        'max_threads',  max_threads, ...
        'max_iter',     max_iter, ...
        'write_LLt',    1, ...
        'writestep',    writeStep);
    
    % Apply weights and sphere
    EEG.icaweights = weights;
    EEG.icasphere  = sphere;
    EEG = eeg_checkset(EEG);
    
    log_message(logfile, 'AMICA completed. ICA weights applied.');
    
    % Extract LL trace
    if isfield(mods, 'LL')
        LL_trace = mods.LL;
    else
        LL_trace = [];
    end
    
    % Clean up AMICA directory
    try
        rmdir(outdir, 's');
    catch ME
        warning('Could not remove AMICA directory: %s', ME.message);
    end
end


function EEG = flag_and_remove_artifacts(EEG, logfile)
    % Flag and remove eye/muscle artifacts based on ICLabel
    
    if ~isfield(EEG.etc, 'ic_classification') || ...
       ~isfield(EEG.etc.ic_classification, 'ICLabel')
        EEG = iclabel(EEG);
        log_message(logfile, 'ICLabel applied.');
    end
    
    % Extract probabilities: [Brain Muscle Eye Heart LineNoise ChannelNoise Other]
    probs = EEG.etc.ic_classification.ICLabel.classifications;
    eyeProb = probs(:, 3);
    muscleProb = probs(:, 2);
    channelNoiseProb = probs(:, 6);

    % Flag ICs for removal if Eye, Muscle, or Channel Noise probability >= 0.8
    toRemove = find(eyeProb >= 0.8 | muscleProb >= 0.8 | channelNoiseProb >= 0.8);
    
    if isempty(toRemove)
        EEG.etc.badICs = [];
        log_message(logfile, 'No ICs flagged for removal.');
        return;
    end
    
    % Log which ICs are being removed
    for idx = toRemove'
        if eyeProb(idx) >= 0.9
            log_message(logfile, sprintf('Removing IC %d (Eye, p=%.2f)', idx, eyeProb(idx)));
        else
            log_message(logfile, sprintf('Removing IC %d (Muscle, p=%.2f)', idx, muscleProb(idx)));
        end
    end
    
    EEG.etc.badICs = toRemove;
    EEG = pop_subcomp(EEG, toRemove, 0);
    
    log_message(logfile, sprintf('Removed %d IC(s).', numel(toRemove)));

    % Log stats inside IC removal helper
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats post-subcomp: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
end


function qc = compute_qc_metrics(EEG, badLabels, logfile)
    % Compute comprehensive QC metrics
    
    qc = struct();
    
    % Bad channels
    qc.nBad = numel(badLabels);
    qc.badChannelLabels = badLabels;
    
    % ASR repair percentage
    if isfield(EEG.etc, 'clean_sample_mask')
        qc.percASRrepaired = 100 * mean(~EEG.etc.clean_sample_mask);
        qc.samplesRetained = sum(EEG.etc.clean_sample_mask);
        qc.totalSamples = length(EEG.etc.clean_sample_mask);
        qc.percSamplesRetained = 100 * qc.samplesRetained / qc.totalSamples;
    else
        qc.percASRrepaired = 0;
        qc.samplesRetained = EEG.pnts;
        qc.totalSamples = EEG.pnts;
        qc.percSamplesRetained = 100;
    end
    
    % ICs removed
    if isfield(EEG.etc, 'badICs')
        qc.ICsRemoved = numel(EEG.etc.badICs);
    else
        qc.ICsRemoved = 0;
    end
    
    % Event-wise retention (if events exist)
    if isfield(EEG, 'event') && ~isempty(EEG.event)
        qc.eventwiseRetention = compute_eventwise_retention(EEG);
    else
        qc.eventwiseRetention = struct();
        log_message(logfile, 'No events found - skipping event-wise retention.');
    end
    
    log_message(logfile, sprintf('QC: %d bad channels, %.1f%% ASR-repaired, %d ICs removed', ...
        qc.nBad, qc.percASRrepaired, qc.ICsRemoved));
end


function eventwiseRetention = compute_eventwise_retention(EEG)
    % Compute data retention percentage for each event type
    
    % Convert all event types to cell array of strings for consistent handling
    eventTypes = {EEG.event.type};
    if ~iscell(eventTypes{1})
        % If types are numeric, convert to strings
        eventTypes = cellfun(@num2str, eventTypes, 'UniformOutput', false);
    end
    
    uniqueTypes = unique(eventTypes);
    eventwiseRetention = struct();
    
    for i = 1:numel(uniqueTypes)
        type = uniqueTypes{i};
        indices = find(strcmp(eventTypes, type));
        latencies = round([EEG.event(indices).latency]);
        
        validLatencies = latencies(latencies > 0 & latencies <= length(EEG.etc.clean_sample_mask));
        
        if isempty(validLatencies)
            continue;
        end
        
        retained = EEG.etc.clean_sample_mask(validLatencies);
        
        % Use valid field names (replace invalid characters)
        fieldName = matlab.lang.makeValidName(sprintf('event_%s', num2str(type)));
        
        eventwiseRetention.(fieldName).nEvents = numel(validLatencies);
        eventwiseRetention.(fieldName).nKept = sum(retained);
        eventwiseRetention.(fieldName).percKept = 100 * sum(retained) / numel(validLatencies);
    end
end


function write_qc_report(qc_folder, participant_num, qc)
    % Write human-readable QC report
    
    fid = fopen(fullfile(qc_folder, sprintf('QC_P%02d.txt', participant_num)), 'w');
    
    fprintf(fid, '=== QC Report for P%02d ===\n\n', participant_num);
    
    fprintf(fid, 'Overall Metrics:\n');
    fprintf(fid, '  Samples retained: %.1f%% (%d / %d)\n', ...
        qc.percSamplesRetained, qc.samplesRetained, qc.totalSamples);
    fprintf(fid, '  Percent ASR-repaired: %.2f%%\n', qc.percASRrepaired);
    fprintf(fid, '  Bad channels: %d\n', qc.nBad);
    fprintf(fid, '  Bad channel labels: %s\n', strjoin(qc.badChannelLabels, ', '));
    fprintf(fid, '  ICs removed: %d\n\n', qc.ICsRemoved);
    
    if ~isempty(fieldnames(qc.eventwiseRetention))
        fprintf(fid, 'Event-wise Retention:\n');
        fields = fieldnames(qc.eventwiseRetention);
        for i = 1:length(fields)
            f = fields{i};
            fprintf(fid, '  %s: %.1f%% (%d / %d)\n', f, ...
                qc.eventwiseRetention.(f).percKept, ...
                qc.eventwiseRetention.(f).nKept, ...
                qc.eventwiseRetention.(f).nEvents);
        end
    end
    
    fclose(fid);
end


function log_message(logfile, message)
    % Log messages with timestamps
    fid = fopen(logfile, 'a');
    if fid == -1
        warning('Cannot open log file: %s', logfile);
        return;
    end
    fprintf(fid, '%s: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
    fclose(fid);
end


