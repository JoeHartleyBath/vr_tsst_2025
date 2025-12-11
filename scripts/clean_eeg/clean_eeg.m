function [EEG, qc] = clean_eeg(raw_set_path, output_folder, participant_num, vis_folder, qc_folder, config)
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
    if nargin < 6 || isempty(config)
        config = yaml.loadFile('config/general.yaml');
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
    
    log_message(logfile, sprintf('Loaded: %d channels, %d samples, %.1f Hz', ...
        EEG.nbchan, EEG.pnts, EEG.srate));
    save_visualization(EEG, vis_folder, sprintf('P%02d_01_raw_loaded.png', participant_num));
    
    % Store original channel locations before any processing
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    
    %% Step 2: Assign Channel Locations
    log_message(logfile, '=== Step 2: Assign Channel Locations ===');
    
    % Assign standard channel locations for the 128 EEG channels
    EEG = pop_chanedit(EEG, 'lookup', ...
        'C:\Users\Joe\Documents\MATLAB\eeglab_current\eeglab2024.0\sample_locs\NA-271.elc');
    
    if isempty(EEG.chanlocs) || isempty(EEG.chanlocs(1).X)
        warning('Failed to assign channel locations.');
    else
        log_message(logfile, 'Channel locations assigned.');
    end
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_02_chanlocs.png', participant_num));
    
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
    
    % Remove 50 Hz line noise
    EEG = pop_cleanline(EEG, 'linefreqs', 50, 'sigtype', 'Channels');
    log_message(logfile, 'Applied CleanLine (50 Hz).');
    
    % Apply 25 Hz notch for participants 1-7
    if ismember(participant_num, 1:7)
        EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
        log_message(logfile, 'Applied 25 Hz notch filter.');
    end
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_03_basic_clean.png', participant_num));
    
    %% Step 4: Advanced Cleaning (ASR + ICA)
    log_message(logfile, '=== Step 4: Advanced Cleaning ===');
    
    % Run clean_artifacts (ASR)
    [EEG, com] = clean_artifacts(EEG, ...
        'FlatlineCriterion',  5, ...
        'ChannelCriterion',   0.85, ...
        'LineNoiseCriterion', 4, ...
        'BurstCriterion',     20, ...
        'WindowCriterion',    0.8);
    
    log_message(logfile, 'clean_artifacts (ASR) completed.');
    
    % Validate clean_channel_mask exists
    if ~isfield(EEG.etc, 'clean_channel_mask')
        warning('clean_channel_mask not found - assuming all channels retained.');
        EEG.etc.clean_channel_mask = true(1, length(EEG.etc.orig_chanlocs));
    end
    
    % Identify bad channels
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
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_04_after_asr.png', participant_num));
    
    %% Step 5: Run AMICA (ICA)
    log_message(logfile, '=== Step 5: Run AMICA ===');
    
    [EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile);
    
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
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_05_after_amica.png', participant_num));
    
    %% Step 6: Apply ICLabel and Remove Artifacts
    log_message(logfile, '=== Step 6: ICLabel and Artifact Removal ===');
    
    EEG = iclabel(EEG);
    log_message(logfile, sprintf('ICLabel applied. %d components classified.', size(EEG.icaweights, 1)));
    
    EEG = flag_and_remove_artifacts(EEG, logfile);
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_06_after_artifact_removal.png', participant_num));
    
    %% Step 7: Interpolate Bad Channels and Re-reference
    log_message(logfile, '=== Step 7: Interpolate and Re-reference ===');
    
    EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
    log_message(logfile, 'Bad channels interpolated.');
    
    EEG = pop_reref(EEG, []);
    log_message(logfile, sprintf('Data re-referenced to average. Mean: %.4f', mean(EEG.data(:))));
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_07_final_clean.png', participant_num));
    
    %% Step 8: Compute QC Metrics
    log_message(logfile, '=== Step 8: QC Metrics ===');
    
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
    save(cleaned_mat_path, 'cleaned_EEG', '-v7.3');
    log_message(logfile, sprintf('Cleaned EEG matrix saved: %s', cleaned_mat_path));
    
    % Save as .set (full EEGLAB structure)
    cleaned_set_path = fullfile(output_folder, sprintf('P%02d_cleaned.set', participant_num));
    EEG.setname = sprintf('P%02d_cleaned', participant_num);
    pop_saveset(EEG, 'filename', sprintf('P%02d_cleaned.set', participant_num), ...
        'filepath', output_folder);
    log_message(logfile, sprintf('Cleaned EEG .set saved: %s', cleaned_set_path));
    
    log_message(logfile, '=== CLEANING PIPELINE COMPLETE ===');
end


%% ========== HELPER FUNCTIONS ==========

function [EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile)
    % Run AMICA with log-likelihood tracking
    
    num_models   = 1;
    numprocs     = 1;
    max_threads  = 4;
    max_iter     = 400;
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
    
    toRemove = find(eyeProb >= 0.9 | muscleProb >= 0.9);
    
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
    
    uniqueTypes = unique({EEG.event.type});
    eventwiseRetention = struct();
    
    for i = 1:numel(uniqueTypes)
        type = uniqueTypes{i};
        indices = find(strcmp({EEG.event.type}, type));
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


function save_visualization(EEG, folder, filename)
    % Save EEG visualization to PNG
    if ~exist(folder, 'dir')
        mkdir(folder);
    end
    
    try
        figure('Visible', 'off');
        pop_eegplot(EEG, 1, 1, 1);
        saveas(gcf, fullfile(folder, filename));
        close(gcf);
    catch ME
        warning('Could not save visualization %s: %s', filename, ME.message);
    end
end
