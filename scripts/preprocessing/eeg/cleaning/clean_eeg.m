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
        % Load config with fallback support (ReadYaml → SimpleYAML → skip)
        projectRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..');
        projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
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
    save_visualization(EEG, vis_folder, sprintf('P%02d_01_raw_loaded.png', participant_num));
    
    %% Step 2: Assign Channel Locations
    log_message(logfile, '=== Step 2: Assign Channel Locations ===');

    % Assign ANT Neuro 128-channel equidistant layout (resolve via project root)
    projectRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..');
    projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
    chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
    % First attempt: lookup by name
    try
        EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);
    catch ME
        log_message(logfile, sprintf('pop_chanedit lookup failed: %s', ME.message));
    end

    % Verify that 3D coordinates were populated; if not, attempt robust force-assignment
    coords_ok = false;
    if ~isempty(EEG.chanlocs) && isfield(EEG.chanlocs(1), 'X') && ~isempty(EEG.chanlocs(1).X)
        coords_ok = true;
    end

    if ~coords_ok
        log_message(logfile, 'Channel locations incomplete after lookup; attempting forced assignment from NA-271.elc (label-normalized matching).');
        try
            locs = readlocs(chanlocs_file);
            % Normalize labels to alphanumeric lowercase for tolerant matching
            norm = @(s) lower(regexprep(strtrim(s), '[^a-z0-9]', ''));
            loc_labels = cellfun(norm, {locs.labels}, 'UniformOutput', false);

            assigned_idx = false(1, numel(EEG.chanlocs));
            for ci = 1:numel(EEG.chanlocs)
                lbl = EEG.chanlocs(ci).labels;
                if isempty(lbl), continue; end
                nlbl = norm(lbl);
                match = find(strcmp(nlbl, loc_labels), 1);
                if ~isempty(match)
                    L = locs(match);
                    EEG.chanlocs(ci).X = L.X;
                    EEG.chanlocs(ci).Y = L.Y;
                    EEG.chanlocs(ci).Z = L.Z;
                    if isfield(L, 'theta'), EEG.chanlocs(ci).theta = L.theta; end
                    if isfield(L, 'radius'), EEG.chanlocs(ci).radius = L.radius; end
                    if isfield(L, 'sph_theta'), EEG.chanlocs(ci).sph_theta = L.sph_theta; end
                    if isfield(L, 'sph_phi'), EEG.chanlocs(ci).sph_phi = L.sph_phi; end
                    assigned_idx(ci) = true;
                end
            end
            assigned = sum(assigned_idx);
            total = numel(EEG.chanlocs);
            if assigned > 0
                log_message(logfile, sprintf('Force-assigned NA-271 coords to %d/%d channels (using normalized label matching).', assigned, total));
                coords_ok = true;
            else
                log_message(logfile, 'Forced assignment found no matches between data labels and NA-271 labels.');
            end

            % If many remain unmatched, log sample of unmatched labels for diagnosis
            if assigned/total < 0.75
                unmatched = {EEG.chanlocs(~assigned_idx).labels};
                nshow = min(10, numel(unmatched));
                log_message(logfile, sprintf('WARNING: only %d/%d channels matched; sample unmatched: %s', assigned, total, strjoin(unmatched(1:nshow), ', ')));
            end
        catch ME
            log_message(logfile, sprintf('Forced assignment failed: %s', ME.message));
        end
    else
        log_message(logfile, 'Channel locations assigned via lookup.');
    end

    % Guarantee spherical coordinates for all channels with valid X/Y/Z
    for ci = 1:numel(EEG.chanlocs)
        c = EEG.chanlocs(ci);
        if ~isempty(c.X) && ~isempty(c.Y) && ~isempty(c.Z)
            % Only fill if missing or NaN
            if ~isfield(c,'theta') || isempty(c.theta) || isnan(c.theta) || ...
               ~isfield(c,'radius') || isempty(c.radius) || isnan(c.radius) || ...
               ~isfield(c,'sph_theta') || isempty(c.sph_theta) || isnan(c.sph_theta) || ...
               ~isfield(c,'sph_phi') || isempty(c.sph_phi) || isnan(c.sph_phi)
                [az,el,r] = cart2sph(c.X, c.Y, c.Z);
                EEG.chanlocs(ci).theta = rad2deg(az);
                EEG.chanlocs(ci).radius = r;
                EEG.chanlocs(ci).sph_theta = rad2deg(az);
                EEG.chanlocs(ci).sph_phi = rad2deg(el);
            end
        end
    end

    % Re-check and warn if still not all channels have valid spherical coords
    hasSph = arrayfun(@(c) isfield(c,'theta') && ~isempty(c.theta) && ~isnan(c.theta), EEG.chanlocs);
    if sum(hasSph) < numel(EEG.chanlocs)
        log_message(logfile, sprintf('WARNING: Only %d/%d channels have valid spherical coordinates after all assignment steps.', sum(hasSph), numel(EEG.chanlocs)));
    end

    if ~coords_ok
        log_message(logfile, 'WARNING: Channel location coordinates not found after all fallbacks; proceeding with location-free cleaning.');
    end

    % Write detailed channel-location diagnostic to file for debugging
    try
        diag_path = fullfile(output_folder, sprintf('P%02d_chanlocs_diag.txt', participant_num));
        fid = fopen(diag_path, 'w');
        fprintf(fid, 'Channel location diagnostic for P%02d\n', participant_num);
        fprintf(fid, 'NA-271 file: %s\n', chanlocs_file);
        if exist(chanlocs_file, 'file')
            fprintf(fid, 'NA-271 exists: YES\n');
            try
                locs_sample = readlocs(chanlocs_file);
                nlocs = numel(locs_sample);
                fprintf(fid, 'NA-271 entries: %d (sample first 10):\n', nlocs);
                for ii = 1:min(10,nlocs)
                    fprintf(fid, '  %s\n', locs_sample(ii).labels);
                end
            catch
                fprintf(fid, 'Could not read NA-271 via readlocs.\n');
            end
        else
            fprintf(fid, 'NA-271 exists: NO\n');
        end

        % Per-channel XYZ presence
        fprintf(fid, '\nPer-channel XYZ presence (label, hasXYZ  X Y Z):\n');
        for ci = 1:numel(EEG.chanlocs)
            ch = EEG.chanlocs(ci);
            hasX = ~isempty(ch.X);
            hasY = ~isempty(ch.Y);
            hasZ = ~isempty(ch.Z);
            fprintf(fid, '%s, %d, %s %s %s\n', ch.labels, hasX && hasY && hasZ, num2str(ch.X), num2str(ch.Y), num2str(ch.Z));
        end

        % Report matched/unmatched summary if we attempted forced assignment
        if exist('assigned_idx','var')
            assigned = sum(assigned_idx);
            total = numel(assigned_idx);
            fprintf(fid, '\nForced assignment summary: %d/%d matched\n', assigned, total);
            if assigned < total
                fprintf(fid, 'Sample unmatched labels:\n');
                unmatched = {EEG.chanlocs(~assigned_idx).labels};
                for ii = 1:min(50, numel(unmatched))
                    fprintf(fid, '  %s\n', unmatched{ii});
                end
            end
        end
        fclose(fid);
        log_message(logfile, sprintf('Wrote channel-location diagnostic: %s', diag_path));
    catch ME
        log_message(logfile, sprintf('Could not write chanlocs diagnostic: %s', ME.message));
    end

    % Persist original chanlocs after template assignment for later interpolation/saving
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    
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
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_03_basic_clean.png', participant_num));
    
    %% Step 4: Advanced Cleaning (ASR + ICA)
    log_message(logfile, '=== Step 4: Advanced Cleaning ===');

    % === DIAGNOSTIC: Check channel locations before ASR ===
    nChans = length(EEG.chanlocs);
    nMissingLocs = 0;
    missingLocLabels = {};
    for ci = 1:nChans
        c = EEG.chanlocs(ci);
        if ~isfield(c,'X') || isempty(c.X) || isnan(c.X) || ...
           ~isfield(c,'Y') || isempty(c.Y) || isnan(c.Y) || ...
           ~isfield(c,'Z') || isempty(c.Z) || isnan(c.Z)
            nMissingLocs = nMissingLocs + 1;
            missingLocLabels{end+1} = c.labels;
        end
    end
    log_message(logfile, sprintf('[PRE-ASR DIAG] Channels: %d, missing locations: %d', nChans, nMissingLocs));
    if nMissingLocs > 0
        log_message(logfile, sprintf('[PRE-ASR DIAG] Channels missing locations: %s', strjoin(missingLocLabels, ', ')));
    end

    % === TEST MODE: Use only a small subset of data for fast failure (first 30 seconds) ===
    test_mode = true; % Set to false for full run
    if test_mode
        log_message(logfile, '[TEST MODE] Using only first 30 seconds of data for fast failure.');
        srate = EEG.srate;
        nTestSamples = min(size(EEG.data,2), round(30 * srate));
        EEG.data = EEG.data(:,1:nTestSamples);
        EEG.pnts = nTestSamples;
        EEG.xmax = EEG.xmin + (nTestSamples-1)/srate;
        if isfield(EEG, 'event') && ~isempty(EEG.event)
            % Remove events outside this window
            EEG.event = EEG.event([EEG.event.latency] <= nTestSamples);
        end
    end

    % ASR calibration: using default behavior (let clean_artifacts auto-select clean data)
    % Run clean_artifacts (ASR) on (possibly subset) data with tuned parameters
    [EEG, com] = clean_artifacts(EEG, ...
        'FlatlineCriterion',  5, ...
        'ChannelCriterion',   0.70, ...
        'LineNoiseCriterion', 4, ...
        'BurstCriterion',     50, ...
        'WindowCriterion',    0.60);

    log_message(logfile, 'clean_artifacts (ASR) completed with tuned parameters (Burst=50, Channel=0.70, Window=0.60).');

    % Log stats after ASR
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after ASR: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    % Validate clean_channel_mask exists
    if ~isfield(EEG.etc, 'clean_channel_mask')
        warning('clean_channel_mask not found - assuming all channels retained.');
        EEG.etc.clean_channel_mask = true(1, length(EEG.etc.orig_chanlocs));
    end

        % Identify bad channels with full diagnostics
        mask = EEG.etc.clean_channel_mask;
        origLocs = EEG.etc.orig_chanlocs;

        log_message(logfile, sprintf('[DIAG] Length of orig_chanlocs: %d', length(origLocs)));
        log_message(logfile, sprintf('[DIAG] Length of clean_channel_mask: %d', length(mask)));
        if isfield(EEG, 'chanlocs')
            log_message(logfile, sprintf('[DIAG] Length of EEG.chanlocs: %d', length(EEG.chanlocs)));
        end

        % Print mask summary
        mask_str = sprintf('%d', mask);
        log_message(logfile, sprintf('[DIAG] clean_channel_mask: %s', mask_str));

        % Print all original channel labels
        allLabels = {origLocs.labels};
        log_message(logfile, sprintf('[DIAG] orig_chanlocs labels: %s', strjoin(allLabels, ', ')));

        if length(origLocs) ~= length(mask)
            log_message(logfile, '[ERROR] Mismatch between orig_chanlocs and clean_channel_mask lengths!');
            error('Mismatch between orig_chanlocs (%d) and clean_channel_mask (%d).', ...
                length(origLocs), length(mask));
        end

        badIdx = ~mask;
        badLabels = {origLocs(badIdx).labels}';

        log_message(logfile, sprintf('[DIAG] badIdx: %s', mat2str(find(badIdx))));
        log_message(logfile, sprintf('[DIAG] badLabels: %s', strjoin(badLabels, ', ')));

        log_message(logfile, sprintf('%d bad channels identified: %s', ...
            sum(badIdx), strjoin(badLabels, ', ')));

    % === PATCH: After ASR, re-match chanlocs from orig_chanlocs by label ===
    % This ensures all remaining channels have correct coordinates for ICA/AMICA
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs) && isfield(EEG.etc, 'orig_chanlocs')
        orig_labels = {EEG.etc.orig_chanlocs.labels};
        for ci = 1:numel(EEG.chanlocs)
            lbl = EEG.chanlocs(ci).labels;
            idx = find(strcmp(lbl, orig_labels), 1);
            if ~isempty(idx)
                EEG.chanlocs(ci) = EEG.etc.orig_chanlocs(idx);
            end
        end
        % Guarantee spherical coordinates for all remaining channels
        for ci = 1:numel(EEG.chanlocs)
            c = EEG.chanlocs(ci);
            if ~isempty(c.X) && ~isempty(c.Y) && ~isempty(c.Z)
                if ~isfield(c,'theta') || isempty(c.theta) || isnan(c.theta) || ...
                   ~isfield(c,'radius') || isempty(c.radius) || isnan(c.radius) || ...
                   ~isfield(c,'sph_theta') || isempty(c.sph_theta) || isnan(c.sph_theta) || ...
                   ~isfield(c,'sph_phi') || isempty(c.sph_phi) || isnan(c.sph_phi)
                    [az,el,r] = cart2sph(c.X, c.Y, c.Z);
                    EEG.chanlocs(ci).theta = rad2deg(az);
                    EEG.chanlocs(ci).radius = r;
                    EEG.chanlocs(ci).sph_theta = rad2deg(az);
                    EEG.chanlocs(ci).sph_phi = rad2deg(el);
                end
            end
        end
    end

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

    % Log stats after AMICA unmixing applied
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after AMICA: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_05_after_amica.png', participant_num));
    
    %% Step 6: Apply ICLabel and Remove Artifacts
    log_message(logfile, '=== Step 6: ICLabel and Artifact Removal ===');
    
    EEG = iclabel(EEG);
    log_message(logfile, sprintf('ICLabel applied. %d components classified.', size(EEG.icaweights, 1)));
    
    EEG = flag_and_remove_artifacts(EEG, logfile);

    % Log stats after IC removal step
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after IC removal: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    
    save_visualization(EEG, vis_folder, sprintf('P%02d_06_after_artifact_removal.png', participant_num));

        %% Save pre-interp, pre-reref EEG with ICA weights for future reuse
        preinterp_path = fullfile(output_folder, sprintf('P%02d_preinterp_pre_reref_ICA.mat', participant_num));
        try
            save(preinterp_path, 'EEG', '-v7.3');
            log_message(logfile, sprintf('Saved pre-interp, pre-reref EEG with ICA: %s', preinterp_path));
        catch ME
            log_message(logfile, sprintf('WARNING: Could not save pre-interp, pre-reref EEG: %s', ME.message));
        end
    
    %% Step 7: Interpolate Bad Channels and Re-reference
    log_message(logfile, '=== Step 7: Interpolate and Re-reference ===');
    

    EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
    log_message(logfile, 'Bad channels interpolated.');


    % Exclude interpolated channels from rereferencing: use only original good channels (robust label matching)
    if isfield(EEG.etc, 'clean_channel_mask') && isfield(EEG.etc, 'orig_chanlocs')
        reref_mask = EEG.etc.clean_channel_mask;
        reref_labels = {EEG.etc.orig_chanlocs(reref_mask).labels};
        current_labels = {EEG.chanlocs.labels};
        % Normalize labels for robust matching
        norm = @(s) lower(regexprep(strtrim(s), '[^a-z0-9]', ''));
        reref_labels_norm = cellfun(norm, reref_labels, 'UniformOutput', false);
        current_labels_norm = cellfun(norm, current_labels, 'UniformOutput', false);
        reref_inds = find(ismember(current_labels_norm, reref_labels_norm));

        % Diagnostics: log all label pairs and indices
        log_message(logfile, '[REREF DIAG] reref_labels (original good):');
        for i = 1:length(reref_labels)
            log_message(logfile, sprintf('  %d: %s (norm: %s)', i, reref_labels{i}, reref_labels_norm{i}));
        end
        log_message(logfile, '[REREF DIAG] current_labels (post-interp):');
        for i = 1:length(current_labels)
            log_message(logfile, sprintf('  %d: %s (norm: %s)', i, current_labels{i}, current_labels_norm{i}));
        end
        log_message(logfile, sprintf('[REREF DIAG] reref_inds: %s', mat2str(reref_inds)));

        if ~isempty(reref_inds)
            EEG = pop_reref(EEG, reref_inds);
            log_message(logfile, sprintf('Data re-referenced to average of original good channels (%d channels, using positional arg). Mean: %.4f', length(reref_inds), mean(EEG.data(:))));
        else
            EEG = pop_reref(EEG, []);
            log_message(logfile, 'WARNING: Could not robustly match original good channels for rereferencing, used all channels.');
        end
    else
        EEG = pop_reref(EEG, []);
        log_message(logfile, 'WARNING: clean_channel_mask or orig_chanlocs missing, used all channels for rereferencing.');
    end

    % Log stats after interpolation + reref
    stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
    log_message(logfile, sprintf('Stats after reref: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));

    save_visualization(EEG, vis_folder, sprintf('P%02d_07_final_clean.png', participant_num));

    %% FAIL-FAST: Check ICA weights after rereferencing
    if isempty(EEG.icaweights) || isempty(EEG.icasphere)
        log_message(logfile, 'FAIL-FAST: ICA weights or sphere missing after rereferencing. Aborting.');
        error('FAIL-FAST: ICA weights or sphere missing after rereferencing.');
    else
        log_message(logfile, 'PASS: ICA weights and sphere present after rereferencing.');
    end
    
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
    
    % Save as .mat (data matrix and full EEGLAB `EEG` struct so ICA weights/sphere
    % and ICLabel classifications are persisted for downstream reproducibility)
    cleaned_mat_path = fullfile(output_folder, sprintf('P%02d_cleaned.mat', participant_num));
    cleaned_EEG = double(EEG.data);
    stats = [min(cleaned_EEG(:)), max(cleaned_EEG(:)), mean(cleaned_EEG(:)), std(cleaned_EEG(:))];
    log_message(logfile, sprintf('Stats at save: min=%.6f max=%.6f mean=%.6f std=%.6f', stats));
    % Save both the raw cleaned matrix and the full EEG struct (EEG contains
    % `icaweights`, `icasphere`, and `etc.ic_classification` from ICLabel).
    try
        save(cleaned_mat_path, 'cleaned_EEG', 'EEG', '-v7.3');
        log_message(logfile, sprintf('Cleaned EEG matrix and EEG struct saved: %s', cleaned_mat_path));
    catch ME
        % Fall back to saving the cleaned matrix only if saving the struct fails
        warning('Saving EEG struct failed: %s', ME.message);
        save(cleaned_mat_path, 'cleaned_EEG', '-v7.3');
        log_message(logfile, sprintf('Cleaned EEG matrix saved (EEG struct not saved): %s', cleaned_mat_path));
    end
    
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

function [EEG, LL_trace] = run_amica_pipeline(EEG, participant_num, logfile)
    % Run AMICA with log-likelihood tracking
    
    num_models   = 1;
    numprocs     = 1; 
    max_threads  = 2;       
    max_iter     = 10; % Reduced for fail-fast test mode
    writeStep    = 10;
    
    % Place AMICA outputs in the project's output/_stale folder to avoid
    % interfering with the scripts folder and to preserve outputs for reuse.
    projectRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..');
    projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
    stale_dir = fullfile(projectRoot, 'output', '_stale');
    if ~exist(stale_dir, 'dir')
        mkdir(stale_dir);
    end
    outdir_base = fullfile(stale_dir, sprintf('amicaouttmp_%d', participant_num));
    outdir = outdir_base;
    % If an outdir already exists (possibly from a previous run), create a
    % unique folder by appending a timestamp to avoid concurrent-write errors.
    if exist(outdir, 'dir')
        timestamp = datestr(now, 'yyyymmddTHHMMSS');
        outdir = [outdir_base '_' timestamp];
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
    
    % NOTE: Do NOT remove the AMICA output directory here. Keep AMICA
    % outputs (weights, LL traces, input.param) on disk under
    % `output/_stale` for reproducibility and potential reuse. If you need
    % to clean old directories, do so manually or with a separate cleanup
    % utility.
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

    % Debug: log summary statistics for ICLabel probabilities to detect issues
    maxEye = max(eyeProb);
    maxMuscle = max(muscleProb);
    meanEye = mean(eyeProb);
    meanMuscle = mean(muscleProb);
    log_message(logfile, sprintf('ICLabel probs summary: eye_max=%.3f eye_mean=%.3f muscle_max=%.3f muscle_mean=%.3f', ...
        maxEye, meanEye, maxMuscle, meanMuscle));

    % Log top 3 components by eye probability for inspection
    [sortedEye, idxEye] = sort(eyeProb, 'descend');
    topN = min(3, numel(idxEye));
    for ii = 1:topN
        log_message(logfile, sprintf('ICLabel top_eye %d -> p=%.3f', idxEye(ii), sortedEye(ii)));
    end

    % Also log top 3 by muscle probability
    [sortedMus, idxMus] = sort(muscleProb, 'descend');
    topN = min(3, numel(idxMus));
    for ii = 1:topN
        log_message(logfile, sprintf('ICLabel top_muscle %d -> p=%.3f', idxMus(ii), sortedMus(ii)));
    end

    % Flag components above thresholds (eye or muscle)
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


