% TEST_EACH_STEP_AUTO - Run cleaning pipeline step-by-step (automated)
%
% Non-interactive version that runs all steps automatically

% Add utilities path
addpath('scripts/utils');

% Initialize EEGLAB
[~, ~, ~, ~] = eeglab;

%% Configuration
% Use subset for quick testing (change to P01.set for full run)
input_file = 'output/processed/P01_subset_5min.set';
participant_num = 1;

% Create test output folder
test_folder = 'output/step_by_step_test';
if ~exist(test_folder, 'dir')
    mkdir(test_folder);
end

fprintf('=============================================================\n');
fprintf('STEP-BY-STEP EEG CLEANING TEST (AUTOMATED)\n');
fprintf('=============================================================\n');
fprintf('Input file: %s\n', input_file);
fprintf('Test output: %s\n', test_folder);
fprintf('=============================================================\n\n');

%% Check if file exists
if ~exist(input_file, 'file')
    error('Input file not found: %s', input_file);
end

%% STEP 0: Load Raw Data
fprintf('\n=============================================================\n');
fprintf('STEP 0: LOAD RAW DATA\n');
fprintf('=============================================================\n');

[filepath, filename, ext] = fileparts(input_file);
EEG = pop_loadset('filename', [filename, ext], 'filepath', filepath);

fprintf('Loaded successfully:\n');
fprintf('  Channels: %d\n', EEG.nbchan);
fprintf('  Samples: %d\n', EEG.pnts);
fprintf('  Sampling rate: %.1f Hz\n', EEG.srate);
fprintf('  Duration: %.1f seconds\n', EEG.pnts / EEG.srate);
fprintf('  Events: %d\n', length(EEG.event));

if ~isempty(EEG.event)
    unique_events = unique([EEG.event.type]);
    fprintf('  Unique event types: %s\n', mat2str(unique_events));
end

EEG_original = EEG;
pop_saveset(EEG, 'filename', sprintf('step0_loaded_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 1: Assign Channel Locations
fprintf('\n=============================================================\n');
fprintf('STEP 1: ASSIGN CHANNEL LOCATIONS\n');
fprintf('=============================================================\n');

if ~isempty(EEG.chanlocs)
    EEG.etc.orig_chanlocs = EEG.chanlocs;
    fprintf('Original channel locations stored.\n');
end

EEG = pop_chanedit(EEG, 'lookup', ...
    'C:\Users\Joe\Documents\MATLAB\eeglab_current\eeglab2024.0\sample_locs\NA-271.elc');

if isempty(EEG.chanlocs) || isempty(EEG.chanlocs(1).X)
    warning('Failed to assign channel locations!');
else
    fprintf('Channel locations assigned successfully.\n');
    fprintf('  Channel 1: %s at (%.2f, %.2f, %.2f)\n', ...
        EEG.chanlocs(1).labels, EEG.chanlocs(1).X, EEG.chanlocs(1).Y, EEG.chanlocs(1).Z);
end

pop_saveset(EEG, 'filename', sprintf('step1_chanlocs_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 2: Resample to 125 Hz
fprintf('\n=============================================================\n');
fprintf('STEP 2: RESAMPLE TO 125 HZ\n');
fprintf('=============================================================\n');

original_srate = EEG.srate;
original_pnts = EEG.pnts;

if EEG.srate ~= 125
    fprintf('Resampling from %.1f Hz to 125 Hz...\n', EEG.srate);
    EEG = pop_resample(EEG, 125);
    fprintf('Resampling complete:\n');
    fprintf('  Original: %.1f Hz, %d samples\n', original_srate, original_pnts);
    fprintf('  New: %.1f Hz, %d samples\n', EEG.srate, EEG.pnts);
else
    fprintf('Already at 125 Hz - no resampling needed.\n');
end

pop_saveset(EEG, 'filename', sprintf('step2_resampled_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 3: Band-pass Filter 1-49 Hz
fprintf('\n=============================================================\n');
fprintf('STEP 3: BAND-PASS FILTER 1-49 HZ\n');
fprintf('=============================================================\n');

data_before = EEG.data(:, 1:min(1000, EEG.pnts));
fprintf('Applying 1-49 Hz band-pass filter...\n');
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
data_after = EEG.data(:, 1:min(1000, EEG.pnts));

fprintf('Filter applied successfully.\n');
fprintf('  Data range before: %.2f to %.2f µV\n', min(data_before(:)), max(data_before(:)));
fprintf('  Data range after: %.2f to %.2f µV\n', min(data_after(:)), max(data_after(:)));

pop_saveset(EEG, 'filename', sprintf('step3_bandpass_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 4: CleanLine (50 Hz line noise removal)
fprintf('\n=============================================================\n');
fprintf('STEP 4: CLEANLINE (50 HZ LINE NOISE REMOVAL)\n');
fprintf('=============================================================\n');

fprintf('Applying CleanLine for 50 Hz line noise...\n');
EEG = pop_cleanline(EEG, 'linefreqs', 50, 'sigtype', 'Channels');
fprintf('CleanLine applied successfully.\n');

pop_saveset(EEG, 'filename', sprintf('step4_cleanline_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 5: 25 Hz Notch Filter (P01-P07 only)
fprintf('\n=============================================================\n');
fprintf('STEP 5: 25 HZ NOTCH FILTER (P01-P07 ONLY)\n');
fprintf('=============================================================\n');

if ismember(participant_num, 1:7)
    fprintf('Applying 25 Hz notch filter for P%02d...\n', participant_num);
    EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
    fprintf('25 Hz notch applied.\n');
else
    fprintf('Participant P%02d does not need 25 Hz notch - skipping.\n', participant_num);
end

pop_saveset(EEG, 'filename', sprintf('step5_notch_P%02d.set', participant_num), 'filepath', test_folder);
fprintf('\n*** BASIC CLEANING COMPLETE ***\n\n');

%% STEP 6: Clean Artifacts (ASR)
fprintf('\n=============================================================\n');
fprintf('STEP 6: CLEAN ARTIFACTS (ASR)\n');
fprintf('=============================================================\n');

channels_before = EEG.nbchan;
samples_before = EEG.pnts;

fprintf('Running clean_artifacts...\n');
[EEG, ~] = clean_artifacts(EEG, ...
    'FlatlineCriterion',  5, ...
    'ChannelCriterion',   0.85, ...
    'LineNoiseCriterion', 4, ...
    'BurstCriterion',     20, ...
    'WindowCriterion',    0.8);

fprintf('ASR complete.\n');
fprintf('  Channels: %d -> %d\n', channels_before, EEG.nbchan);
fprintf('  Samples: %d -> %d\n', samples_before, EEG.pnts);

if isfield(EEG.etc, 'clean_channel_mask')
    badIdx = ~EEG.etc.clean_channel_mask;
    if any(badIdx)
        badLabels = {EEG.etc.orig_chanlocs(badIdx).labels};
        fprintf('  Bad channels (%d): %s\n', sum(badIdx), strjoin(badLabels, ', '));
    else
        fprintf('  No bad channels detected.\n');
    end
end

if isfield(EEG.etc, 'clean_sample_mask')
    bad_samples = sum(~EEG.etc.clean_sample_mask);
    fprintf('  Bad samples: %d (%.2f%%)\n', bad_samples, ...
        100 * bad_samples / length(EEG.etc.clean_sample_mask));
end

pop_saveset(EEG, 'filename', sprintf('step6_asr_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 7: Run AMICA (ICA)
fprintf('\n=============================================================\n');
fprintf('STEP 7: RUN AMICA (ICA DECOMPOSITION)\n');
fprintf('=============================================================\n');

fprintf('Running AMICA (this may take several minutes)...\n');

outdir = fullfile(pwd, sprintf('amicaouttmp_test_%d', participant_num));
if exist(outdir, 'dir')
    rmdir(outdir, 's');
end
mkdir(outdir);

tic;
[weights, sphere, mods] = runamica15(EEG.data, ...
    'num_models',   1, ...
    'outdir',       outdir, ...
    'numprocs',     1, ...
    'max_threads',  4, ...
    'max_iter',     400, ...
    'write_LLt',    1, ...
    'writestep',    10);
elapsed = toc;

EEG.icaweights = weights;
EEG.icasphere = sphere;
EEG = eeg_checkset(EEG);

fprintf('AMICA complete in %.1f seconds (%.1f minutes).\n', elapsed, elapsed/60);
fprintf('  ICA components: %d\n', size(EEG.icaweights, 1));

if isfield(mods, 'LL') && ~isempty(mods.LL)
    fprintf('  Final log-likelihood: %.2f\n', mods.LL(end));
end

rmdir(outdir, 's');
pop_saveset(EEG, 'filename', sprintf('step7_amica_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 8: ICLabel Classification
fprintf('\n=============================================================\n');
fprintf('STEP 8: ICLABEL CLASSIFICATION\n');
fprintf('=============================================================\n');

fprintf('Running ICLabel...\n');
EEG = iclabel(EEG);
fprintf('ICLabel complete. Components classified: %d\n', size(EEG.icaweights, 1));

probs = EEG.etc.ic_classification.ICLabel.classifications;
labels = {'Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Channel Noise', 'Other'};

fprintf('\nTop 10 component classifications:\n');
fprintf('  %-4s  %-12s  %-6s\n', 'IC#', 'Class', 'Prob');
fprintf('  %-4s  %-12s  %-6s\n', '---', '------------', '------');

for ic = 1:min(10, size(probs, 1))
    [max_prob, max_idx] = max(probs(ic, :));
    fprintf('  %-4d  %-12s  %.2f\n', ic, labels{max_idx}, max_prob);
end

pop_saveset(EEG, 'filename', sprintf('step8_iclabel_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 9: Remove Artifact Components
fprintf('\n=============================================================\n');
fprintf('STEP 9: REMOVE ARTIFACT COMPONENTS\n');
fprintf('=============================================================\n');

eyeProb = probs(:, 3);
muscleProb = probs(:, 2);
toRemove = find(eyeProb >= 0.9 | muscleProb >= 0.9);

if isempty(toRemove)
    fprintf('No components flagged for removal (threshold: 0.9).\n');
    EEG.etc.badICs = [];
else
    fprintf('Components flagged for removal:\n');
    for idx = toRemove'
        if eyeProb(idx) >= 0.9
            fprintf('  IC %d: Eye (p=%.2f)\n', idx, eyeProb(idx));
        else
            fprintf('  IC %d: Muscle (p=%.2f)\n', idx, muscleProb(idx));
        end
    end
    fprintf('Removing %d component(s)...\n', length(toRemove));
    EEG.etc.badICs = toRemove;
    EEG = pop_subcomp(EEG, toRemove, 0);
    fprintf('Components removed.\n');
end

pop_saveset(EEG, 'filename', sprintf('step9_artifacts_removed_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 10: Interpolate Bad Channels
fprintf('\n=============================================================\n');
fprintf('STEP 10: INTERPOLATE BAD CHANNELS\n');
fprintf('=============================================================\n');

if isfield(EEG.etc, 'orig_chanlocs')
    n_original = length(EEG.etc.orig_chanlocs);
    n_current = EEG.nbchan;
    n_to_interpolate = n_original - n_current;
    
    fprintf('Interpolating bad channels...\n');
    fprintf('  Original channels: %d\n', n_original);
    fprintf('  Current channels: %d\n', n_current);
    fprintf('  Channels to interpolate: %d\n', n_to_interpolate);
    
    EEG = pop_interp(EEG, EEG.etc.orig_chanlocs, 'spherical');
    fprintf('Interpolation complete. Final channels: %d\n', EEG.nbchan);
else
    fprintf('No orig_chanlocs found - skipping interpolation.\n');
end

pop_saveset(EEG, 'filename', sprintf('step10_interpolated_P%02d.set', participant_num), 'filepath', test_folder);

%% STEP 11: Re-reference to Average
fprintf('\n=============================================================\n');
fprintf('STEP 11: RE-REFERENCE TO AVERAGE\n');
fprintf('=============================================================\n');

mean_before = mean(EEG.data(:));
fprintf('Re-referencing to average...\n');
EEG = pop_reref(EEG, []);
mean_after = mean(EEG.data(:));

fprintf('Re-referencing complete.\n');
fprintf('  Mean before: %.6f µV\n', mean_before);
fprintf('  Mean after: %.6f µV (should be ~0)\n', mean_after);

pop_saveset(EEG, 'filename', sprintf('step11_final_P%02d.set', participant_num), 'filepath', test_folder);

%% Final Summary
fprintf('\n=============================================================\n');
fprintf('CLEANING PIPELINE COMPLETE\n');
fprintf('=============================================================\n');
fprintf('All steps completed successfully!\n\n');

fprintf('Summary:\n');
fprintf('  Input file: %s\n', input_file);
fprintf('  Original: %d channels, %d samples, %.1f Hz\n', ...
    EEG_original.nbchan, EEG_original.pnts, EEG_original.srate);
fprintf('  Final: %d channels, %d samples, %.1f Hz\n', ...
    EEG.nbchan, EEG.pnts, EEG.srate);
fprintf('  Events preserved: %d\n', length(EEG.event));

if isfield(EEG.etc, 'clean_channel_mask')
    bad_ch = sum(~EEG.etc.clean_channel_mask);
    fprintf('  Bad channels interpolated: %d\n', bad_ch);
end

if isfield(EEG.etc, 'badICs')
    fprintf('  ICs removed: %d\n', length(EEG.etc.badICs));
end

fprintf('\nAll intermediate files saved to: %s\n', test_folder);
fprintf('=============================================================\n');
