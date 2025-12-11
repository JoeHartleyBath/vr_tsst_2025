% TEST_CLEAN_EEG - Test the clean_eeg function with synthetic data
%
% This script creates a minimal synthetic EEG dataset and tests the
% clean_eeg pipeline to ensure all steps execute without errors.

% Initialize EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

% Create test directories
test_output_folder = fullfile(pwd, 'test_output', 'cleaned_eeg');
test_vis_folder = fullfile(pwd, 'test_output', 'vis', 'test');
test_qc_folder = fullfile(pwd, 'test_output', 'qc');

if ~exist(test_output_folder, 'dir')
    mkdir(test_output_folder);
end
if ~exist(test_vis_folder, 'dir')
    mkdir(test_vis_folder);
end
if ~exist(test_qc_folder, 'dir')
    mkdir(test_qc_folder);
end

%% Create Synthetic EEG Data
fprintf('=== Creating Synthetic EEG Data ===\n');

% Parameters
nbchan = 128;
pnts = 125 * 60 * 5;  % 5 minutes at 125 Hz
srate = 125;
trials = 1;

% Generate synthetic data (random noise + some sine waves)
EEG = eeg_emptyset();
EEG.data = randn(nbchan, pnts) * 10;  % Random noise baseline

% Add some synthetic oscillations
for ch = 1:nbchan
    t = (0:pnts-1) / srate;
    % Alpha band (10 Hz)
    EEG.data(ch, :) = EEG.data(ch, :) + 5 * sin(2 * pi * 10 * t);
    % Theta band (6 Hz)
    EEG.data(ch, :) = EEG.data(ch, :) + 3 * sin(2 * pi * 6 * t);
end

EEG.setname = 'test_raw';
EEG.nbchan = nbchan;
EEG.pnts = pnts;
EEG.srate = srate;
EEG.trials = trials;
EEG.xmin = 0;
EEG.xmax = (pnts - 1) / srate;
EEG.times = (0:pnts-1) / srate * 1000;  % in ms

% Create channel labels (A1-A64, B1-B64)
for i = 1:64
    EEG.chanlocs(i).labels = sprintf('A%d', i);
end
for i = 1:64
    EEG.chanlocs(64 + i).labels = sprintf('B%d', i);
end

% Add some synthetic events
n_events = 10;
event_times = linspace(1000, pnts - 1000, n_events);
for i = 1:n_events
    EEG.event(i).type = randi([101, 104]);  % Random condition codes
    EEG.event(i).latency = round(event_times(i));
    EEG.event(i).duration = 0;
    EEG.urevent(i).type = EEG.event(i).type;
    EEG.urevent(i).latency = EEG.event(i).latency;
end

EEG = eeg_checkset(EEG);

fprintf('Created synthetic EEG: %d channels, %d samples, %.1f Hz\n', ...
    EEG.nbchan, EEG.pnts, EEG.srate);
fprintf('Added %d synthetic events\n', length(EEG.event));

%% Save Synthetic Raw .set File
raw_set_path = fullfile(test_output_folder, 'test_raw.set');
pop_saveset(EEG, 'filename', 'test_raw.set', 'filepath', test_output_folder);
fprintf('Saved synthetic raw .set: %s\n\n', raw_set_path);

%% Run clean_eeg Pipeline
fprintf('=== Running clean_eeg Pipeline ===\n');

try
    [EEG_cleaned, qc] = clean_eeg(raw_set_path, ...
                                   test_output_folder, ...
                                   99, ...  % Test participant number
                                   test_vis_folder, ...
                                   test_qc_folder, ...
                                   []);  % Use default config
    
    fprintf('\n=== Pipeline Completed Successfully ===\n');
    
    % Display results
    fprintf('\nCleaned EEG Stats:\n');
    fprintf('  Channels: %d\n', EEG_cleaned.nbchan);
    fprintf('  Samples: %d\n', EEG_cleaned.pnts);
    fprintf('  Sampling rate: %.1f Hz\n', EEG_cleaned.srate);
    fprintf('  Events: %d\n', length(EEG_cleaned.event));
    
    fprintf('\nQC Metrics:\n');
    fprintf('  Bad channels: %d\n', qc.nBad);
    fprintf('  Samples retained: %.1f%%\n', qc.percSamplesRetained);
    fprintf('  ASR repaired: %.2f%%\n', qc.percASRrepaired);
    fprintf('  ICs removed: %d\n', qc.ICsRemoved);
    
    % Check output files exist
    fprintf('\nOutput Files:\n');
    mat_path = fullfile(test_output_folder, 'P99_cleaned.mat');
    set_path = fullfile(test_output_folder, 'P99_cleaned.set');
    qc_mat_path = fullfile(test_qc_folder, 'P99_qc.mat');
    qc_txt_path = fullfile(test_qc_folder, 'QC_P99.txt');
    
    if exist(mat_path, 'file')
        fprintf('  ✓ .mat file created: %s\n', mat_path);
    else
        fprintf('  ✗ .mat file missing\n');
    end
    
    if exist(set_path, 'file')
        fprintf('  ✓ .set file created: %s\n', set_path);
    else
        fprintf('  ✗ .set file missing\n');
    end
    
    if exist(qc_mat_path, 'file')
        fprintf('  ✓ QC .mat created: %s\n', qc_mat_path);
    else
        fprintf('  ✗ QC .mat missing\n');
    end
    
    if exist(qc_txt_path, 'file')
        fprintf('  ✓ QC report created: %s\n', qc_txt_path);
    else
        fprintf('  ✗ QC report missing\n');
    end
    
    fprintf('\n=== TEST PASSED ===\n');
    
catch ME
    fprintf('\n=== TEST FAILED ===\n');
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end
