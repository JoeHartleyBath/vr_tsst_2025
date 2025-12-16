% Compare amplitudes: original vs cleaned
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

% Load original
fprintf('Loading ORIGINAL P01.set...\n');
EEG_orig = pop_loadset('filename', 'P01.set', 'filepath', 'C:/vr_tsst_2025/output/sets');
fprintf('Original: %d channels × %d samples\n', EEG_orig.nbchan, EEG_orig.pnts);
fprintf('  Min: %.3f µV\n', min(EEG_orig.data(:)));
fprintf('  Max: %.3f µV\n', max(EEG_orig.data(:)));
fprintf('  Mean: %.3f µV\n', mean(EEG_orig.data(:)));
fprintf('  Std: %.3f µV\n', std(EEG_orig.data(:)));

% Load cleaned
fprintf('\nLoading CLEANED P01_cleaned.set...\n');
EEG_clean = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
fprintf('Cleaned: %d channels × %d samples\n', EEG_clean.nbchan, EEG_clean.pnts);
fprintf('  Min: %.6f µV\n', min(EEG_clean.data(:)));
fprintf('  Max: %.6f µV\n', max(EEG_clean.data(:)));
fprintf('  Mean: %.6f µV\n', mean(EEG_clean.data(:)));
fprintf('  Std: %.6f µV\n', std(EEG_clean.data(:)));

fprintf('\nAmplitude ratio (original/cleaned): %.0fx\n', std(EEG_orig.data(:)) / std(EEG_clean.data(:)));
