% Debug why region matching fails in feature extraction
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
addpath('C:/MATLAB/toolboxes/yamlmatlab');
eeglab nogui;

% Load config
cfg = yaml.ReadYaml('C:/vr_tsst_2025/config/eeg_feature_extraction.yaml');

% Load cleaned data
EEG = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');

fprintf('Dataset has %d channels\n', EEG.nbchan);
fprintf('First 10 channel labels from EEG.chanlocs:\n');
chan_labels = {EEG.chanlocs.labels};
for i=1:min(10, length(chan_labels))
    fprintf('  %d: "%s" (class: %s)\n', i, chan_labels{i}, class(chan_labels{i}));
end

fprintf('\nFrontalLeft from YAML (first 5):\n');
fl_chans = cfg.regions.FrontalLeft;
for i=1:min(5, length(fl_chans))
    fprintf('  %d: "%s" (class: %s)\n', i, fl_chans{i}, class(fl_chans{i}));
end

fprintf('\nTesting ismember:\n');
test_match = ismember(chan_labels, fl_chans);
fprintf('  Number of matches: %d\n', sum(test_match));
fprintf('  Matched indices: ');
fprintf('%d ', find(test_match));
fprintf('\n');

fprintf('\nManual check - is L1 in chan_labels?\n');
fprintf('  strcmp result for position 7: %d\n', strcmp(chan_labels{7}, 'L1'));
fprintf('  Direct comparison: "%s" == "L1"?\n', chan_labels{7});

fprintf('\nChecking for whitespace issues:\n');
fprintf('  Length of chan_labels{7}: %d\n', length(chan_labels{7}));
fprintf('  Length of L1: %d\n', length('L1'));
fprintf('  Bytes in chan_labels{7}: ');
fprintf('%d ', uint8(chan_labels{7}));
fprintf('\n');
