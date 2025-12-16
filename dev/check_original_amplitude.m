% Quick amplitude check on raw P01.set (subset)
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
addpath(genpath('C:/MATLAB/toolboxes/eeglab2025.1.0/functions'));

EEG = pop_loadset('filename','P01.set','filepath','C:/vr_tsst_2025/output/sets');
EEG.data = EEG.data(:, 1:min(10000, size(EEG.data,2))); % first 10k samples
stats = [min(EEG.data(:)), max(EEG.data(:)), mean(EEG.data(:)), std(EEG.data(:))];
fprintf('Raw subset min/max/mean/std (uV): %.2f / %.2f / %.2f / %.2f\n', stats);
