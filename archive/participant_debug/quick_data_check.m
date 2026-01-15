% Quick test - no EEGLAB initialization needed
% Load data directly and compute one band power

addpath('C:/MATLAB/toolboxes/yamlmatlab');

% Load YAML to verify it works
cfg = yaml.ReadYaml('C:/vr_tsst_2025/config/eeg_feature_extraction.yaml');
fprintf('YAML loaded: %d regions, %d bands\n', length(fieldnames(cfg.regions)), length(fieldnames(cfg.frequency_bands)));

% Load .set file manually (without pop_loadset)
set_file = 'C:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.set';
EEG = load(set_file, '-mat');

fprintf('Loaded SET file: %d channels, %d timepoints\n', EEG.nbchan, EEG.pnts);
fprintf('Sampling rate: %d Hz\n', EEG.srate);
fprintf('Number of channel locations: %d\n', length(EEG.chanlocs));
fprintf('First 5 channel labels: ');
for i=1:5
    fprintf('%s ', EEG.chanlocs(i).labels);
end
fprintf('\n');

% Check if data field exists
if isfield(EEG, 'data')
    fprintf('Data shape: %d × %d\n', size(EEG.data, 1), size(EEG.data, 2));
else
    % Data in .fdt file
    fdt_file = strrep(set_file, '.set', '.fdt');
    fprintf('Reading data from: %s\n', fdt_file);
    fid = fopen(fdt_file, 'r', 'ieee-le');
    data = fread(fid, [EEG.nbchan, EEG.pnts], 'float32');
    fclose(fid);
    fprintf('Data loaded: %d × %d\n', size(data, 1), size(data, 2));
    fprintf('Data range: [%.3f, %.3f]\n', min(data(:)), max(data(:)));
end
