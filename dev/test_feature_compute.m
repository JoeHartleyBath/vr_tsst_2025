% Test feature computation for one condition
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
addpath('C:/MATLAB/toolboxes/yamlmatlab');
eeglab nogui;

% Load configs
cfg_feat = yaml.ReadYaml('C:/vr_tsst_2025/config/eeg_feature_extraction.yaml');
cfg_cond = yaml.ReadYaml('C:/vr_tsst_2025/config/conditions_pilot.yaml');

% Load cleaned data
EEG = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
fprintf('Loaded P01_cleaned.set: %d channels, %d samples (%.1f sec @ %d Hz)\n', ...
    EEG.nbchan, EEG.pnts, EEG.pnts/EEG.srate, EEG.srate);

% Get regions and bands
regions = cfg_feat.regions;
frequency_bands = cfg_feat.frequency_bands;
chan_labels = {EEG.chanlocs.labels};

% Test with Blink_Calibration condition
test_cond = 'Blink_Calibration';
cond_cfg = cfg_cond.conditions.(test_cond);

fprintf('\nExtracting condition: %s\n', test_cond);
fprintf('  Start time: %.1f sec\n', cond_cfg.start);
fprintf('  Duration: %.1f sec\n', cond_cfg.duration);

% Find events
start_marker = cond_cfg.start_marker;
end_marker = cond_cfg.end_marker;
fprintf('  Looking for events: %s -> %s\n', start_marker, end_marker);

event_types = {EEG.event.type};
fprintf('  Total events in dataset: %d\n', length(event_types));
fprintf('  Sample event types: %s\n', strjoin(event_types(1:min(10, length(event_types))), ', '));

start_idx = find(strcmp(event_types, start_marker), 1);
end_idx = find(strcmp(event_types, end_marker), 1);

if isempty(start_idx)
    error('Start marker not found: %s', start_marker);
end
if isempty(end_idx)
    error('End marker not found: %s', end_marker);
end

start_sample = EEG.event(start_idx).latency;
end_sample = EEG.event(end_idx).latency;

fprintf('  Found markers at samples %d -> %d\n', start_sample, end_sample);

% Extract data
data = EEG.data(:, start_sample:end_sample);
fprintf('  Extracted data: %d channels × %d samples\n', size(data, 1), size(data, 2));
fprintf('  Data range: [%.3f, %.3f]\n', min(data(:)), max(data(:)));
fprintf('  Data has NaN? %d\n', any(isnan(data(:))));
fprintf('  Data has Inf? %d\n', any(isinf(data(:))));

% Compute PSD
window_length = min(2 * EEG.srate, size(data, 2));
overlap = round(window_length / 2);
nfft = 2^nextpow2(window_length);
fprintf('\nPSD parameters: window=%d, overlap=%d, nfft=%d\n', window_length, overlap, nfft);

[psd, freqs] = pwelch(data', hamming(window_length), overlap, nfft, EEG.srate);
psd = psd';  % [channels × freqs]

fprintf('PSD computed: %d channels × %d frequencies\n', size(psd, 1), size(psd, 2));
fprintf('PSD range: [%.3e, %.3e]\n', min(psd(:)), max(psd(:)));
fprintf('Frequency range: [%.2f, %.2f] Hz\n', freqs(1), freqs(end));

% Test FrontalLeft Alpha band power
fl_chans = regions.FrontalLeft;
chan_mask = ismember(chan_labels, fl_chans);
fprintf('\nFrontalLeft region:\n');
fprintf('  Expected channels: %s\n', strjoin(fl_chans(1:min(5, length(fl_chans))), ', '));
fprintf('  Matched %d/%d channels\n', sum(chan_mask), length(fl_chans));
fprintf('  Channel indices: ');
fprintf('%d ', find(chan_mask));
fprintf('\n');

alpha_range = frequency_bands.Alpha;
freq_mask = freqs >= alpha_range(1) & freqs <= alpha_range(2);
fprintf('\nAlpha band [%.1f-%.1f Hz]:\n', alpha_range(1), alpha_range(2));
fprintf('  Matched %d frequency bins\n', sum(freq_mask));

% Compute band power
region_psd = psd(chan_mask, :);
mean_psd = mean(region_psd, 1);
bp_raw = trapz(freqs(freq_mask), mean_psd(freq_mask));
bp_log = log10(max(bp_raw, 1e-10));

fprintf('  Raw band power: %.6e\n', bp_raw);
fprintf('  Log10 band power: %.3f\n', bp_log);
