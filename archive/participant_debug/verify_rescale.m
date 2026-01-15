% Quick verification after V->uV rescale
% - Checks amplitudes in raw P01.set and cleaned P01_cleaned.set
% - Confirms channel label matching for regions
% - Computes one sample band power to ensure non-floor values
% Wrap in try/catch to surface any errors in batch logs

try

addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
addpath(genpath('C:/MATLAB/toolboxes/eeglab2025.1.0/functions')); % ensure pop_* are on path
addpath('C:/MATLAB/toolboxes/yamlmatlab');

% Load configs
cfg_feat = yaml.ReadYaml('C:/vr_tsst_2025/config/eeg_feature_extraction.yaml');
regions = cfg_feat.regions;
frequency_bands = cfg_feat.frequency_bands;

%% Cleaned SET amplitude
fprintf('--- Cleaned P01_cleaned.set ---\n');
EEG_clean = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
clean_stats = [min(EEG_clean.data(:)), max(EEG_clean.data(:)), mean(EEG_clean.data(:)), std(EEG_clean.data(:))];
fprintf('Channels x samples: %d x %d\n', EEG_clean.nbchan, EEG_clean.pnts);
fprintf('Min/Max/Mean/Std (uV): %.6f / %.6f / %.6f / %.6f\n', clean_stats);

%% Region label match check
chan_labels = {EEG_clean.chanlocs.labels};
region_names = fieldnames(regions);
fprintf('\n--- Region channel matches ---\n');
for ri = 1:length(region_names)
    reg = region_names{ri};
    reg_chans = regions.(reg);
    mask = ismember(chan_labels, reg_chans);
    fprintf('%s: matched %d of %d (%s)\n', reg, sum(mask), numel(reg_chans), strjoin(chan_labels(mask), ', '));
end

%% Sample feature: FrontalLeft alpha power on first 30 seconds
win_samples = min(round(30 * EEG_clean.srate), size(EEG_clean.data, 2));
data = EEG_clean.data(:, 1:win_samples);

[psd, freqs] = pwelch(data', hamming(min(2*EEG_clean.srate, size(data,2))), [], [], EEG_clean.srate);
psd = psd';
fl_mask = ismember(chan_labels, regions.FrontalLeft);
alpha_range = frequency_bands.Alpha;
if iscell(alpha_range), alpha_range = cell2mat(alpha_range); end
alpha_range = double(alpha_range);
freq_mask = freqs >= alpha_range(1) & freqs <= alpha_range(2);
region_psd = psd(fl_mask, :);
mean_psd = mean(region_psd, 1);
bp = trapz(freqs(freq_mask), mean_psd(freq_mask));
bp_log = log10(max(bp, 1e-10));
fprintf('\n--- Sample feature ---\n');
fprintf('FrontalLeft Alpha band power (Blink_Calibration): raw=%.3e, log10=%.3f\n', bp, bp_log);

fprintf('\nIf log10 value is >> -10, scaling is fixed.\n');

catch ME
    fprintf('\nVerification failed: %s\n', ME.message);
    for k = 1:length(ME.stack)
        fprintf('  at %s:%d\n', ME.stack(k).name, ME.stack(k).line);
    end
    rethrow(ME);
end
