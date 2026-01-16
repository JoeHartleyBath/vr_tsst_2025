% FINAL UNIT/AMPLITUDE AUDIT (P01) — EEGLAB ground-truth slice export
%
% Creates matched-slice CSV exports directly from EEGLAB's EEG.data for:
%  - cleaned: output/cleaned_eeg/P01_cleaned.set (+ .fdt)
%  - raw:     output/sets/P01.set
%
% Required outputs:
%  - tmp/P01_cleaned_slice_matlab.csv
%  - tmp/P01_raw_slice_matlab.csv
%
% Slice definition (MATCH PYTHON):
%  - channels 1:5
%  - samples 1:1250  (first 10 seconds if srate=125 Hz)
%
% IMPORTANT:
%  - Values are written EXACTLY as stored in EEG.data (no unit conversion).
%  - If your srate is not 125 Hz, this is still the first 1250 samples.

clear; clc;

% Ensure EEGLAB is on path (adjust if needed)
% addpath('C:/path/to/eeglab');
% eeglab nogui;

out_dir = 'C:/vr_tsst_2025/tmp/';
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

slice_ch = 1:5;
slice_samp = 1:1250;

function export_one(set_path, out_csv)
    fprintf('\nLoading: %s\n', set_path);
    EEG = pop_loadset('filename', set_path);

    fprintf('EEG.datatype: %s\n', EEG.datatype);
    fprintf('class(EEG.data): %s\n', class(EEG.data));
    fprintf('EEG.srate: %.10g\n', EEG.srate);
    fprintf('EEG.nbchan: %d\n', EEG.nbchan);
    fprintf('EEG.pnts: %d\n', EEG.pnts);

    slice_ch = 1:5;
    slice_samp = 1:1250;

    if EEG.nbchan < max(slice_ch)
        error('Not enough channels: EEG.nbchan=%d', EEG.nbchan);
    end
    if EEG.pnts < max(slice_samp)
        error('Not enough samples: EEG.pnts=%d', EEG.pnts);
    end

    X = EEG.data(slice_ch, slice_samp);

    fprintf('\nMatched slice stats (channels 1:5, samples 1:1250):\n');
    fprintf('min:  %.10e\n', min(X(:)));
    fprintf('max:  %.10e\n', max(X(:)));
    fprintf('mean: %.10e\n', mean(X(:)));
    fprintf('std:  %.10e\n', std(X(:), 0, 'all'));

    fprintf('\nPer-channel std (rows 1..5):\n');
    disp(std(X, 0, 2));

    writematrix(X, out_csv);
    fprintf('Wrote CSV: %s\n', out_csv);
end

% CLEANED
export_one('C:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.set', 'C:/vr_tsst_2025/tmp/P01_cleaned_slice_matlab.csv');

% RAW / PRE-CLEAN
export_one('C:/vr_tsst_2025/output/sets/P01.set', 'C:/vr_tsst_2025/tmp/P01_raw_slice_matlab.csv');

fprintf('\nDone.\n');
