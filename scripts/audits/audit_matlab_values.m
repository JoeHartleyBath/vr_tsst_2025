% UNIT AUDIT - Compare MATLAB .fdt values to Python interpretation
%
% This script reads the P01_cleaned.fdt file directly in MATLAB
% to verify what values MATLAB actually stores/reads.

%% Load the cleaned .set file
fprintf('Loading P01_cleaned.set in MATLAB...\n');
EEG = pop_loadset('filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg/');

%% Get channel 1 data
ch1_data = EEG.data(1, :);

fprintf('\n=== MATLAB VALUES FOR CHANNEL 1 ===\n');
fprintf('Mean:   %.10e\n', mean(ch1_data));
fprintf('Std:    %.10e\n', std(ch1_data));
fprintf('Min:    %.10e\n', min(ch1_data));
fprintf('Max:    %.10e\n', max(ch1_data));
fprintf('P2P:    %.10e\n', max(ch1_data) - min(ch1_data));

fprintf('\nFirst 10 samples:\n');
disp(ch1_data(1:10));

fprintf('\n=== INTERPRETATION ===\n');
fprintf('If MATLAB values are in microvolts (EEGLAB convention):\n');
fprintf('  Std: %.6f uV\n', std(ch1_data));
fprintf('  Expected: 5-10 uV (cleaned, filtered)\n');

fprintf('\nIf MATLAB needs to be multiplied by 1000:\n');
fprintf('  Std: %.6f uV\n', std(ch1_data) * 1000);
fprintf('  Expected: 5-10 uV\n');

fprintf('\n=== CONCLUSION ===\n');
if std(ch1_data) < 0.1
    fprintf('WARNING: Std = %.6f uV is ~1000x too small!\n', std(ch1_data));
    fprintf('Expected std ~5-10 uV for cleaned EEG.\n');
    fprintf('This suggests the raw .set file input was incorrectly scaled.\n');
elseif std(ch1_data) > 100
    fprintf('WARNING: Std = %.6f uV is too large!\n', std(ch1_data));
    fprintf('Expected std ~5-10 uV for cleaned EEG.\n');
else
    fprintf('OK: Std = %.6f uV is in the expected range.\n', std(ch1_data));
end

fprintf('\nDone.\n');
