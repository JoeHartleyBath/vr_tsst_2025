% Check if cleaned EEG data is actually zeros
fdt_file = 'C:/vr_tsst_2025/output/cleaned_eeg/P01_cleaned.fdt';

fprintf('Reading first 1000 samples from %s...\n', fdt_file);
fid = fopen(fdt_file, 'r', 'ieee-le');
if fid == -1
    error('Cannot open file');
end

% Read first 128 channels × 1000 samples
data_sample = fread(fid, [128, 1000], 'float32');
fclose(fid);

fprintf('Data shape: %d channels × %d samples\n', size(data_sample, 1), size(data_sample, 2));
fprintf('Data statistics:\n');
fprintf('  Min: %.6f\n', min(data_sample(:)));
fprintf('  Max: %.6f\n', max(data_sample(:)));
fprintf('  Mean: %.6f\n', mean(data_sample(:)));
fprintf('  Std: %.6f\n', std(data_sample(:)));
fprintf('  Number of zeros: %d / %d (%.1f%%)\n', sum(data_sample(:) == 0), numel(data_sample), 100 * sum(data_sample(:) == 0) / numel(data_sample));
fprintf('  Number of NaNs: %d\n', sum(isnan(data_sample(:))));

% Sample from middle of first channel
fprintf('\nSample values from channel 1, samples 100-110:\n');
fprintf('  %.6f\n', data_sample(1, 100:110));
