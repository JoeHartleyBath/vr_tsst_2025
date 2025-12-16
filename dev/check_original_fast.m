% Quick check - load original P01.set via scipy directly
try
    EEG = load('C:/vr_tsst_2025/output/sets/P01.set', '-mat');
    fprintf('Dataset loaded\n');
    fprintf('Data shape: %d channels × %d samples\n', size(EEG.EEG.data, 1), size(EEG.EEG.data, 2));
    fprintf('Data range: [%.6f, %.6f]\n', min(EEG.EEG.data(:)), max(EEG.EEG.data(:)));
    fprintf('Data std: %.6f\n', std(EEG.EEG.data(:)));
    fprintf('\nFirst 10 values from channel 1:\n');
    disp(EEG.EEG.data(1, 1:10));
catch ME
    fprintf('Error: %s\n', ME.message);
end
