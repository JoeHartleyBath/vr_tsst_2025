% fix_eeglab_set_files.m
% One-off script to convert embedded-data EEGLAB .set files to two-file format (.set + .fdt)
% Input:  C:\vr_tsst_2025\output\cleaned_eeg\P*.set
% Output: C:\vr_tsst_2025\output\cleaned_eeg_fixed\P*.set (and .fdt)

% Initialise EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');

input_dir = 'C:\vr_tsst_2025\output\sets';
output_dir = 'C:\vr_tsst_2025\output\cleaned_sets_fixed';

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, 'P*.set'));

fprintf('Found %d .set files to process.\n', numel(files));

for k = 1:numel(files)
    infile = fullfile(input_dir, files(k).name);
    outfile = fullfile(output_dir, files(k).name);
    fprintf('Processing %s...\n', files(k).name);
    EEG = pop_loadset('filename', files(k).name, 'filepath', input_dir);
    if ~isa(EEG.data, 'single')
        EEG.data = single(EEG.data);
    end
    EEG = pop_saveset(EEG, 'filename', files(k).name, 'filepath', output_dir, 'savemode', 'twofiles');
    clear EEG;
end

fprintf('All files processed. Output written to %s\n', output_dir);
