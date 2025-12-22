% Quick script: Create channel index mapping from NA-271.elc
% and update P01_cleaned.set with proper labels

clearvars; close all; clc;
addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

% Parse electrode file
elc_file = 'C:/vr_tsst_2025/config/chanlocs/NA-271.elc';
fid = fopen(elc_file);
lines = textscan(fid, '%s', 'Delimiter', '\n');
fclose(fid);
lines = lines{1};

% Extract channel labels (skip header lines)
chan_names = {};
for i = 1:length(lines)
    line = lines{i};
    if contains(line, ':')
        parts = strsplit(line, ':');
        name = strtrim(parts{1});
        if ~isempty(name) && ~strcmp(name, 'NumberPositions') && ~strcmp(name, 'Positions')
            chan_names{end+1} = name; %#ok<SAGROW>
        end
    end
end

fprintf('Parsed %d channel names from electrode file\n', length(chan_names));
fprintf('First 10: %s\n', strjoin(chan_names(1:min(10,length(chan_names))), ', '));

% Load cleaned set
fprintf('\nLoading P01_cleaned_patched.set...\n');
EEG = pop_loadset('filename', 'P01_cleaned_patched.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
fprintf('  Has %d channels\n', EEG.nbchan);

% Assign channel labels (first N from electrode file)
if length(chan_names) >= EEG.nbchan
    for ch = 1:EEG.nbchan
        EEG.chanlocs(ch).labels = chan_names{ch};
    end
    fprintf('  Assigned %d channel labels\n', EEG.nbchan);
    fprintf('  Sample: %s, %s, %s\n', EEG.chanlocs(1).labels, EEG.chanlocs(2).labels, EEG.chanlocs(3).labels);
    
    % Save as P01_cleaned.set
    fprintf('\nSaving P01_cleaned.set...\n');
    EEG = pop_saveset(EEG, 'filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
    fprintf('Saved!\n');
else
    error('Not enough channel names in electrode file');
end

fprintf('\n=== DONE ===\n');
