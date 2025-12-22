% Fix P01_cleaned_patched.set by copying chanlocs from original P01.set
clearvars; close all; clc;

addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

fprintf('Loading original P01.set...\n');
EEG_orig = pop_loadset('filename', 'P01.set', 'filepath', 'C:/vr_tsst_2025/output/sets');
fprintf('  Original has %d channels, chanlocs length = %d\n', EEG_orig.nbchan, length(EEG_orig.chanlocs));

if length(EEG_orig.chanlocs) > 0
    fprintf('  Sample original labels:\n');
    for i = 1:min(5, length(EEG_orig.chanlocs))
        if isfield(EEG_orig.chanlocs, 'labels')
            fprintf('    %d: %s\n', i, EEG_orig.chanlocs(i).labels);
        end
    end
end

fprintf('\nLoading cleaned P01_cleaned_patched.set...\n');
EEG_clean = pop_loadset('filename', 'P01_cleaned_patched.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
fprintf('  Cleaned has %d channels, chanlocs length = %d\n', EEG_clean.nbchan, length(EEG_clean.chanlocs));

% Copy chanlocs from original to cleaned (matching first N channels)
if length(EEG_orig.chanlocs) >= EEG_clean.nbchan
    fprintf('\nCopying chanlocs from original to cleaned...\n');
    EEG_clean.chanlocs = EEG_orig.chanlocs(1:EEG_clean.nbchan);
    fprintf('  Done. Cleaned now has chanlocs length = %d\n', length(EEG_clean.chanlocs));
    
    fprintf('  Sample new labels in cleaned:\n');
    for i = 1:min(5, length(EEG_clean.chanlocs))
        fprintf('    %d: %s\n', i, EEG_clean.chanlocs(i).labels);
    end
    
    % Save fixed version
    fprintf('\nSaving fixed P01_cleaned.set...\n');
    EEG_clean = pop_saveset(EEG_clean, 'filename', 'P01_cleaned.set', 'filepath', 'C:/vr_tsst_2025/output/cleaned_eeg');
    fprintf('  Saved!\n');
else
    error('Original has fewer channels than cleaned - cannot copy chanlocs');
end

fprintf('\n=== DONE ===\n');
