% run_add_chanlocs_p26.m
% Adds channel locations to the raw P26 EEGLAB dataset and saves a NEW copy
% (no overwrites) into share/P26_package_2026-02-03/raw.

projectRoot = 'C:\vr_tsst_2025';
run(fullfile(projectRoot, 'startup.m'));

inSet = fullfile(projectRoot, 'output', 'sets', 'P26.set');
outDir = fullfile(projectRoot, 'share', 'P26_package_2026-02-03', 'raw');
locsFile = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

fprintf('[add-chanlocs] Loading: %s\n', inSet);
[fp, fn, ext] = fileparts(inSet);
EEG = pop_loadset('filename', [fn ext], 'filepath', fp);

fprintf('[add-chanlocs] Applying chanlocs lookup: %s\n', locsFile);
EEG = pop_chanedit(EEG, 'lookup', locsFile);
EEG = eeg_checkset(EEG);

% Keep a copy for downstream interpolation / auditing
if ~isfield(EEG, 'etc') || isempty(EEG.etc)
    EEG.etc = struct();
end
EEG.etc.orig_chanlocs = EEG.chanlocs;

EEG.setname = 'P26_raw_chanlocs';
outName = 'P26_raw_chanlocs.set';

fprintf('[add-chanlocs] Saving NEW dataset to: %s\\%s\n', outDir, outName);
% Force two-file format so the .set stays small and data goes to .fdt
pop_saveset(EEG, 'filename', outName, 'filepath', outDir, 'savemode', 'twofiles');

fprintf('[add-chanlocs] Done.\n');
