function qc = quick_asr_check(raw_set_path, participant_num, varargin)
% QUICK_ASR_CHECK - Load raw .set, assign chanlocs, basic filters, run ASR,
% and report bad channel count without running AMICA.

% Setup EEGLAB
if exist('pop_loadset','file') ~= 2
    error('EEGLAB not found on MATLAB path.');
end

% Load
[filepath, filename, ext] = fileparts(raw_set_path);
EEG = pop_loadset('filename', [filename, ext], 'filepath', filepath);
EEG.xmin = double(EEG.xmin); EEG.xmax = double(EEG.xmax);
EEG.srate = double(EEG.srate); EEG.pnts = double(EEG.pnts);
EEG.nbchan = double(EEG.nbchan); EEG.trials = double(EEG.trials);

% Chanlocs (ANT Neuro template)
projectRoot = fullfile(fileparts(mfilename('fullpath')), '..', '..', '..', '..');
projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);
EEG.etc.orig_chanlocs = EEG.chanlocs;

% Basic filters
if EEG.srate ~= 125
    EEG = pop_resample(EEG, 125);
end
EEG = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
EEG = pop_eegfiltnew(EEG, 'locutoff', 49, 'hicutoff', 51, 'revfilt', 1);
if ismember(participant_num, 1:7)
    EEG = pop_eegfiltnew(EEG, 'locutoff', 24.5, 'hicutoff', 25.5, 'revfilt', 1);
end

% Optional calibration window: Pre_Exposure_Blank_Fixation_Cross (code 10)
try
    calib_code = 10; calibDurSec = 60;
    calibEvIdx = find(strcmp({EEG.event.type}, num2str(calib_code)), 1, 'first');
    if ~isempty(calibEvIdx)
        calibStartSample = round(EEG.event(calibEvIdx).latency);
        calibEndSample = min(EEG.pnts, calibStartSample + round(calibDurSec * EEG.srate));
        EEG_calib = pop_select(EEG, 'point', [calibStartSample calibEndSample]);
        [~, ~] = clean_artifacts(EEG_calib, 'FlatlineCriterion',5,'ChannelCriterion',0.70,'LineNoiseCriterion',4,'BurstCriterion',50,'WindowCriterion',0.60);
    end
end

% Parse ASR parameters or use defaults
params = inputParser;
addParameter(params, 'FlatlineCriterion', 5);
addParameter(params, 'ChannelCriterion', 0.70);
addParameter(params, 'LineNoiseCriterion', 4);
addParameter(params, 'BurstCriterion', 50);
addParameter(params, 'WindowCriterion', 0.60);
parse(params, varargin{:});

% ASR on full data (tuned or swept params)
[EEG, ~] = clean_artifacts(EEG, ...
    'FlatlineCriterion', params.Results.FlatlineCriterion, ...
    'ChannelCriterion', params.Results.ChannelCriterion, ...
    'LineNoiseCriterion', params.Results.LineNoiseCriterion, ...
    'BurstCriterion', params.Results.BurstCriterion, ...
    'WindowCriterion', params.Results.WindowCriterion);

% Bad channels via mask
if ~isfield(EEG.etc, 'clean_channel_mask')
    EEG.etc.clean_channel_mask = true(1, length(EEG.etc.orig_chanlocs));
end
mask = EEG.etc.clean_channel_mask;
origLocs = EEG.etc.orig_chanlocs;
badIdx = ~mask;
badLabels = {origLocs(badIdx).labels}';

% QC summary
qc = struct();
qc.nBad = sum(badIdx);
qc.badLabels = badLabels;
if isfield(EEG.etc, 'clean_sample_mask')
    qc.percASRrepaired = 100 * mean(~EEG.etc.clean_sample_mask);
else
    qc.percASRrepaired = NaN;
end
qc.nbchan = EEG.nbchan;
qc.pnts = EEG.pnts;
qc.srate = EEG.srate;

fprintf('P%02d ASR-only QC: bad channels = %d, ASR repaired = %.2f%%\n', participant_num, qc.nBad, qc.percASRrepaired);
if ~isempty(badLabels)
    fprintf('Bad labels: %s\n', strjoin(badLabels, ', '));
end

end
