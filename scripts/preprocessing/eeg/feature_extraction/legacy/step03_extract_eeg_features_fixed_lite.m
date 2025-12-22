%% step03_extract_eeg_features_fixed_lite.m
% -------------------------------------------------------------------------
% VR-TSST EEG feature extraction (rolling + full window) with LITE MODE.
%
% LITE MODE accelerates export for ML prototyping by computing only the
% features you currently need and filling all other columns with NaN so
% downstream code that expects the full schema (281 columns) still works.
%
% --------------------- LITE MODE KEEPS -----------------------------------
%   • Ratios: Frontal_Alpha_Asymmetry, Alpha_Beta_Ratio, Theta_Beta_Ratio,
%             RightFrontal_Alpha, Theta_Alpha_Ratio   (per rolling window)
%   • Full-window MEAN ONLY (no SD, no Slope) for Theta, Alpha, Beta
%   • Regions: OverallFrontal, Temporal, Central, Parietal, Occipital,
%              FrontalMidline
%   • Full-window SampleEntropy & SpectralEntropy for the above regions
%
% --------------------- LITE MODE DROPS -----------------------------------
%   • Rolling region×band power columns (all 64) -> NaN
%   • Full-window SD & Slope -> NaN
%   • Full-window bands Delta, LowAlpha, HighAlpha, LowBeta, HighBeta -> NaN
%   • Full-window stats for FrontalLeft, FrontalRight -> NaN
%   • Entropies for FrontalLeft, FrontalRight -> NaN
%
% Header remains identical to FULL mode (281 columns).
%
% -------------------------------------------------------------------------
% USER OVERRIDES (set in workspace *before* calling this function):
%   PARTICIPANT_FILTER = [];   % [] = all (default [1] if absent)
%   LITE_MODE         = true;  % or false for full compute
%   DRY_RUN           = false; % compute but don't write if true
%   DEBUG             = true;
%   MAX_DEBUG_ROWS    = 5;
% -------------------------------------------------------------------------
% Dependencies:
%   EEGLAB, yaml.loadFile, SampEn, this script's helpers.
% -------------------------------------------------------------------------
% Extra toolboxes required:
%   - Brain Connectivity Toolbox (for wPLI)
%   - FOOOF-MATLAB or specparam (for 1/f slope)
% Add their paths once:



%% ---------------- USER CONFIG BOOTSTRAP --------------------------------
if ~exist('PARTICIPANT_FILTER','var') || isempty(PARTICIPANT_FILTER)
    PARTICIPANT_FILTER = [];         % default: process P1 only (safety)
end
if ~exist('LITE_MODE','var');       LITE_MODE = true; end
if ~exist('DRY_RUN','var');         DRY_RUN   = false; end
if ~exist('DEBUG','var');           DEBUG     = false;  end
if ~exist('MAX_DEBUG_ROWS','var');  MAX_DEBUG_ROWS = 5; end

%% ---------------- PATHS & CONFIG ---------------------------------------
addpath('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/utils');
config = yaml.loadFile('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/config.yaml');


%% ---------------- PARTICIPANT LIST -------------------------------------
participant_numbers = 1:48;
if ~isempty(PARTICIPANT_FILTER)
    participant_numbers = intersect(participant_numbers, PARTICIPANT_FILTER);
end
fprintf('Participants to process: %s\n', mat2str(participant_numbers));

%% ---------------- NOTCH EXCLUSION --------------------------------------
notch_participants = 1:7;   % adjust if needed
notch_band        = [24 26];

%% ---------------- LABEL MAPPING ----------------------------------------
label_changes = {
    'High_Stress',    'HighStress';
    'Low_Stress',     'LowStress';
    'Subtraction',    'HighCog';
    'Addition',       'LowCog';
    'Fixation_Cross', 'Baseline';
    'Forest',         'Relaxation';
};

%% ---------------- OUTPUT PATH ------------------------------------------
main_path      = config.paths.output;
metrics_folder = fullfile(main_path, 'aggregated');
if ~exist(metrics_folder,'dir'), mkdir(metrics_folder); end
output_csv = fullfile(metrics_folder, 'eeg_features.csv');

%% ---------------- FREQUENCY BANDS & REGIONS ----------------------------
frequency_bands = struct( ...
    'Delta',     [0.5, 4], ...
    'Theta',     [4, 7.5], ...
    'LowAlpha',  [7.5, 9.25], ...
    'HighAlpha', [9.25, 12], ...
    'LowBeta',   [12, 20], ...
    'HighBeta',  [20, 30], ...
    'Alpha',     [7.5, 12], ...
    'Beta',      [12, 30] ...
);

regions = struct();
regions.FrontalLeft    = {'L1','L2','L3','L4','LL1','LL2','LL3','LB1','LC1','LD1','LD2','LC2','LD3','LE1'};
regions.FrontalRight   = {'R1','R2','R3','R4','RR1','RR2','RR3','RB1','RC1','RD1','RD2','RC2','RD3','RE1'};
regions.OverallFrontal = {'L1','L2','L3','L4','LL1','LL2','LL3','LB1','LC1','LD1','LD2','LC2','LD3','LE1', ...
                         'R1','R2','R3','R4','RR1','RR2','RR3','RB1','RC1','RD1','RD2','RC2','RD3','RE1', ...
                         'Z1','Z2','Z3'};
regions.TemporalLeft = {'LC3','LC4','LC5','LC6','LD4','LD5','LD6','LE2','LE3','LM'};
regions.TemporalRight = {'RC3','RC4','RC5','RC6','RD4','RD5','RD6','RE2','RE3','RM'};
regions.Temporal       = {'LC3','LC4','LC5','LC6','LD4','LD5','LD6','LE2','LE3','LM', ...
                         'RC3','RC4','RC5','RC6','RD4','RD5','RD6','RE2','RE3','RM'};
regions.Central        = {'Z4','Z5','Z6','L5','L6','L7','LL4','LL5','LL6','LA1','LA2','LA3','LB2','LB3','LB4', ...
                         'R5','R6','R7','RR4','RR5','RR6','RA1','RA2','RA3','RB2','RB3','RB4'};
regions.ParietalLeft = {'L8','L9','L10','LL7','LL8','LL9','LL10','LL11','LA4','LA5','LB6','LC7','LB5'};
regions.ParietalRight = {'R8','R9','R10','RR7','RR8','RR9','RR10','RR11','RA4','RA5','RB6','RC7','RB5'};
regions.Parietal       = {'Z7','Z8','Z9','Z10','Z11','L8','L9','L10','LL7','LL8','LL9','LL10','LL11','LA4','LA5','LB6','LC7','LB5', ...
                         'R8','R9','R10','RR7','RR8','RR9','RR10','RR11','RA4','RA5','RB6','RC7','RB5'};
regions.Occipital      = {'Z12','Z13','L11','L12','L13','L14','LL12','LL13','LC7','LD7','LE4', ...
                         'R11','R12','R13','R14','RR12','RR13','RC7','RD7','RE4','LL14','RR14'};
regions.FrontalMidline = {'Z1','Z2','Z3'};

%% ---------------- CONDITIONS -------------------------------------------
task_conditions = { ...
    'Pre_Exposure_Blank_Baseline','Pre_Exposure_Room_Baseline', ...
    'Post_Exposure_Blank_Baseline','Post_Exposure_Room_Baseline', ...
    'HighStress_HighCog_Task','HighStress_LowCog_Task', ...
    'LowStress_HighCog_Task','LowStress_LowCog_Task', ...
    'Relaxation1','Relaxation2','Relaxation3','Relaxation4' ...
};

conditions = { ...
 'Primary_Calibrations','Blink_Calibration','Movement_Baseline', ...
 'Pre_Exposure_Blank_Baseline','Pre_Exposure_Room_Baseline', ...
 'Post_Exposure_Blank_Baseline','Post_Exposure_Room_Baseline', ...
 'HighStress_HighCog_Preamble','HighStress_LowCog_Preamble', ...
 'LowStress_HighCog_Preamble','LowStress_LowCog_Preamble', ...
 'HighStress_HighCog_Task','HighStress_LowCog_Task', ...
 'LowStress_HighCog_Task','LowStress_LowCog_Task', ...
 'HighStress_HighCog_Finish','HighStress_LowCog_Finish', ...
 'LowStress_HighCog_Finish','LowStress_LowCog_Finish', ...
 'Relaxation1','Relaxation2','Relaxation3','Relaxation4' ...
};

durations = [120,60,90,60,60,60,60,30,30,30,30,180,180,180,180,15,15,15,15,180,180,180,180];
assert(numel(conditions)==numel(durations),'Conditions vs durations mismatch');
condition_durations = containers.Map(conditions,durations);

%% ---------------- HEADER BUILD ----------------------------------------
stats = {'Mean','SD','Slope'};
region_entropy_stats = {'SampleEntropy','SpectralEntropy'};
region_names = fieldnames(regions);
band_names   = fieldnames(frequency_bands);

temporal_header = {};
for ri = 1:numel(region_names)
  for bi = 1:numel(band_names)
    for si = 1:numel(stats)
      temporal_header{end+1} = sprintf('Full_%s_%s_%s', ...
        region_names{ri}, band_names{bi}, stats{si}); %#ok<SAGROW>
    end
  end
  for si = 1:numel(region_entropy_stats)
    temporal_header{end+1} = sprintf('Full_%s_%s', region_names{ri}, region_entropy_stats{si}); %#ok<SAGROW>
  end
end


aperiodic_cols = {};
for r = ["FrontalMidline","Parietal"]
    aperiodic_cols{end+1} = sprintf('Full_%s_aperiodic_slope',      r); %#ok<SAGROW>
    aperiodic_cols{end+1} = sprintf('Full_%s_aperiodic_intercept',  r); %#ok<SAGROW>
end

wpli_cols = { ...
    'Full_wpli_fm_p_theta', ...
    'Full_wpli_fm_c_theta' ...
};

temporal_header = [temporal_header, aperiodic_cols, wpli_cols];

% Final header
cols = {'Participant','Window_Start_Second','Condition','Sample_Frame'};
for ri = 1:numel(region_names)
    for bi = 1:numel(band_names)
        cols{end+1} = sprintf('%s_%s_Power', region_names{ri}, band_names{bi}); %#ok<SAGROW>
    end
end
cols = [cols, ...
        {'Frontal_Alpha_Asymmetry','Alpha_Beta_Ratio','Theta_Beta_Ratio', ...
         'RightFrontal_Alpha','Theta_Alpha_Ratio'}, ...
        temporal_header, ...
        aperiodic_cols, ...
        wpli_cols];

col_index = containers.Map(cols, 1:numel(cols));

%% ---------------- LITE CONFIG OBJECT -----------------------------------
if LITE_MODE
    fprintf('*** LITE MODE ACTIVE (fast ML set) ***\n');
    LITE.include_regions = fieldnames(regions);
    LITE.include_bands   = fieldnames(frequency_bands);
    LITE.include_stats   = {'Mean'};     
    LITE.compute_rolling = false;        
    LITE.compute_entropy = false;         
    LITE.compute_ratios  = true;         
    LITE.compute_SD      = false;
    LITE.compute_Slope   = false;
      LITE.compute_aperiodic = false;             
    LITE.compute_connectivity = false; 
else
    LITE.include_regions = {}; % all
    LITE.include_bands   = {}; % all
    LITE.include_stats   = {}; % all
    LITE.compute_rolling = true;
    LITE.compute_entropy = true;
    LITE.compute_ratios  = true;
    LITE.compute_SD      = true;
    LITE.compute_Slope   = true;
    LITE.compute_aperiodic = true;          
    LITE.compute_connectivity = true; 
    
end

%% ---------------- WRITE HEADER -----------------------------------------
if ~DRY_RUN
    fid = fopen(output_csv,'w');
    if fid == -1, error('Could not open %s for writing.', output_csv); end
    fprintf(fid, '%s\n', strjoin(cols, ','));
    fclose(fid);
    fprintf('Header written (%d cols) -> %s\n', numel(cols), output_csv);
else
    fprintf('[DRY_RUN] Header NOT written (would be %d cols).\n', numel(cols));
end

%% ---------------- MAIN LOOP --------------------------------------------
%% ---------------- PATHS (main session) ----------------
addpath(genpath('C:/Program Files/MATLAB/R2025b/toolbox/eeglab2025.1.0'));
addpath(genpath('C:/Program Files/MATLAB/R2025b/toolbox/EntropyHub')); 
addpath(genpath('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/utils'));
addpath(genpath('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts')); 


%% ---------------- START POOL ----------------
pool = gcp('nocreate');
if ~isempty(pool)
    delete(pool);              % Close current pool
end

parpool('local', 8);         

pool = gcp('nocreate');
fprintf('NumWorkers in pool: %d\n', pool.NumWorkers);   % <-- confirm 8

%% ---------------- SYNC PATHS TO WORKERS ----------------
pctRunOnAll addpath('C:/Program Files/MATLAB/R2025b/toolbox/eeglab2025.1.0');
pctRunOnAll addpath('C:/Program Files/MATLAB/R2025b/toolbox/EntropyHub'); 
pctRunOnAll addpath('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/utils');
pctRunOnAll addpath('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts');
%% ---------------- INIT EEGLAB (on all workers) ---------
pctRunOnAll eeglab nogui;

LITE_local = LITE; 
bands_local = frequency_bands;
regions_local = regions;
cols_local = cols;

fprintf('Participants to process: %s\n', mat2str(participant_numbers));
pool = gcp('nocreate');
fprintf('NumWorkers in pool: %d\n', pool.NumWorkers);

%% ---------------- RUN MAIN LOOP ------------------------

parfor l = 1:numel(participant_numbers)
    p = participant_numbers(l);
    eeglab nogui;  % local worker init
    
    fprintf('\n▶ Processing P%d …\n', p);

    LITE = LITE_local;
    frequency_bands = bands_local;
    regions = regions_local;
    cols = cols_local;
    
        % cleaned .mat
        matname = sprintf('P%d_cleaned.mat', p);
        setpath = fullfile(config.paths.cleaned_eeg, matname);
        if ~isfile(setpath)
            warning('P%d: cleaned EEG file not found, skipping.', p);
            continue;
        end
    
        % derive participant #
        participant_num = p;
    
        % load .set metadata
        filtered_set_path = fullfile(config.paths.eeg_data, 'filtered', sprintf('P%02d_filtered.set', participant_num));
        if ~isfile(filtered_set_path)
            warning('P%d: filtered .set not found (%s), skipping.', p, filtered_set_path);
            continue;
        end
        EEG = pop_loadset('filename', sprintf('P%02d_filtered.set', participant_num), ...
                          'filepath', char(fullfile(config.paths.eeg_data, 'filtered')));
    
        % load cleaned data
        cleaned_data  = load(setpath);
        cleaned_field = fieldnames(cleaned_data);
        EEG.data = cleaned_data.(cleaned_field{1});
    
        % drop ch129
        if size(EEG.data,1) >= 129
            EEG.data(129,:,:) = [];
            if numel(EEG.chanlocs) >= 129, EEG.chanlocs(129) = []; end
            fprintf('  Dropped channel 129.\n');
        end
    
        % update dims
        [EEG.nbchan, EEG.pnts, EEG.trials] = size(EEG.data);

        if ndims(EEG.data) == 2, EEG.trials = 1; end

        EEG.setname = sprintf('Advanced_Cleaned_EEG_Data_P%d', participant_num);
        EEG = eeg_checkset(EEG);
    
        % optional save
        try
            pop_saveset(EEG,'filename',[EEG.setname '.set'],'filepath',char(fileparts(setpath)));
            fprintf('  Saved reconstructed .set.\n');
        catch ME
            warning('P%d: save .set failed (%s).', p, ME.message);
        end
    
        % relabel events
        EEG = update_event_labels_eeg(EEG, label_changes);
        evtf = fullfile(fileparts(setpath), [EEG.setname '.evt']);
        update_event_file(evtf, label_changes);
    
        % load events csv
        evp = fullfile(char(config.paths.events), sprintf('P%02d_events.csv', p));
        if ~isfile(evp)
            warning('P%d: events csv missing.', p);
            continue;
        end
        eventTable = readtable(evp);
        eventTable = sortrows(eventTable,'latency');
        if ~ismember('type',eventTable.Properties.VariableNames)
            error('Events table missing column \"type\".');
        end
        for ii = 1:height(eventTable)
            lbl = eventTable.type{ii};
            for kk = 1:size(label_changes,1)
                lbl = strrep(lbl, label_changes{kk,1}, label_changes{kk,2});
            end
            if ~contains(lbl,'Relaxation')
                lbl = regexprep(lbl,'\\d','');
            end
            lbl = regexprep(lbl,'__+','_');
            eventTable.type{ii} = lbl;
        end
    
        % amplitude sanity
        fprintf('  Max abs amplitude (µV) P%d: %.1f\n', p, max(abs(EEG.data(:))));
    
            % ---------------- DEBUG: show normalized raw event labels ------------
        if DEBUG
            fprintf('\n-- Raw event labels (first %d) P%d --\n', height(eventTable), p);
            disp(unique(eventTable.type));
        end
    
        rows_this_participant = {};
        seen = {};  % canonical labels already exported
    
        for i = 1:height(eventTable)
            raw_cond = eventTable.type{i};
            cond = canonicalize_condition_task_only(raw_cond);
    
            if DEBUG
                fprintf('Evt %2d: %-40s -> %-28s', i, raw_cond, cond);
            end
    
            % Skip if not mapped to an analysis label
            if isempty(cond)
                if DEBUG; fprintf('  SKIP\n'); end
                continue;
            end
    
            % Skip if we've already processed this canonical label
            if ismember(cond, seen)
                if DEBUG; fprintf('  SKIP (already seen)\n'); end
                continue;
            end
            seen{end+1} = cond;
    
            % Must be in analysis set (task_conditions list holds Baselines,
            % 4 Stress×Workload Tasks, 4 Relaxations)
            if ~ismember(cond, task_conditions)
                if DEBUG; fprintf('  WARN: not in task_conditions list, but processing anyway.\n'); end
                % you can flip to continue; I'm permissive here.
            else
                if DEBUG; fprintf('  PROCESS\n'); end
            end
    
            % ---------------- timing ----------------------------------------
            lat = round(eventTable.latency(i));  % sample index start
            duration = condition_durations(cond); % seconds
            t0 = max(1, lat);
            t1_req = t0 + duration*EEG.srate - 1;
    
            if t1_req > EEG.pnts
                if DEBUG
                    fprintf('    Clip %s: requested end %d > data %d samples.\n', cond, t1_req, EEG.pnts);
                end
                t1 = EEG.pnts;
            else
                t1 = t1_req;
            end
    
            seg_len_s = (t1 - t0 + 1)/EEG.srate;
            if seg_len_s < 5
                warning('    %s segment <5s (%.2f s) -> skipping.', cond, seg_len_s);
                continue;
            end
    
            segment = EEG.data(:, t0:t1);
    
            % Notch exclusion
            if ismember(participant_num, notch_participants)
                excludeBand = notch_band;
            else
                excludeBand = [];
            end
    
            % Compute full-window features (Lite aware)
            full_feats = compute_full_features_lite(segment, EEG.srate, frequency_bands, ...
                                            regions, {EEG.chanlocs.labels}, cond, ...
                                            excludeBand, LITE);
    
    
            % Rolling windows (still written; Lite fills NaNs for dropped cols)
            % Use the ACTUAL clipped segment duration so windows fit
            duration_actual_s = seg_len_s;
            offsets = 0:15:(duration_actual_s - 30);
            if isempty(offsets)
                offsets = 0;  % write one row anchored at start
            end
    
            for off = offsets
                w0 = t0 + round(off*EEG.srate);
                w1 = w0 + 30*EEG.srate - 1;
                if w1 > t1
                    break;
                end
                [spec, freqs] = calc_psd_with_rolling_window(EEG.data(:,w0:w1), EEG.srate);
    
                rowcells = build_row_cells_lite(p, (w0-1)/EEG.srate, cond, off, ...
                                    spec, freqs, regions, frequency_bands, EEG, excludeBand, ...
                                    full_feats, cols, temporal_header, col_index, LITE);
    
                rows_this_participant{end+1} = strjoin_safe(rowcells, ',');
            end
    
        end
    
        tmp_file = fullfile(metrics_folder, sprintf('tmp_eeg_P%d.csv', p));
        
        fid = fopen(tmp_file, 'w');

        fprintf(fid, '%s\n', strjoin(cols, ','));

            for k = 1:numel(rows_this_participant)
                fprintf(fid, '%s\n', rows_this_participant{k});
            end

        fclose(fid);
end

tmp_files = dir(fullfile(metrics_folder, 'tmp_eeg_P*.csv'));

% Extract participant IDs from filenames
pnums = arrayfun(@(x) sscanf(x.name,'tmp_eeg_P%d.csv'), tmp_files);
[~, idx] = sort(pnums);
tmp_files = tmp_files(idx);

fid_out = fopen(output_csv, 'a');  % append mode

for k = 1:numel(tmp_files)
    tmp_path = fullfile(tmp_files(k).folder, tmp_files(k).name);
    C = readlines(tmp_path);
    if numel(C) > 1
        fprintf(fid_out, '%s\n', C(2:end));  % skip header
    end
end

fclose(fid_out);
fprintf('Merged %d temp files into master: %s\n', numel(tmp_files), output_csv);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                          HELPER FUNCTIONS                            %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function rowcells = build_row_cells_lite(p, start_sec, cond, frame, ...
                                    spec, freqs, regions, bands, EEG, excludeBand, ...
                                    full_feats, cols, temporal_header, col_index, LITE)
% Return 1xN cell array in header order. Honor LITE gating.

    rowcells = repmat({''}, 1, numel(cols));

    % metadata
    rowcells{col_index('Participant')}         = sprintf('%d', p);
    rowcells{col_index('Window_Start_Second')} = sprintf('%.2f', start_sec);
    rowcells{col_index('Condition')}           = cond;
    rowcells{col_index('Sample_Frame')}        = sprintf('%d', frame);

    chanLabels = {EEG.chanlocs.labels};
    regNames  = fieldnames(regions);
    bandNames = fieldnames(bands);

    % rolling metrics
    do_roll = ~isfield(LITE,'compute_rolling') || LITE.compute_rolling;
    for r = 1:numel(regNames)
        for b = 1:numel(bandNames)
            cname = sprintf('%s_%s_Power', regNames{r}, bandNames{b});
            if do_roll && (isempty(LITE.include_regions)  || ismember(regNames{r}, LITE.include_regions)) && ...
                          (isempty(LITE.include_bands)    || ismember(bandNames{b}, LITE.include_bands))
                mask = ismember(chanLabels, regions.(regNames{r}));
                fr    = bands.(bandNames{b});
                fmask = freqs >= fr(1) & freqs <= fr(2);
                if ~isempty(excludeBand)
                    fmask = fmask & ~(freqs >= excludeBand(1) & freqs <= excludeBand(2));
                end
                if any(fmask) && any(mask)
                    psd_mean = mean(spec(mask, fmask), 1);
                    df = mean(diff(freqs));
                    pwr = sum(psd_mean) * df;
                else
                    pwr = NaN;
                end
                rowcells{col_index(cname)} = num2str(pwr, '%.6f');
            else
                rowcells{col_index(cname)} = '';  % -> NaN
            end
        end
    end

    % ratios 
    do_ratios = ~isfield(LITE,'compute_ratios') || LITE.compute_ratios;
    if do_ratios
        Lidx  = ismember(chanLabels, regions.FrontalLeft);
        Ridx  = ismember(chanLabels, regions.FrontalRight);
        FMidx = ismember(chanLabels, regions.FrontalMidline);

        aL  = band_power_integrated(spec, freqs, Lidx,  bands.Alpha,  excludeBand);
        aR  = band_power_integrated(spec, freqs, Ridx,  bands.Alpha,  excludeBand);
        aFM = band_power_integrated(spec, freqs, FMidx, bands.Alpha,  excludeBand);
        bFM = band_power_integrated(spec, freqs, FMidx, bands.Beta,   excludeBand);
        tFM = band_power_integrated(spec, freqs, FMidx, bands.Theta,  excludeBand);

        FAA = log_safe(aR) - log_safe(aL);
        ABR = deal_nan_if_zero(aFM, bFM);
        TBR = deal_nan_if_zero(tFM, bFM);
        RFA = aR;
        TAR = deal_nan_if_zero(tFM, aFM);

        rowcells{col_index('Frontal_Alpha_Asymmetry')} = num2str(FAA, '%.6f');
        rowcells{col_index('Alpha_Beta_Ratio')}        = num2str(ABR, '%.6f');
        rowcells{col_index('Theta_Beta_Ratio')}        = num2str(TBR, '%.6f');
        rowcells{col_index('RightFrontal_Alpha')}      = num2str(RFA, '%.6f');
        rowcells{col_index('Theta_Alpha_Ratio')}       = num2str(TAR, '%.6f');
    else
        rowcells{col_index('Frontal_Alpha_Asymmetry')} = '';
        rowcells{col_index('Alpha_Beta_Ratio')}        = '';
        rowcells{col_index('Theta_Beta_Ratio')}        = '';
        rowcells{col_index('RightFrontal_Alpha')}      = '';
        rowcells{col_index('Theta_Alpha_Ratio')}       = '';
    end

    % full-window (copy from feats)
    for oi = 1:numel(temporal_header)
        key = temporal_header{oi};
        if isfield(full_feats, key)
            rowcells{col_index(key)} = num2str(full_feats.(key), '%.6f');
        else
            rowcells{col_index(key)} = '';
        end
    end
end


function feats = compute_full_features_lite(win, sr, fb, regs, chanlabels, conditionName, excludeBand, LITE)
% Full-window stats with LITE gating.
    if nargin < 7, excludeBand = []; end
    if nargin < 8 || isempty(LITE)
        LITE.compute_entropy = true;
        LITE.compute_SD      = true;
        LITE.compute_Slope   = true;
        LITE.include_regions = {};
        LITE.include_bands   = {};
        LITE.include_stats   = {};
    end

    feats = struct();
    rnames = fieldnames(regs);
    bnames = fieldnames(fb);

    hop = floor(sr/2);
    PSD = [];
    for t0 = 1:hop:(size(win,2) - sr + 1)
        seg = win(:, t0:(t0 + sr - 1));
        [P, f] = pwelch(seg', sr, hop, [], sr);
        PSD(:,:,end+1) = P'; %#ok<AGROW>
    end

    for ri = 1:numel(rnames)
        rname = rnames{ri};
        mask  = ismember(chanlabels, regs.(rname));

        % region gating
        region_included = isempty(LITE.include_regions) || ismember(rname, LITE.include_regions);

        for bi = 1:numel(bnames)
            band = bnames{bi};
            base = sprintf('Full_%s_%s', rname, band);

            band_included = region_included && (isempty(LITE.include_bands) || ismember(band, LITE.include_bands));

            if band_included
                fr    = fb.(band);
                fmask = f >= fr(1) & f <= fr(2);
                if ~isempty(excludeBand)
                    fmask = fmask & ~(f >= excludeBand(1) & f <= excludeBand(2));
                end
                if ~any(fmask) || ~any(mask)
                    ts = NaN;
                else
                    psd_rf = squeeze(mean(PSD(mask, fmask, :),1));  % (freq x win)
                    df = mean(diff(f));
                    ts = sum(psd_rf,1) * df;
                end
                feats.([base '_Mean']) = mean(ts,'omitnan');
                feats.([base '_SD'])   = ternary_val(LITE.compute_SD,   std(ts,'omitnan'),   NaN);
                if LITE.compute_Slope && numel(ts) >= 2 && all(~isnan(ts))
                    coeffs = polyfit((1:numel(ts))', ts(:), 1);
                    feats.([base '_Slope']) = coeffs(1);
                else
                    feats.([base '_Slope']) = NaN;
                end
            else
                feats.([base '_Mean'])  = NaN;
                feats.([base '_SD'])    = NaN;
                feats.([base '_Slope']) = NaN;
            end
        end

        % entropies
        % Compute Sample Entropy (m=2, tau=1) for each EEG channel in 'mask',
        % then average the channel-wise values (ignoring NaNs) to give a single
        % entropy feature for the region. Uses r = 0.2*SD as tolerance.

        % ENTROPIES
        % Sample Entropy (m=2, tau=1) per channel in 'mask' → average (omit NaNs).
        % r = 0.2*SD per channel; SampEn reported in nats (Logx = e).
        
        if region_included
    % ----- Sample Entropy using region-averaged signal -----
            if LITE.compute_entropy
                chanIdx = find(mask);
                if isempty(chanIdx)
                    feats.(['Full_' rname '_SampleEntropy']) = NaN;
                else
                    % 1) Average the channels in this region
                    x = mean(double(win(chanIdx, :)), 1);  % single time series
        
                    % 2) Basic validity check
                    if ~all(isfinite(x))
                        feats.(['Full_' rname '_SampleEntropy']) = NaN;
                    else
                        % 3) Compute tolerance r based on region signal
                        rtol = max(eps, 0.2 * std(x));
        
                        % 4) Call SampEn once
                        try
                            [SampVec, ~, ~] = SampEn(x, 'm', 2, 'tau', 1, ...
                                'r', rtol, 'Logx', exp(1), 'Vcp', false);
                            feats.(['Full_' rname '_SampleEntropy']) = SampVec(2);  % m=2 index
                        catch
                            feats.(['Full_' rname '_SampleEntropy']) = NaN;
                        end
                    end
        
                    % Optional: keep debug print
                    fprintf('  [%s | %s] SampEn(avg) = %.4f\n', ...
                        conditionName, rname, feats.(['Full_' rname '_SampleEntropy']));
                end
            else
                feats.(['Full_' rname '_SampleEntropy']) = NaN;
            end
                
            % ----- Spectral Entropy -----
            % Frequency mask: 1–30 Hz, with optional band exclusion
            fmask_full = (f >= 1 & f <= 30);
            if ~isempty(excludeBand)
                fmask_full = fmask_full & ~(f >= excludeBand(1) & f <= excludeBand(2));
            end
        
            if ~any(fmask_full) || ~any(mask)
                feats.(['Full_' rname '_SpectralEntropy']) = NaN;
            else
                % Average PSD across selected channels, then across time/windows.
                % Assumes PSD is [Channels x Freq x TimeWindows]
                PSD_chan = squeeze(mean(PSD(mask, fmask_full, :), 1));  % [F x T]
                Pfreq    = mean(PSD_chan, 2);                           % [F x 1]
                fvec     = f(fmask_full);
        
                tot = sum(Pfreq);
                if ~isfinite(tot) || tot <= 0
                    feats.(['Full_' rname '_SpectralEntropy']) = NaN;
                else
                    % Unnormalized spectral entropy in bits (dimensionless).
                    % Use 'Normalized',false to avoid 0–1 scaling.
                    se_bits = spectralEntropy(Pfreq(:), fvec(:), Instantaneous=false, Scaled=true);
                    feats.(['Full_' rname '_SpectralEntropy']) = se_bits;
                end
            end
        else
            feats.(['Full_' rname '_SampleEntropy'])   = NaN;
            feats.(['Full_' rname '_SpectralEntropy']) = NaN;
        end
        
    end

end


function bp = band_power_integrated(spec, freqs, chan_mask, bandrange, excludeBand)
%BAND_POWER_INTEGRATED Integrate PSD over band (µV^2).
    if nargin < 5, excludeBand = []; end
    fmask = freqs >= bandrange(1) & freqs <= bandrange(2);
    if ~isempty(excludeBand)
        fmask = fmask & ~(freqs >= excludeBand(1) & freqs <= excludeBand(2));
    end
    if ~any(fmask) || ~any(chan_mask)
        bp = NaN; return; end
    psd_mean = mean(spec(chan_mask, fmask), 1);
    df = mean(diff(freqs));
    bp = sum(psd_mean) * df;
end


function v = log_safe(x)
    if isnan(x) || x <= 0
        v = NaN;
    else
        v = log(x);
    end
end

function r = deal_nan_if_zero(num, den)
    if isnan(num) || isnan(den) || den <= 0
        r = NaN;
    else
        r = num / den;
    end
end


function out = ternary_val(cond, a, b)
    if cond, out = a; else, out = b; end
end


function [psd, f] = calc_psd_with_rolling_window(data, sr)
    nwin  = sr*4;
    nover = nwin/2;
    [P,f] = pwelch(data', nwin, nover, [], sr);
    psd   = P';
end


function EEG = update_event_labels_eeg(EEG, label_changes)
    if ~isfield(EEG,'event') || isempty(EEG.event), return; end
    for j = 1:numel(EEG.event)
        if ~isfield(EEG.event(j),'type'), continue; end
        lbl = EEG.event(j).type;
        for k = 1:size(label_changes,1)
            lbl = strrep(lbl, label_changes{k,1}, label_changes{k,2});
        end
        if ~contains(lbl,'Relaxation')
            lbl = regexprep(lbl,'\\d','');
        end
        lbl = regexprep(lbl,'__+','_');
        EEG.event(j).type = lbl;
    end
end


function update_event_file(evtfile, label_changes)
    if ~isfile(evtfile), return; end
    C = readlines(evtfile);
    for k = 1:size(label_changes,1)
        C = strrep(C, label_changes{k,1}, label_changes{k,2});
    end
    writelines(C, evtfile);
end


function s = strjoin_safe(c, delim)
    if nargin < 2, delim = ','; end
    c = c(:)'; for i = 1:numel(c), if isempty(c{i}), c{i}=''; end; end %#ok<AGROW>
    s = strjoin(c, delim);
end


function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end

function canon = canonicalize_condition_task_only(lbl)
%CANONICALIZE_CONDITION_TASK_ONLY
% Map raw event labels to *analysis* condition names.
% Rules:
%   • Accept ONLY the true *_Task labels for Stress×Workload blocks.
%   • Explicitly skip *_Preamble and *_Finish variants (return '').
%   • Map arithmetic numeric suffixes (Subtraction1022, etc.) → HighCog / LowCog.
%   • Map Fixation_Cross variants to Baseline names.
%   • Map Forest# → Relaxation#.
%   • Strip digits everywhere except Relaxation# (digits matter there).
%   • Return '' if label not of interest.
%
% Outputs one of:
%   Pre_Exposure_Blank_Baseline
%   Pre_Exposure_Room_Baseline
%   Post_Exposure_Blank_Baseline
%   Post_Exposure_Room_Baseline
%   HighStress_HighCog_Task
%   HighStress_LowCog_Task
%   LowStress_HighCog_Task
%   LowStress_LowCog_Task
%   Relaxation1 .. Relaxation4
%   ''  (skip)
%
% NOTE: Case-sensitive; incoming `lbl` may be string or char.

    canon = '';
    if isempty(lbl); return; end
    if isstring(lbl); lbl = char(lbl); end
    lbl = strtrim(lbl);

    % ------------------------------------------------------------------
    % Relaxation (Forest#) — keep numeric suffix (1..4) because your
    % downstream schema expects Relaxation1..4
    % ------------------------------------------------------------------
    if startsWith(lbl,'Forest','IgnoreCase',true) || startsWith(lbl,'Relaxation','IgnoreCase',true)
        d = regexp(lbl,'(\d+)$','tokens','once');
        if isempty(d), d = {'1'}; end  % fallback
        canon = ['Relaxation' d{1}];
        return;
    end

    % ------------------------------------------------------------------
    % Fixation_Cross → Baselines
    % ------------------------------------------------------------------
    if contains(lbl,'Fixation_Cross','IgnoreCase',true)
        if contains(lbl,'Pre_Exposure_Blank','IgnoreCase',true)
            canon = 'Pre_Exposure_Blank_Baseline'; return;
        elseif contains(lbl,'Pre_Exposure_Room','IgnoreCase',true)
            canon = 'Pre_Exposure_Room_Baseline'; return;
        elseif contains(lbl,'Post_Exposure_Blank','IgnoreCase',true)
            canon = 'Post_Exposure_Blank_Baseline'; return;
        elseif contains(lbl,'Post_Exposure_Room','IgnoreCase',true)
            canon = 'Post_Exposure_Room_Baseline'; return;
        end
    end

    % Already-canonical baselines pass through
    if any(strcmp(lbl, {
        'Pre_Exposure_Blank_Baseline','Pre_Exposure_Room_Baseline', ...
        'Post_Exposure_Blank_Baseline','Post_Exposure_Room_Baseline'}))
        canon = lbl; return;
    end

    % ------------------------------------------------------------------
    % Explicitly SKIP *_Preamble / *_Finish  (return '')
    % ------------------------------------------------------------------
    if contains(lbl,'Preamble','IgnoreCase',true) || contains(lbl,'Finish','IgnoreCase',true)
        return;
    end

    % ------------------------------------------------------------------
    % Stress×Workload TRUE TASKS ONLY
    % (These must contain '_Task'; others already filtered out above.)
    % ------------------------------------------------------------------
    if contains(lbl,'_Task','IgnoreCase',true)
        % normalize tokens
        lbl2 = lbl;
        lbl2 = strrep(lbl2,'High_Stress','HighStress');
        lbl2 = strrep(lbl2,'Low_Stress','LowStress');
        lbl2 = regexprep(lbl2,'Subtraction','HighCog');
        lbl2 = regexprep(lbl2,'Addition','LowCog');

        % strip digits globally
        lbl2 = regexprep(lbl2,'\d','');

        % collapse underscores
        lbl2 = regexprep(lbl2,'__+','_');

        % Expect something like HighStress_HighCog_Task
        if contains(lbl2,'HighStress') && contains(lbl2,'HighCog')
            canon = 'HighStress_HighCog_Task'; return;
        elseif contains(lbl2,'HighStress') && contains(lbl2,'LowCog')
            canon = 'HighStress_LowCog_Task'; return;
        elseif contains(lbl2,'LowStress') && contains(lbl2,'HighCog')
            canon = 'LowStress_HighCog_Task'; return;
        elseif contains(lbl2,'LowStress') && contains(lbl2,'LowCog')
            canon = 'LowStress_LowCog_Task'; return;
        end
    end

    % ------------------------------------------------------------------
    % Known blocks to ignore silently
    % ------------------------------------------------------------------
    if any(strcmp(lbl, {'Blink_Calibration','Movement_Baseline','Primary_Calibrations'}))
        canon = ''; return;
    end

    % default: skip
end
