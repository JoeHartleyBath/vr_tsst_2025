%% extract_eeg_features_simple.m
% Simplified EEG feature extraction for pilot P01-P03
% No YAML, no parallel, works in batch mode

clearvars;
close all;
clc;

%% Setup paths
thisFile = mfilename('fullpath');
thisDir = fileparts(thisFile);
projectRoot = fullfile(thisDir, '..', '..', '..');
projectRoot = char(java.io.File(projectRoot).getCanonicalPath());
cd(projectRoot);

fprintf('Project root: %s\n', projectRoot);

%% Initialize EEGLAB (must be done before any EEGLAB functions)
% Dynamically find EEGLAB
eeglabRoot = 'C:/MATLAB/toolboxes';
eeglabDirs = dir(fullfile(eeglabRoot, 'eeglab*'));
if ~isempty(eeglabDirs)
    eeglabPath = fullfile(eeglabRoot, eeglabDirs(1).name);
    addpath(genpath(eeglabPath));
    fprintf('Added EEGLAB: %s\n', eeglabPath);
end

% Initialize EEGLAB
try
    eeglab nogui;
    fprintf('EEGLAB initialized\n');
catch
    fprintf('Warning: EEGLAB initialization issue, continuing anyway\n');
end

%% Configuration (hardcoded for pilot)
cleaned_folder = fullfile(projectRoot, 'output', 'cleaned_eeg');
output_folder = fullfile(projectRoot, 'output', 'eeg_features');
if ~exist(output_folder, 'dir'), mkdir(output_folder); end

output_csv = fullfile(output_folder, 'eeg_features.csv');
participants = [1, 2, 3];  % Pilot only

% Frequency bands
bands = struct();
bands.delta = [1 4];
bands.theta = [4 8];
bands.alpha = [8 13];
bands.beta = [13 30];
bands.gamma = [30 49];

fprintf('Output: %s\n', output_csv);
fprintf('Processing %d participants\n', length(participants));

%% Build column header
cols = {'Participant', 'Condition', ...
        'Delta_Power', 'Theta_Power', 'Alpha_Power', 'Beta_Power', 'Gamma_Power', ...
        'Alpha_Beta_Ratio', 'Theta_Beta_Ratio'};

% Write header
fid = fopen(output_csv, 'w');
fprintf(fid, '%s\n', strjoin(cols, ','));
fclose(fid);
fprintf('Header written\n');

%% Process each participant
fprintf('\n=== PROCESSING PARTICIPANTS ===\n');
tic;

for p = participants
    try
        fprintf('[P%02d] Loading...', p);
        
        % Load cleaned .set file
        cleaned_file = fullfile(cleaned_folder, sprintf('P%02d_cleaned.set', p));
        if ~exist(cleaned_file, 'file')
            fprintf(' SKIP (file not found)\n');
            continue;
        end
        
        % Load EEG
        EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), ...
                          'filepath', cleaned_folder);
        
        if isempty(EEG.data)
            fprintf(' SKIP (empty data)\n');
            continue;
        end
        
        fprintf(' %d ch x %d samples @ %.0f Hz...', EEG.nbchan, EEG.pnts, EEG.srate);
        
        % Average across channels
        data_avg = mean(EEG.data, 1);
        
        % Compute power spectrum using pwelch
        nperseg = min(EEG.srate * 4, length(data_avg));
        [pxx, f] = pwelch(data_avg, nperseg, nperseg/2, [], EEG.srate);
        
        % Band powers
        delta_pow = mean(pxx(f >= bands.delta(1) & f <= bands.delta(2)));
        theta_pow = mean(pxx(f >= bands.theta(1) & f <= bands.theta(2)));
        alpha_pow = mean(pxx(f >= bands.alpha(1) & f <= bands.alpha(2)));
        beta_pow = mean(pxx(f >= bands.beta(1) & f <= bands.beta(2)));
        gamma_pow = mean(pxx(f >= bands.gamma(1) & f <= bands.gamma(2)));
        
        % Ratios
        if beta_pow > 0
            alpha_beta = alpha_pow / beta_pow;
            theta_beta = theta_pow / beta_pow;
        else
            alpha_beta = 0;
            theta_beta = 0;
        end
        
        % Write row
        fid = fopen(output_csv, 'a');
        fprintf(fid, 'P%02d,Aggregated,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
                p, delta_pow, theta_pow, alpha_pow, beta_pow, gamma_pow, ...
                alpha_beta, theta_beta);
        fclose(fid);
        
        fprintf(' OK\n');
        
    catch ME
        fprintf(' ERROR: %s\n', ME.message);
    end
end

elapsed = toc;
fprintf('\n=== COMPLETE ===\n');
fprintf('Processed %d participants in %.1f seconds\n', length(participants), elapsed);
fprintf('Output: %s\n', output_csv);

% Display first few rows
fprintf('\nPreview:\n');
try
    data = readtable(output_csv);
    disp(data);
catch
    fprintf('(Could not display preview)\n');
end
