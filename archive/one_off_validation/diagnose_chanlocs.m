%% Diagnose Chanlocs Issue - Trace Through Pipeline
clearvars; close all; clc;

fprintf('=== CHANLOCS DIAGNOSTIC ===\n\n');

addpath('C:/MATLAB/toolboxes/eeglab2025.1.0');
eeglab nogui;

projectRoot = pwd;

%% Check 1: Original P01.set
fprintf('CHECK 1: Loading original P01.set from output/sets...\n');
try
    EEG_orig = pop_loadset('filename', 'P01.set', 'filepath', fullfile(projectRoot, 'output', 'sets'));
    fprintf('  Channels: %d\n', EEG_orig.nbchan);
    fprintf('  Chanlocs length: %d\n', length(EEG_orig.chanlocs));
    if length(EEG_orig.chanlocs) > 0 && isfield(EEG_orig.chanlocs, 'labels')
        fprintf('  First 5 labels: ');
        for i = 1:min(5, length(EEG_orig.chanlocs))
            fprintf('%s ', EEG_orig.chanlocs(i).labels);
        end
        fprintf('\n');
    else
        fprintf('  WARNING: No chanlocs in original set!\n');
    end
    fprintf('\n');
catch ME
    fprintf('  ERROR loading P01.set: %s\n\n', ME.message);
end

%% Check 2: After pop_chanedit
fprintf('CHECK 2: Testing pop_chanedit with NA-271.elc...\n');
chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
fprintf('  Electrode file: %s\n', chanlocs_file);
fprintf('  File exists: %d\n', exist(chanlocs_file, 'file'));

if exist(chanlocs_file, 'file')
    try
        % Create a minimal test EEG structure
        EEG_test = eeg_emptyset();
        EEG_test.nbchan = 128;
        EEG_test.data = zeros(128, 1000);
        EEG_test.srate = 500;
        EEG_test.pnts = 1000;
        EEG_test.xmin = 0;
        EEG_test.xmax = 2;
        
        % Apply chanedit
        EEG_test = pop_chanedit(EEG_test, 'lookup', chanlocs_file);
        
        fprintf('  After pop_chanedit:\n');
        fprintf('    Chanlocs length: %d\n', length(EEG_test.chanlocs));
        if length(EEG_test.chanlocs) > 0
            fprintf('    First 5 labels: ');
            for i = 1:min(5, length(EEG_test.chanlocs))
                fprintf('%s ', EEG_test.chanlocs(i).labels);
            end
            fprintf('\n');
            fprintf('    Has X coords: %d\n', ~isempty(EEG_test.chanlocs(1).X));
        end
    catch ME
        fprintf('  ERROR in pop_chanedit: %s\n', ME.message);
    end
else
    fprintf('  ERROR: Electrode file not found!\n');
end
fprintf('\n');

%% Check 3: Cleaned set (if exists)
fprintf('CHECK 3: Checking existing P01_cleaned.set...\n');
cleaned_path = fullfile(projectRoot, 'output', 'cleaned_eeg', 'P01_cleaned.set');

if exist(cleaned_path, 'file')
    try
        EEG_clean = pop_loadset('filename', 'P01_cleaned.set', 'filepath', fullfile(projectRoot, 'output', 'cleaned_eeg'));
        fprintf('  Channels: %d\n', EEG_clean.nbchan);
        fprintf('  Chanlocs length: %d\n', length(EEG_clean.chanlocs));
        
        if length(EEG_clean.chanlocs) > 0 && isfield(EEG_clean.chanlocs, 'labels')
            fprintf('  First 5 labels: ');
            for i = 1:min(5, length(EEG_clean.chanlocs))
                fprintf('%s ', EEG_clean.chanlocs(i).labels);
            end
            fprintf('\n');
            
            % Check for orig_chanlocs
            if isfield(EEG_clean, 'etc') && isfield(EEG_clean.etc, 'orig_chanlocs')
                fprintf('  Has etc.orig_chanlocs: YES (length=%d)\n', length(EEG_clean.etc.orig_chanlocs));
            else
                fprintf('  Has etc.orig_chanlocs: NO\n');
            end
        else
            fprintf('  WARNING: Chanlocs empty or missing labels field!\n');
            
            % Check structure
            if length(EEG_clean.chanlocs) > 0
                fprintf('  Chanlocs fields: %s\n', strjoin(fieldnames(EEG_clean.chanlocs), ', '));
            end
        end
    catch ME
        fprintf('  ERROR loading P01_cleaned.set: %s\n', ME.message);
    end
else
    fprintf('  P01_cleaned.set does not exist yet\n');
end
fprintf('\n');

%% Check 4: Verify log file
fprintf('CHECK 4: Checking processing log...\n');
log_path = fullfile(projectRoot, 'output', 'cleaned_eeg', 'P01_processing_log.txt');

if exist(log_path, 'file')
    fprintf('  Log file exists: %s\n', log_path);
    fprintf('  Searching for chanlocs-related messages...\n\n');
    
    fid = fopen(log_path);
    lines = {};
    while ~feof(fid)
        lines{end+1} = fgetl(fid);
    end
    fclose(fid);
    
    % Find relevant lines
    for i = 1:length(lines)
        if contains(lower(lines{i}), {'chanloc', 'channel', 'verified'})
            fprintf('  %s\n', lines{i});
        end
    end
else
    fprintf('  Log file not found\n');
end
fprintf('\n');

%% Check 5: Test interpolation impact
fprintf('CHECK 5: Testing if pop_interp removes chanlocs...\n');
try
    % Create test dataset with chanlocs
    EEG_interp_test = eeg_emptyset();
    EEG_interp_test.nbchan = 10;
    EEG_interp_test.data = randn(10, 1000);
    EEG_interp_test.srate = 500;
    EEG_interp_test.pnts = 1000;
    EEG_interp_test.xmin = 0;
    EEG_interp_test.xmax = 2;
    
    % Assign simple chanlocs
    for ch = 1:10
        EEG_interp_test.chanlocs(ch).labels = sprintf('Ch%d', ch);
        EEG_interp_test.chanlocs(ch).X = ch;
        EEG_interp_test.chanlocs(ch).Y = ch;
        EEG_interp_test.chanlocs(ch).Z = ch;
    end
    
    orig_chanlocs_test = EEG_interp_test.chanlocs;
    
    fprintf('  Before pop_interp: %d chanlocs\n', length(EEG_interp_test.chanlocs));
    
    % Mark channel 5 as bad and remove it
    EEG_interp_test.data(5,:) = [];
    EEG_interp_test.chanlocs(5) = [];
    EEG_interp_test.nbchan = 9;
    
    % Try interpolation
    EEG_interp_test = pop_interp(EEG_interp_test, orig_chanlocs_test, 'spherical');
    
    fprintf('  After pop_interp: %d chanlocs\n', length(EEG_interp_test.chanlocs));
    fprintf('  Chanlocs preserved: %d\n', length(EEG_interp_test.chanlocs) == 10);
catch ME
    fprintf('  ERROR in interpolation test: %s\n', ME.message);
end
fprintf('\n');

%% Summary
fprintf('=== DIAGNOSTIC SUMMARY ===\n');
fprintf('1. Check if original P01.set has chanlocs\n');
fprintf('2. Verify pop_chanedit works with NA-271.elc\n');
fprintf('3. Check current P01_cleaned.set chanlocs status\n');
fprintf('4. Review processing log for chanlocs messages\n');
fprintf('5. Test if pop_interp preserves chanlocs\n');
fprintf('\nPlease review the output above to identify where chanlocs are lost.\n');
