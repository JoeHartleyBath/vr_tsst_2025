% CREATE_TEST_SUBSET - Extract first 5 minutes of P01.set for quick testing
%
% This creates a smaller test file for rapid pipeline validation

fprintf('Creating test subset from P01.set\n');
fprintf('==================================\n\n');

% Navigate to project root
cd('c:\phd_projects\vr_tsst_2025');

% Add paths
addpath('scripts/utils');

% Load the full dataset
input_file = fullfile('output', 'processed', 'P01.set');
fprintf('Loading full dataset: %s\n', input_file);
fprintf('This may take a minute...\n');

try
    EEG = pop_loadset(input_file);
    
    fprintf('Full dataset loaded successfully:\n');
    fprintf('  Channels: %d\n', EEG.nbchan);
    fprintf('  Samples: %d\n', EEG.pnts);
    fprintf('  Sampling rate: %.1f Hz\n', EEG.srate);
    fprintf('  Duration: %.1f seconds (%.1f minutes)\n', ...
        EEG.pnts / EEG.srate, (EEG.pnts / EEG.srate) / 60);
    fprintf('  Total events: %d\n\n', length(EEG.event));
    
    % Extract first 5 minutes
    duration_sec = 5 * 60;  % 5 minutes
    n_samples = round(duration_sec * EEG.srate);
    
    fprintf('Extracting first %d seconds (%d samples)...\n', duration_sec, n_samples);
    
    % Create subset
    EEG_subset = EEG;
    EEG_subset.data = EEG.data(:, 1:n_samples);
    EEG_subset.pnts = n_samples;
    EEG_subset.xmax = duration_sec;
    EEG_subset.times = (0:n_samples-1) / EEG.srate * 1000;  % in ms
    
    % Filter events that fall within the subset
    if ~isempty(EEG.event)
        event_latencies = [EEG.event.latency];
        valid_events = event_latencies <= n_samples;
        EEG_subset.event = EEG.event(valid_events);
        EEG_subset.urevent = EEG.urevent(valid_events);
        
        fprintf('  Events in subset: %d (out of %d total)\n', ...
            sum(valid_events), length(EEG.event));
        
        if sum(valid_events) > 0
            unique_event_types = unique([EEG_subset.event.type]);
            fprintf('  Event types in subset: %s\n', mat2str(unique_event_types));
        end
    end
    
    % Save subset
    output_folder = 'output/processed';
    output_file = 'P01_subset_5min.set';
    
    EEG_subset.setname = 'P01_subset_5min';
    fprintf('\nSaving subset to: %s/%s\n', output_folder, output_file);
    
    pop_saveset(EEG_subset, 'filename', output_file, 'filepath', output_folder);
    
    % Get file size
    output_path = fullfile(output_folder, output_file);
    file_info = dir(output_path);
    file_size_mb = file_info.bytes / (1024^2);
    
    fprintf('\nSUCCESS! Test subset created:\n');
    fprintf('  File: %s\n', output_path);
    fprintf('  Size: %.1f MB (original: ~982 MB)\n', file_size_mb);
    fprintf('  Channels: %d\n', EEG_subset.nbchan);
    fprintf('  Samples: %d\n', EEG_subset.pnts);
    fprintf('  Duration: %.1f seconds\n', duration_sec);
    fprintf('  Events: %d\n', length(EEG_subset.event));
    fprintf('\nYou can now test the cleaning pipeline on this smaller file.\n');
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    rethrow(ME);
end
