% QUICK_TEST_CLEAN - Quick test of cleaning steps
%
% Simple test without full AMICA to verify pipeline works

fprintf('Quick EEG Cleaning Test\n');
fprintf('=======================\n\n');

% Add paths
addpath('scripts/utils');

% Check if file exists
input_file = 'output/processed/P01.set';
if ~exist(input_file, 'file')
    error('File not found: %s', input_file);
end

fprintf('Loading %s...\n', input_file);

try
    % Try loading without EEGLAB GUI
    EEG = pop_loadset(input_file);
    
    fprintf('SUCCESS! File loaded:\n');
    fprintf('  Channels: %d\n', EEG.nbchan);
    fprintf('  Samples: %d\n', EEG.pnts);
    fprintf('  Sampling rate: %.1f Hz\n', EEG.srate);
    fprintf('  Duration: %.1f seconds\n', EEG.pnts / EEG.srate);
    fprintf('  Events: %d\n', length(EEG.event));
    
    if ~isempty(EEG.event)
        unique_events = unique([EEG.event.type]);
        fprintf('  Event types: %s\n', mat2str(unique_events));
    end
    
    % Test basic operations
    fprintf('\nTesting basic operations...\n');
    
    % Test resampling
    if EEG.srate ~= 125
        fprintf('  Resampling to 125 Hz...\n');
        EEG_test = pop_resample(EEG, 125);
        fprintf('    OK - New rate: %.1f Hz\n', EEG_test.srate);
    end
    
    % Test filtering
    fprintf('  Testing band-pass filter...\n');
    EEG_test = pop_eegfiltnew(EEG, 'locutoff', 1, 'hicutoff', 49);
    fprintf('    OK - Filter applied\n');
    
    fprintf('\nAll basic tests passed!\n');
    fprintf('The P01.set file is valid and ready for full cleaning.\n');
    
catch ME
    fprintf('ERROR: %s\n', ME.message);
    fprintf('Stack:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end
