function EEG = create_mock_eeg_data(participant_id, config)
% CREATE_MOCK_EEG_DATA Generate minimal synthetic EEG data for testing
%
% Inputs:
%   participant_id - Participant number (e.g., 1, 2)
%   config        - Configuration structure with paths and conditions
%
% Output:
%   EEG - EEGLAB structure with synthetic data and events

    % Initialize EEGLAB structure
    EEG = eeg_emptyset();
    
    % Basic parameters (minimal for speed)
    EEG.setname = sprintf('P%02d_mock', participant_id);
    EEG.nbchan = 8;  % Just 8 channels instead of full 64
    EEG.srate = 250; % Lower sample rate for speed
    EEG.xmin = 0;
    
    % Channel locations (subset of standard 10-20)
    chan_labels = {'Fz', 'Cz', 'Pz', 'Oz', 'F3', 'F4', 'P3', 'P4'};
    for i = 1:EEG.nbchan
        EEG.chanlocs(i).labels = chan_labels{i};
    end
    
    % Generate synthetic data (5 minutes = 300 seconds)
    duration_seconds = 300;
    EEG.pnts = duration_seconds * EEG.srate;
    EEG.trials = 1;
    EEG.times = (0:EEG.pnts-1) / EEG.srate;
    
    % Generate realistic-looking EEG (pink noise + alpha oscillations)
    EEG.data = zeros(EEG.nbchan, EEG.pnts);
    for ch = 1:EEG.nbchan
        % Pink noise background
        pink_noise = cumsum(randn(1, EEG.pnts)) * 5;
        pink_noise = pink_noise - mean(pink_noise);
        
        % Add alpha oscillations (8-12 Hz)
        t = EEG.times;
        alpha = 10 * sin(2 * pi * 10 * t + randn * 2 * pi);
        
        EEG.data(ch, :) = pink_noise + alpha;
    end
    
    % Add events for each condition
    conditions = fieldnames(config.conditions);
    event_idx = 1;
    latency = EEG.srate * 10; % Start 10 seconds in
    
    for i = 1:length(conditions)
        cond = conditions{i};
        EEG.event(event_idx).type = cond;
        EEG.event(event_idx).latency = latency;
        EEG.event(event_idx).duration = config.conditions.(cond).duration;
        
        % Space events 30 seconds apart
        latency = latency + 30 * EEG.srate;
        event_idx = event_idx + 1;
    end
    
    % Check and save
    EEG = eeg_checkset(EEG);
    
    % Save to cleaned_eeg folder
    output_path = config.paths.cleaned_eeg;
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end
    
    pop_saveset(EEG, 'filename', sprintf('P%02d_cleaned.set', participant_id), ...
                'filepath', output_path);
    
    fprintf('Created mock data for P%02d: %d channels, %.1f seconds\n', ...
            participant_id, EEG.nbchan, duration_seconds);
end
