addpath('D:\PhD_Projects\TSST_Stress_Workload_Pipeline\scripts\utils')

% Load YAML config
config = yaml.loadFile('D:/PhD_Projects/TSST_Stress_Workload_Pipeline/scripts/config.yaml');


%Initialize EEGLAB
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;


% Define participant numbers
participant_numbers = [40];
  

% Define conditions (now 29 items)
conditions = {
    'Primary_Calibrations', 'Blink_Calibration', 'Movement_Baseline', ...
    'Pre_Exposure_Blank_Baseline', 'Pre_Exposure_Room_Baseline', ...
    'Post_Exposure_Blank_Baseline', 'Post_Exposure_Room_Baseline', ... 
    'HighStress_HighCog1022_Preamble', 'HighStress_HighCog2043_Preamble', ...
    'HighStress_LowCog_Preamble', 'LowStress_HighCog1022_Preamble', ...
    'LowStress_HighCog2043_Preamble', 'LowStress_LowCog_Preamble', ...
    'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', ...
    'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', ...
    'HighStress_LowCog_Task', 'LowStress_LowCog_Task', ...
    'HighStress_HighCog1022_Finish', 'HighStress_HighCog2043_Finish', ...
    'HighStress_LowCog_Finish', 'LowStress_HighCog1022_Finish', ...
    'LowStress_HighCog2043_Finish', 'LowStress_LowCog_Finish', ...  % Added missing condition
    'Relaxation1', 'Relaxation2', 'Relaxation3', 'Relaxation4'};


% Define condition durations in seconds
condition_durations = containers.Map(...
    conditions, ...  % 28 conditions
    [120, 60, 90, 60, 60, 60, 60, ...       % 7 elements
    30, 30, 30, 30, 30, 30, ...             % 6 elements
    180, 180, 180, 180, 180, 180 ...        % 6 elements
    15, 15, 15, 15, 15, 15, ...             % 6 elements
    180, 180, 180, 180]...                  % 4 elements
);  % Total = 7+6+6+6+4 = 29 elements


% Loop through each participant
for i = 1:length(participant_numbers)
    participant_num = participant_numbers(i);
    
    raw_data_path = config.paths.eeg_data;
    cleaned_data_folder = fullfile(config.paths.cleaned_eeg);
    logs_folder         = fullfile(config.paths.logs);
    disp(sprintf('Debug: Processing Participant %d.', participant_num));

    % Define file paths
    raw_file = sprintf('P%02d_raw.set', participant_num);
    vis_path = fullfile(config.paths.vis, sprintf('P%02d', participant_num));
    qc_path = fullfile(config.paths.eeg_qc);
    basic_cleaned_mat_filename = fullfile(config.paths.filtered_eeg, sprintf('P%02d_filtered.mat', participant_num));
    basic_cleaned_set_filename = fullfile(config.paths.filtered_eeg, sprintf('P%02d_filtered.set', participant_num));
    events_path = fullfile(config.paths.events, sprintf('P%02d_events.csv', participant_num));
    logfile = fullfile(config.paths.logs, sprintf('P%02d_processing_log.txt', participant_num));

    try

        % --- Step 1: Load Raw Data ---
       disp('======================================');
       disp('Step 1: Load Raw Data');
       disp('======================================');

        EEG = load_raw_data(raw_file, raw_data_path, events_path, logfile, 'Raw', vis_path);
        EEG.data = double(EEG.data);

        % Define filename for raw EEG data
        raw_data_filename = fullfile(config.paths.eeg_data, sprintf('Raw_EEG_Data_P%02d.mat', participant_num));
        
        % Save raw EEG data matrix
        raw_EEG = EEG.data;
        save(raw_data_filename, 'raw_EEG', '-v7.3');
        log_message(logfile, sprintf('Participant %d: Raw EEG data saved to %s.', participant_num, raw_data_filename));
        
        disp(['Raw EEG data saved as: ', raw_data_filename]);

         % --- Step 2: Apply Basic Cleaning ---
        disp('======================================');
        disp('Step 2: Apply Basic Cleaning to Raw Data');
        disp('======================================');

        EEG = basic_cleaning(EEG, logfile, raw_data_path, participant_num, vis_path);
        basic_EEG_matrix = EEG.data;
        save(basic_cleaned_mat_filename, 'basic_EEG_matrix', '-v7.3');
        log_message(logfile, sprintf('Participant %d: Basic cleaned EEG matrix saved to %s.', ...
                  participant_num, basic_cleaned_mat_filename));

        EEG.setname = sprintf('Basic_Cleaned_EEG_Data_P%d', participant_num);
        
        [save_path, save_name, ext] = fileparts(basic_cleaned_set_filename);
        pop_saveset(EEG, 'filename', [char(save_name), char(ext)], 'filepath', char(save_path));
        
        log_message(logfile, sprintf('Participant %d: Basic cleaned EEG saved as .set to %s.', ...
                  participant_num, basic_cleaned_set_filename))
        disp(['Basic cleaned EEG data saved as: ', basic_cleaned_mat_filename, ' and ', basic_cleaned_set_filename]);

         % --- Step 3: Apply Advanced Cleaning on Basic Cleaned Continuous Data ---
         disp('======================================');
         disp('Step 3: Apply Advanced Cleaning on Basic Cleaned Continuous Data ');
         disp('======================================');

        cleaned_EEG = advanced_cleaning(basic_cleaned_set_filename, cleaned_data_folder, logfile, participant_num, events_path, vis_path, qc_path);
        advanced_cleaned_filename = fullfile(cleaned_data_folder, sprintf('P%d_cleaned.mat', participant_num));
        
      
    catch ME
        log_message(logfile, sprintf('Error occurred: %s', ME.message));
        continue;
    end
end

% --- Helper Functions ---

function EEG = load_raw_data(file, path, events, log, stage, vis_path)
    try
        fprintf('%s: Starting data loading process.\n', stage);

        % Step 1: Load raw dataset
        fprintf('%s: Loading dataset from %s.\n', stage, fullfile(path, file));
        EEG = pop_loadset('filename', char(file), 'filepath', char(path));
        
        if isempty(EEG.data)
            fprintf('%s: Dataset loaded but contains no data.\n', stage);
            return;
        else
            fprintf('%s: Dataset loaded with %d channels and %d data points.\n', ...
                stage, size(EEG.data, 1), size(EEG.data, 2));
            save_visualization(EEG, vis_path, sprintf('%s_loaded.png', stage));
        end

        %% 


        disp('Debug: Raw data loaded successfully. Proceeding to add trigger channel.');

        % Assign channel locations for standard EEG channels first
        fprintf('%s: Assigning standard channel locations.\n', stage);
        EEG = pop_chanedit(EEG, 'lookup', 'C:\Users\Joe\Documents\MATLAB\eeglab_current\eeglab2024.0\sample_locs\NA-271.elc');
        EEG.etc.orig_chanlocs = EEG.chanlocs;    % <-- save the “complete” set before any channels get dropped
        
        fprintf('%s: Before pop_chanedit, total channels: %d', stage, EEG.nbchan);

        if isempty(EEG.chanlocs)
            fprintf('%s: Warning - Channel locations are empty after pop_chanedit.\n', stage);
        end

        fprintf('%s: Adding trigger channel.\n', stage);
        num_samples = size(EEG.data, 2);
        EEG.data(end + 1, :) = zeros(1, num_samples);  % Add trigger channel initialized to 0
        EEG.nbchan = EEG.nbchan + 1;  % Increase channel count
        fprintf('%s: Trigger channel added. New total channels: %d.', stage, EEG.nbchan);

        % Ensure chanlocs is properly initialized before adding the trigger channel
        if isempty(EEG.chanlocs)
            fprintf('%s: Initializing chanlocs structure.\n', stage);
            EEG.chanlocs = struct('labels', {}, 'X', {}, 'Y', {}, 'Z', {});
        end
        EEG.chanlocs(end + 1).labels = 'Trigger';
        EEG.chanlocs(end).X = [];
        EEG.chanlocs(end).Y = [];
        EEG.chanlocs(end).Z = [];
        EEG.chanlocs(end).type = 'trigger';

        fprintf('%s: Trigger channel label added.\n', stage);

        % Step 4: Import events and populate trigger channel
        fprintf('%s: Reading events from %s.\n', stage, events);
        event_data_table = readtable(events);  % Read event file (assuming CSV format with headers)
        
        % Validate CSV content
        disp('First few rows of event data:');
        disp(event_data_table(1:5, :));
        
        % Extract latency and type columns safely
        if any(strcmpi(event_data_table.Properties.VariableNames, 'latency')) && ...
           any(strcmpi(event_data_table.Properties.VariableNames, 'type'))
        
            latencies = event_data_table.latency;  
            event_types = string(event_data_table.type);  
        else
            error('CSV does not contain expected columns: Latency and Type');
        end
        
        fprintf('%s: Successfully extracted %d latency values.\n', stage, length(latencies));
        
        % Populate EEG.event manually if it is empty
        EEG.event = [];  
        
        for i = 1:length(latencies)
            EEG.event(i).latency = latencies(i);
            EEG.event(i).type = event_types(i);
        end
        
        EEG = eeg_checkset(EEG);  
        
        fprintf('%s: EEG.event populated with %d events.\n', stage, length(EEG.event));
        disp('First few EEG events:');
        disp(EEG.event(1:min(5, length(EEG.event))));


        % Define event label corrections to match condition names
        label_changes = {
            'High_Stress', 'HighStress';
            'Low_Stress', 'LowStress';
            'Subtraction', 'HighCog';
            'Addition', 'LowCog';
            'Fixation_Cross', 'Baseline';
            'Forest', 'Relaxation';
        };

        % Define condition mapping (ensure it is available before processing events)
        condition_mapping = containers.Map(...
            {'Primary_Calibrations', 'Blink_Calibration', 'Movement_Baseline', ...
            'Pre_Exposure_Blank_Baseline', 'Pre_Exposure_Room_Baseline', ...
            'Post_Exposure_Blank_Baseline', 'Post_Exposure_Room_Baseline', ...
            'HighStress_HighCog1022_Preamble', 'HighStress_HighCog2043_Preamble', ...
            'HighStress_LowCog_Preamble', 'LowStress_HighCog1022_Preamble', ...
            'LowStress_HighCog2043_Preamble', 'LowStress_LowCog_Preamble', ...
            'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', ...
            'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', ...
            'HighStress_LowCog_Task', 'LowStress_LowCog_Task', ...
            'HighStress_HighCog1022_Finish', 'HighStress_HighCog2043_Finish', ...
            'HighStress_LowCog_Finish', 'LowStress_HighCog1022_Finish', ...
            'LowStress_HighCog2043_Finish', 'LowStress_LowCog_Finish', ...
            'Relaxation1', 'Relaxation2', 'Relaxation3', 'Relaxation4'}, ...
            1:29 ...
        );


        % Apply renaming to event labels
        fprintf('%s: Renaming event labels.\n', stage);
        for i = 1:length(event_types)
            for j = 1:size(label_changes, 1)
                event_types{i} = strrep(event_types{i}, label_changes{j, 1}, label_changes{j, 2});
            end
            event_types{i} = strtrim(event_types{i});
        end

        % Apply renamed event labels back to the EEG structure
        for idx = 1:length(EEG.event)
            EEG.event(idx).type = event_types{idx};
        end
        fprintf('%s: Event labels renamed successfully.\n', stage);

        % Insert triggers into the trigger channel
        fprintf('%s: Populating trigger channel with events.\n', stage);
        for i = 1:height(event_data_table)  % Use height instead of size for table
            latency = round(latencies(i));  % Convert to sample index
           event_type = (event_types(i)); 
        
            % Ensure renaming aligns with condition mapping
            for j = 1:size(label_changes, 1)
                event_type = strrep(event_type, lower(label_changes{j, 1}), label_changes{j, 2});
            end
        
            if isKey(condition_mapping, event_type)
                condition_code = condition_mapping(event_type);
                if latency > 0 && latency <= num_samples
                    EEG.data(end, latency) = condition_code;  % Assign condition code
                else
                    fprintf('Warning: Latency %d is out of bounds.\n', latency);
                end
            else
                fprintf('Warning: Unrecognized event type %s.\n', event_type);
            end
        end
        fprintf('%s: Events mapped to trigger channel.\n', stage);


        % ── Immediately after adding the trigger row ───────────────────────────
        trigger_idx = EEG.nbchan;          % 129 once the row is added
        eeg_idx     = 1:(trigger_idx-1);   % pure EEG channels 1-128
        
        % Optionally store them so every function can see them
        EEG.userdata.trigger_idx = trigger_idx;
        EEG.userdata.eeg_idx     = eeg_idx;


        % Step 5: Assign channel locations for standard EEG channels only
        EEG = pop_chanedit(EEG, 'lookup', 'C:\Users\Joe\Documents\MATLAB\eeglab_current\eeglab2024.0\sample_locs\NA-271.elc');
        
        if isempty(EEG.chanlocs(1).X)
            fprintf('%s: Failed to assign standard channel locations.\n', stage);
        else
            fprintf('%s: Channel locations assigned for EEG channels.\n', stage);
            save_visualization(EEG, vis_path, sprintf('%s_chanlocs.png', stage));
        end

    catch ME
        fprintf('Error in %s: %s', stage, ME.message);
        rethrow(ME);
    end
end

function EEG = basic_cleaning(EEG, log, outPath, pNum, vis_path)
    try
        eegIdx = 1:128;
        if EEG.srate ~= 125
            EEG = pop_resample(EEG, 125);
            log_message(log, sprintf('P%d: resampled to 125 Hz.', pNum));
        end

        % Band-pass 1–49 Hz on channels 1:128
        EEG = pop_eegfiltnew(EEG, ...
            'locutoff',  1, ...
            'hicutoff', 49, ...
            'channels',  eegIdx);

        log_message(log, sprintf('P%d: 1–49 Hz band-pass applied.', pNum));

        EEG = pop_cleanline(EEG, 'linefreqs', 50, 'chanlist', eegIdx, 'sigtype', 'Channels');
        log_message(log, sprintf('P%d: CleanLine 50 Hz applied.', pNum));

        if ismember(pNum, 1:7)
            EEG = pop_eegfiltnew(EEG, ...
                'locutoff',  24.5, ...
                'hicutoff',  25.5, ...
                'revfilt',    1, ...
                'channels',  eegIdx);
            log_message(log, sprintf('P%d: 25 Hz notch applied.', pNum));
        end


        save_visualization(EEG, vis_path, sprintf('P%d_basic_clean.png', pNum));
    catch ME
        log_message(log, sprintf('Basic cleaning error P%d: %s', pNum, ME.message));
        rethrow(ME);
    end
end


function EEG = advanced_cleaning(file, cleaned_data_folder, log, participant_num, events, vis_path, qc_path)
    try
        % Debug: show full path
        disp(['Full file path passed in: ', char(file)]);


        % Split filename and path properly
        [filepath, filename, ext] = fileparts(file);
        EEG = pop_loadset('filename', [char(filename), char(ext)], 'filepath', char(filepath));
        
        % Check number of channels after loading
        disp(['Number of channels after loading: ', num2str(size(EEG.data, 1))]);

        % Remove trigger channel from data and chanlocs
        trigger_channel_index = 129;
        
        EEG.data(trigger_channel_index, :) = [];  % Remove trigger channel data
        EEG.chanlocs(trigger_channel_index) = []; % Remove corresponding channel location
        EEG.nbchan = EEG.nbchan - 1;  % Update channel count

         EEG.etc.orig_chanlocs = EEG.chanlocs;
        
        % Verify channel consistency before proceeding
        disp(['Number of channels in data: ', num2str(size(EEG.data, 1))]);
        disp(['Number of channels in chanlocs: ', num2str(length(EEG.chanlocs))]);

        % Check number of channels after removing the trigger channel
        disp(['Number of channels after removing trigger: ', num2str(size(EEG.data, 1))]);

        % Run the cleaning
        [EEG, com] = clean_artifacts(EEG, ...
            'FlatlineCriterion',  5, ...
            'ChannelCriterion',   0.85, ...
            'LineNoiseCriterion', 4, ...
            'BurstCriterion',     20, ...
            'WindowCriterion',    0.8);
    
        log_message(log, sprintf('P%d: clean_artifacts completed.', participant_num));

        % ========== DEBUG INFO ==========
        if isempty(EEG.chanlocs)
            warning('P%d: EEG.chanlocs is empty after cleaning!', participant_num);
        else
            log_message(log, sprintf('P%d: EEG.chanlocs length: %d', participant_num, length(EEG.chanlocs)));
        end
        
        % Force creation of clean_channel_mask if missing
        if ~isfield(EEG.etc, 'clean_channel_mask')
            warning('P%d: clean_channel_mask not found – assuming all channels retained.', participant_num);
            EEG.etc.clean_channel_mask = true(1, length(EEG.etc.orig_chanlocs));  % assume none rejected
        end
        
        if ~isfield(EEG.etc, 'orig_chanlocs')
            error('P%d: Missing field EEG.etc.orig_chanlocs after cleaning.', participant_num);
        end

        mask     = EEG.etc.clean_channel_mask;
        origLocs = EEG.etc.orig_chanlocs;
        
        if length(origLocs) ~= length(mask)
            error('P%d: Mismatch between orig_chanlocs (%d) and clean_channel_mask (%d).', ...
                participant_num, length(origLocs), length(mask));
        end
        
        % Identify bad channels
        badIdx    = ~mask;
        badLabels = { origLocs(badIdx).labels }';
        
        log_message(log, sprintf('P%d: %d bad channels identified.', participant_num, sum(badIdx)));
        log_message(log, sprintf('P%d: Bad channels: %s', participant_num, strjoin(badLabels, ', ')));

        % Run AMICA
        EEG = run_amica(EEG, log, cleaned_data_folder, participant_num, vis_path);
        save_visualization(EEG, vis_path, sprintf('Preprocessing_P%d_amica.png', participant_num));

        % Apply ICLabel
        EEG = iclabel(EEG);
        log_message(log, sprintf('Preprocessing: ICLabel applied. Components classified: %d.', size(EEG.icaweights, 1)));
        save_visualization(EEG, vis_path, sprintf('Preprocessing_P%d_iclabel.png', participant_num));

        % Remove Artifacts Using ICLabel
        EEG = flag_and_remove_artifacts(EEG, log);
        save_visualization(EEG, vis_path, sprintf('Preprocessing_P%d_after_artifact_removal.png', participant_num));

        % Interpolate bad channels and Re-reference the data (excluding the trigger channel)
        EEG = pop_interp(EEG, EEG.etc.orig_chanlocs,'spherical');
        EEG = pop_reref(EEG, []);  
        log_message(log, sprintf('Preprocessing: Data re-referenced. Mean value: %f.', mean(EEG.data(:))));
        save_visualization(EEG, vis_path, sprintf('Preprocessing_P%d_rereferenced.png', participant_num));


        % === Step 7: Add back the trigger channel and map events ===
        disp('======================================');
        disp('Adding back Trigger Channel and Mapping Events');
        disp('======================================');

        % Add the trigger channel back to EEG data
        EEG.data(trigger_channel_index, :) = zeros(1, size(EEG.data, 2));  % Initialize trigger channel
        EEG.nbchan = EEG.nbchan + 1;  % Update channel count
        
        % Ensure EEG.chanlocs is initialized correctly
        if isempty(EEG.chanlocs) || length(EEG.chanlocs) < trigger_channel_index
            EEG.chanlocs(trigger_channel_index).labels = 'Trigger';
            EEG.chanlocs(trigger_channel_index).X = [];
            EEG.chanlocs(trigger_channel_index).Y = [];
            EEG.chanlocs(trigger_channel_index).Z = [];
            EEG.chanlocs(trigger_channel_index).type = 'trigger';
        end

        disp('Trigger channel restored and initialized.');

        % Read event data and populate trigger channel
        event_data_table = readtable(events);  % Assuming CSV format with headers
        
        disp('First few rows of event data:');
        disp(event_data_table(1:5, :));

        % Validate event data structure
        if all(ismember({'latency', 'type'}, event_data_table.Properties.VariableNames))
            latencies = event_data_table.latency;
            event_types = string(event_data_table.type);
        else
            error('CSV does not contain expected columns: Latency and Type');
        end

        disp(sprintf('Successfully extracted %d latency values.', length(latencies)));

        % Standardize event labels first
        label_changes = {
            'High_Stress', 'HighStress';
            'Low_Stress', 'LowStress';
            'Subtraction', 'HighCog';
            'Addition', 'LowCog';
            'Fixation_Cross', 'Baseline';
            'Forest', 'Relaxation';
        };
        
        for i = 1:length(event_types)
            for j = 1:size(label_changes, 1)
                event_types{i} = strrep(event_types{i}, label_changes{j, 1}, label_changes{j, 2});
            end
            event_types{i} = strtrim(event_types{i});
        end
        
        disp('Event labels renamed successfully.');

        % Map events to the trigger channel
        condition_mapping = containers.Map(...
            {'Primary_Calibrations', 'Blink_Calibration', 'Movement_Baseline', ...
            'Pre_Exposure_Blank_Baseline', 'Pre_Exposure_Room_Baseline', ...
            'Post_Exposure_Blank_Baseline', 'Post_Exposure_Room_Baseline', ...
            'HighStress_HighCog1022_Preamble', 'HighStress_HighCog2043_Preamble', ...
            'HighStress_LowCog_Preamble', 'LowStress_HighCog1022_Preamble', ...
            'LowStress_HighCog2043_Preamble', 'LowStress_LowCog_Preamble', ...
            'HighStress_HighCog1022_Task', 'HighStress_HighCog2043_Task', ...
            'LowStress_HighCog1022_Task', 'LowStress_HighCog2043_Task', ...
            'HighStress_LowCog_Task', 'LowStress_LowCog_Task', ...
            'HighStress_HighCog1022_Finish', 'HighStress_HighCog2043_Finish', ...
            'HighStress_LowCog_Finish', 'LowStress_HighCog1022_Finish', ...
            'LowStress_HighCog2043_Finish', 'LowStress_LowCog_Finish', ...
            'Relaxation1', 'Relaxation2', 'Relaxation3', 'Relaxation4'}, ...
            1:29 ...
        );

        for i = 1:height(event_data_table)
            latency = round(latencies(i));
            event_type = event_types(i);

            % Apply renaming corrections before mapping
            for j = 1:size(label_changes, 1)
                event_type = strrep(event_type, (label_changes{j, 1}), label_changes{j, 2});
            end

            if isKey(condition_mapping, event_type) && latency > 0 && latency <= size(EEG.data, 2)
                EEG.data(trigger_channel_index, latency) = condition_mapping(event_type);
            else
                fprintf('Warning: Unrecognized or out-of-bounds event at latency %d\n', latency);
            end
        end

        disp('Events mapped to trigger channel.');

       
        % Save cleaned EEG matrix with 129 channels
        % Define path for saving advanced cleaned EEG .mat file
        advanced_cleaned_mat_filename = fullfile(cleaned_data_folder, sprintf('P%d_cleaned.mat', participant_num));
        
        % Extract data and save
        Advanced_cleaned_EEG = double(EEG.data(1:129, :));  % Save only the first 129 channels
        save(advanced_cleaned_mat_filename, 'Advanced_cleaned_EEG', '-v7.3');
        
        % Log
        log_message(log, sprintf('Participant %d: Advanced cleaned EEG data saved as .mat to %s.', ...
                  participant_num, advanced_cleaned_mat_filename));
        
        disp(['Advanced cleaned EEG data saved as: ', advanced_cleaned_mat_filename]);

         %--- Preflight checks for QC fields ---

        uniqueTypes = unique({EEG.event.type});
        eventwiseRetention = struct();
        
        for i = 1:numel(uniqueTypes)
            type = uniqueTypes{i};
            indices = find(strcmp({EEG.event.type}, type));
            latencies = round([EEG.event(indices).latency]);
        
            validLatencies = latencies(latencies > 0 & latencies <= length(EEG.etc.clean_sample_mask));
            retained = EEG.etc.clean_sample_mask(validLatencies);
            
            eventwiseRetention.(type).nEvents  = numel(validLatencies);
            eventwiseRetention.(type).nKept    = sum(retained);
            eventwiseRetention.(type).percKept = 100 * sum(retained) / numel(validLatencies);
        end


        % 1) badLabels must exist
        if ~exist('badLabels','var')
            error('Cannot compute qc.nBad or qc.badChannelLabels: ''badLabels'' does not exist.');
        end
        
        % 2) EEG.etc.clean_sample_mask must exist
        if ~isfield(EEG.etc, 'clean_sample_mask')
            error('Cannot compute qc.percASRrepaired: EEG.etc.clean_sample_mask is missing.');
        end
        
        % 3) EEG.etc.badICs must exist
        if ~isfield(EEG.etc, 'badICs')
            error('Cannot compute qc.ICsRemoved: EEG.etc.badICs is missing.');
        end
        
        % 3) Build your QC struct exactly as before, plus a field for the names
        qc.nBad            = numel(badLabels);
        qc.badChannelLabels = badLabels;                      % <--- new field
        qc.percASRrepaired = 100 * mean(~EEG.etc.clean_sample_mask);
        qc.ICsRemoved = numel( EEG.etc.badICs );
        qc.samplesRetained = sum(EEG.etc.clean_sample_mask);
        qc.totalSamples    = length(EEG.etc.clean_sample_mask);
        qc.eventwiseRetention = eventwiseRetention;
        qc.percSamplesRetained = 100 * qc.samplesRetained / qc.totalSamples;

        
        % 4) Save everything (including the list of bad channels) to disk
        % After building qc struct...
        save(fullfile(qc_path,sprintf('P%d_qc.mat',participant_num)), 'qc');
        % And also write a simple text log:
        fid = fopen(fullfile(qc_path,sprintf('QC_P%d.txt',participant_num)), 'a');
        fprintf(fid, 'Samples retained overall: %.1f%% (%d / %d)\n', ...
            qc.percSamplesRetained, qc.samplesRetained, qc.totalSamples);
        
        % Add full QC metric logging here
        fprintf(fid, 'Number of bad channels: %d\n', qc.nBad);
        fprintf(fid, 'Bad channel labels: %s\n', strjoin(qc.badChannelLabels, ', '));
        fprintf(fid, 'Percent ASR-repaired: %.2f%%\n', qc.percASRrepaired);
        fprintf(fid, 'Number of ICs removed: %d\n', qc.ICsRemoved);
        
        fprintf(fid, '\n--- Retention by event type ---\n');
        fields = fieldnames(qc.eventwiseRetention);
        for i = 1:length(fields)
            f = fields{i};
            fprintf(fid, '%s: %.1f%% (%d / %d)\n', f, ...
                qc.eventwiseRetention.(f).percKept, ...
                qc.eventwiseRetention.(f).nKept, ...
                qc.eventwiseRetention.(f).nEvents);
        end
        fclose(fid);

    catch ME
        log_message(log, sprintf('Error in Advanced Cleaning: %s', ME.message));
        rethrow(ME);
    end
end


function EEG = run_amica(EEG, log, cleaned_data_folder, participant_num, vis_path)
    % Define parameters for AMICA
    num_models   = 1;    % Typically 1 unless multiple models are needed
    numprocs     = 1;    % Adjust to number of cores you want to use
    max_threads  = 4;    % Adjust to number of threads you want to use
    max_iter     = 400;  % Hard ceiling on iterations
    writeStep    = 10;   % Record LL every 10 iterations
    
    % Define a unique output directory for AMICA for each participant
    outdir = fullfile(pwd, sprintf('amicaouttmp_%d', participant_num));
    
    % Remove any old directory to avoid conflicts
    if exist(outdir, 'dir')
        try  
            rmdir(outdir, 's');
        catch ME
            warning('Could not remove existing AMICA output directory: %s\nError: %s', outdir, ME.message);
        end
    end
    mkdir(outdir);

    % Run AMICA (version 15) with LL‐logging turned on
    %
    %   - 'write_LLt', 1:   tell AMICA to save its log‐likelihood at every writestep
    %   - 'writestep', writeStep: write LL every writeStep iterations
    %   - 'max_iter', max_iter:   stop after max_iter (unless convergence hits first)
    %
    % The call now returns a “mods” struct that contains mods.LL
    [weights, sphere, mods] = runamica15(EEG.data, ...
        'num_models',   num_models, ...
        'outdir',       outdir, ...
        'numprocs',     numprocs, ...
        'max_threads',  max_threads, ...
        'max_iter',     max_iter, ...
        'write_LLt',    1, ...
        'writestep',    writeStep);    

    % (1) Apply weights + sphere to EEG
    EEG.icaweights = weights;
    EEG.icasphere  = sphere;
    EEG = eeg_checkset(EEG);
    log_message(log, 'ICA weights and sphere applied.');
    save_visualization(EEG, vis_path, sprintf('EEG_%d_AfterICA.png', participant_num));

    save_visualization(EEG, vis_path, sprintf('Preprocessing_P%d_after_artifact_removal.png', participant_num));

    % (2) Immediately append AMICA’s LL trace into the same logfile
    if isfield(mods, 'LL')
        LL_trace = mods.LL; 
        log_message(log, '--- AMICA LL (every 10 iters) ---');
        for idx = 1:length(LL_trace)
            if mod(idx, writeStep)==0   % writeStep=10
                log_message(log, sprintf('  iter %d → LL = %.4f', idx, LL_trace(idx)));
            end
        end
        log_message(log, '--- End of AMICA LL trace ---');
    end


    % Cleanup AMICA output directory to avoid clutter
    try
        rmdir(outdir, 's');
    catch ME
        warning('Could not remove AMICA output directory after processing: %s\nError: %s', outdir, ME.message);
    end
end


function EEG = flag_and_remove_artifacts(EEG, log)
    try
        % Ensure ICLabel has been run
        if ~isfield(EEG.etc, 'ic_classification') || ...
           ~isfield(EEG.etc.ic_classification, 'ICLabel')
            EEG = iclabel(EEG);
            log_message(log, 'ICLabel applied.');
        end

        % Extract ICLabel probabilities: [Brain Muscle Eye Heart LineNoise ChannelNoise Other]
        probs     = EEG.etc.ic_classification.ICLabel.classifications;
        eyeProb   = probs(:, 3);  % column 3 = Eye
        muscleProb= probs(:, 2);  % column 2 = Muscle
        toRemove  = find(eyeProb >= 0.9 | muscleProb >= 0.9);

        % If none flagged, log and return
        if isempty(toRemove)
            EEG.etc.badICs = [];   % explicitly store as empty
            log_message(log, 'Artifact Removal: No ICs flagged for removal.');
            return;
        end

        % Log which ICs are removed
        for idx = toRemove'
            if eyeProb(idx) >= 0.9
                log_message(log, sprintf('Removing IC %d (Eye, p=%.2f).', idx, eyeProb(idx)));
            else % muscleProb(idx) >= 0.9
                log_message(log, sprintf('Removing IC %d (Muscle, p=%.2f).', idx, muscleProb(idx)));
            end
        end

        % Store the badICs in EEG structure
        EEG.etc.badICs = toRemove;

        % Remove flagged components
        EEG = pop_subcomp(EEG, toRemove, 0);
        log_message(log, sprintf('Artifact Removal: Removed %d IC(s).', numel(toRemove)));

    catch ME
        log_message(log, sprintf('Error in Artifact Removal: %s', ME.message));
        rethrow(ME);
    end
end


function log_message(logfile, message)
    % Log messages with timestamps
    fid = fopen(logfile, 'a');
    if fid == -1
        error('Cannot open log file.');
    end
    fprintf(fid, '%s: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'), message);
    fclose(fid);
end

function save_visualization(EEG, folder, filename)
    if ~exist(folder, 'dir')
        mkdir(folder);  % 🛡️ Create the folder if it doesn't exist
    end
    figure;
    pop_eegplot(EEG, 1, 1, 1);
    saveas(gcf, fullfile(folder, filename));
    close(gcf);
end



