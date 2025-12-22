% Check what events are present in participants with missing conditions
% This helps diagnose why some conditions weren't extracted

% Participants with missing conditions (from EEG features completeness check)
participants_to_check = [1 : 28];

% Load configs
config_gen = yaml.ReadYaml(fullfile('config', 'general.yaml'));
config_cond = yaml.ReadYaml(fullfile('config', 'conditions.yaml'));

% Expected conditions (from config)
expected_conditions = fieldnames(config_cond.conditions);
fprintf('Expected conditions (%d):\n', length(expected_conditions));
for i = 1:length(expected_conditions)
    fprintf('  - %s\n', expected_conditions{i});
end
fprintf('\n');

% Check each participant
for p = participants_to_check
    fprintf('=== P%02d ===\n', p);
    
    % Load .set file
    set_file = fullfile(config_gen.paths.cleaned_eeg, sprintf('P%02d_cleaned.set', p));
    if ~isfile(set_file)
        fprintf('  ✗ .set file not found\n\n');
        continue;
    end
    
    try
        EEG = pop_loadset('filename', sprintf('P%02d_cleaned.set', p), ...
                          'filepath', config_gen.paths.cleaned_eeg);
        
        % Get all event labels
        if isfield(EEG, 'event') && ~isempty(EEG.event)
            fprintf('  Found %d events in .set file:\n', length(EEG.event));
            
            % Count unique event types
            event_types = {};
            for i = 1:length(EEG.event)
                raw_label = EEG.event(i).type;
                if isnumeric(raw_label)
                    raw_label = num2str(raw_label);
                end
                event_types{end+1} = raw_label; %#ok<AGROW>
            end
            
            unique_events = unique(event_types);
            fprintf('  Unique event labels (%d):\n', length(unique_events));
            for i = 1:length(unique_events)
                count = sum(strcmp(event_types, unique_events{i}));
                fprintf('    - "%s" (n=%d)\n', unique_events{i}, count);
                
                % Try to normalize this label
                norm_label = normalize_condition_label(unique_events{i}, config_cond);
                if ~isempty(norm_label)
                    fprintf('      → maps to: %s\n', norm_label);
                else
                    fprintf('      → NOT MATCHED (skipped)\n');
                end
            end
        else
            fprintf('  ✗ No events found\n');
        end
        
    catch ME
        fprintf('  ✗ Error loading: %s\n', ME.message);
    end
    
    fprintf('\n');
end

fprintf('Done.\n');
