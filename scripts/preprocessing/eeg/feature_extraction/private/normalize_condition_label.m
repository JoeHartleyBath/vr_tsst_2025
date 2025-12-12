function cond = normalize_condition_label(raw_label, config_cond)
% NORMALIZE_CONDITION_LABEL Map raw event label to canonical condition name
%
% Inputs:
%   raw_label   - Raw event label from EEG.event (string, char, or numeric)
%   config_cond - Conditions configuration
%
% Outputs:
%   cond - Canonical condition name, or empty string if not found

    cond = '';
    
    if isempty(raw_label)
        return;
    end
    
    % Convert numeric to string
    if isnumeric(raw_label)
        raw_label = num2str(raw_label);
    end
    
    % Convert to char if string
    if isstring(raw_label)
        raw_label = char(raw_label);
    end
    
    raw_label = strtrim(raw_label);
    
    % First, check if this is a numeric event code from export_event_labels
    if isfield(config_cond, 'export_event_labels')
        event_codes = config_cond.export_event_labels;
        label_names = fieldnames(event_codes);
        
        for i = 1:length(label_names)
            label_name = label_names{i};
            code = event_codes.(label_name);
            
            % Check if raw_label matches this numeric code
            if strcmp(raw_label, num2str(code))
                cond = label_name;
                return;
            end
        end
    end
    
    % If not found in export_event_labels, check condition names and aliases
    cond_names = fieldnames(config_cond.conditions);
    for i = 1:length(cond_names)
        cond_name = cond_names{i};
        cond_info = config_cond.conditions.(cond_name);
        
        % Check aliases
        if isfield(cond_info, 'aliases')
            for j = 1:length(cond_info.aliases)
                if contains(raw_label, cond_info.aliases{j}, 'IgnoreCase', true)
                    cond = cond_name;
                    return;
                end
            end
        end
        
        % Check exact match
        if strcmpi(raw_label, cond_name)
            cond = cond_name;
            return;
        end
    end
end
