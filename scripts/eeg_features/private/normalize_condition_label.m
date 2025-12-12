function cond = normalize_condition_label(raw_label, config_cond)
% NORMALIZE_CONDITION_LABEL Map raw event label to canonical condition name
%
% Inputs:
%   raw_label   - Raw event label from events file
%   config_cond - Conditions configuration
%
% Outputs:
%   cond - Canonical condition name, or empty string if not found

    cond = '';
    
    if isempty(raw_label)
        return;
    end
    
    % Convert to char if string
    if isstring(raw_label)
        raw_label = char(raw_label);
    end
    
    raw_label = strtrim(raw_label);
    
    % Check each condition in config
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
