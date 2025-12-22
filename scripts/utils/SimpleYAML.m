classdef SimpleYAML
    % SimpleYAML - Minimal YAML parser for simple key:value configs
    % Handles basic YAML like:
    %   key: value
    %   nested:
    %     subkey: subvalue
    % Returns a MATLAB struct.
    
    methods(Static)
        function config = readFile(filepath)
            % Read a YAML file and return as struct
            if ~exist(filepath, 'file')
                error('SimpleYAML:FileNotFound', 'YAML file not found: %s', filepath);
            end
            
            fid = fopen(filepath, 'r');
            lines = textscan(fid, '%s', 'Delimiter', '\n', 'ReturnOnError', false);
            fclose(fid);
            lines = lines{1};
            
            config = SimpleYAML.parseLines(lines, 0);
        end
        
        function config = parseLines(lines, baseIndent)
            % Recursively parse YAML lines
            config = struct();
            i = 1;
            while i <= length(lines)
                line = lines{i};
                
                % Skip empty lines and comments
                if isempty(strtrim(line)) || startsWith(strtrim(line), '#')
                    i = i + 1;
                    continue;
                end
                
                % Get indentation level
                indent = length(line) - length(lstrip(line));
                
                % If indentation is less than base, we're done with this block
                if indent < baseIndent
                    break;
                end
                
                % If indentation is greater, skip (handled by recursion)
                if indent > baseIndent
                    i = i + 1;
                    continue;
                end
                
                % Parse key:value or key: (nested)
                trimmed = strtrim(line);
                
                if contains(trimmed, ':')
                    [key, rest] = strtok(trimmed, ':');
                    key = strtrim(key);
                    value = strtrim(rest(2:end)); % Remove ':'
                    
                    if isempty(value)
                        % Nested structure
                        [nested, nextIdx] = SimpleYAML.parseLines(lines(i+1:end), indent + 2);
                        config.(key) = nested;
                        i = i + nextIdx;
                    else
                        % Simple key:value
                        config.(key) = value;
                        i = i + 1;
                    end
                else
                    i = i + 1;
                end
            end
        end
    end
end

function str = lstrip(s)
    % Remove leading whitespace
    idx = find(~isspace(s), 1);
    if isempty(idx)
        str = '';
    else
        str = s(idx:end);
    end
end
