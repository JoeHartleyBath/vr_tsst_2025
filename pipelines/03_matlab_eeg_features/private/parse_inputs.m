function params = parse_inputs(varargin)
% PARSE_INPUTS Parse flexible CLI inputs for feature extraction
%
% Usage:
%   params = parse_inputs()                                    % Use defaults
%   params = parse_inputs('participants', [1 5 10])           % Specific participants
%   params = parse_inputs('force_reprocess', true)            % Force reprocess
%   params = parse_inputs('output_folder', 'custom/path')     % Custom output
%   params = parse_inputs('config', 'custom_config.yaml')     % Custom config
%   params = parse_inputs('participants', 1:10, 'parallel', false) % Multiple args
%
% Returns:
%   params - struct with fields:
%     .participants     : Vector of participant numbers (default: 1:48)
%     .force_reprocess  : Logical, skip resume logic (default: false)
%     .output_folder    : Custom output folder (default: from config)
%     .config_file      : Custom config file (default: config/eeg_feature_extraction.yaml)
%     .parallel_enabled : Override parallel setting (default: from config)
%     .num_workers      : Override number of workers (default: from config)

    % Create input parser
    p = inputParser;
    p.CaseSensitive = false;
    p.KeepUnmatched = false;
    
    % Define parameters with validation
    addParameter(p, 'participants', 1:48, ...
        @(x) isnumeric(x) && all(x > 0) && all(mod(x,1) == 0));
    addParameter(p, 'force_reprocess', false, @islogical);
    addParameter(p, 'output_folder', '', @ischar);
    addParameter(p, 'config_file', 'config/eeg_feature_extraction.yaml', @ischar);
    addParameter(p, 'conditions_file', 'config/conditions.yaml', @ischar);
    addParameter(p, 'general_file', 'config/general.yaml', @ischar);
    addParameter(p, 'parallel', [], @(x) isempty(x) || islogical(x));
    addParameter(p, 'num_workers', [], ...
        @(x) isempty(x) || (isnumeric(x) && x > 0 && mod(x,1) == 0));
    
    % Parse inputs
    parse(p, varargin{:});
    
    % Extract results
    params = struct();
    params.participants = p.Results.participants;
    params.force_reprocess = p.Results.force_reprocess;
    params.output_folder = p.Results.output_folder;
    params.config_file = p.Results.config_file;
    params.parallel_enabled = p.Results.parallel;
    params.num_workers = p.Results.num_workers;
    params.conditions_file = p.Results.conditions_file;
    params.general_file = p.Results.general_file;
    
    % Validate participants
    if any(params.participants < 1) || any(params.participants > 100)
        warning('Participant numbers should typically be between 1-48 for this dataset.');
    end
    
    % Validate config file exists
    if ~isfile(params.config_file)
        error('Config file not found: %s', params.config_file);
    end
    % Validate conditions file exists
    if ~isfile(params.conditions_file)
        error('Conditions file not found: %s', params.conditions_file);
    end
    % Validate general file exists
    if ~isfile(params.general_file)
        error('General config file not found: %s', params.general_file);
    end
    
    fprintf('Input parameters:\n');
    fprintf('  Participants: %s\n', mat2str(params.participants));
    fprintf('  Force reprocess: %s\n', string(params.force_reprocess));
    if ~isempty(params.output_folder)
        fprintf('  Output folder: %s\n', params.output_folder);
    end
    if ~isempty(params.parallel_enabled)
        fprintf('  Parallel: %s\n', string(params.parallel_enabled));
    end
    if ~isempty(params.num_workers)
        fprintf('  Workers: %d\n', params.num_workers);
    end
end
