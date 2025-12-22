% MATLAB startup.m - Run automatically on MATLAB launch
% Robustly add EEGLAB, AMICA, and related toolboxes to MATLAB path.
% Uses matlabroot, prefdir, and userpath to find toolboxes in standard locations.
% Falls back to project-local folders if system toolboxes not found.
% This version works reliably in both interactive and batch modes.

fprintf('[startup] Initializing VR-TSST pipeline...\n');

%% Add project utils to path (for SimpleYAML and other utilities)
projUtilsPath = fullfile(pwd, 'scripts', 'utils');
if exist(projUtilsPath, 'dir')
    addpath(projUtilsPath);
    fprintf('[startup] Project utils loaded: %s\n', projUtilsPath);
end

%% Helper: Add toolbox with fallback locations
    function addToolbox(toolboxName, commonDirs)
        % Try to add toolbox from common MATLAB locations or project local paths
        % commonDirs: cell array of candidate directories (absolute or relative to pwd)
        
        added = false;
        
        % Try system MATLAB toolbox locations
        matlabRoot = matlabroot();
        systemToolboxPath = fullfile(matlabRoot, 'toolbox', toolboxName);
        if exist(systemToolboxPath, 'dir')
            addpath(genpath(systemToolboxPath));
            fprintf('[startup] %s loaded (system: %s)\n', toolboxName, systemToolboxPath);
            added = true;
            return;
        end
        
        % Try versioned system toolbox folders (e.g., eeglab2025.1.0)
        toolboxDir = 'c:\MATLAB\toolboxes';
        if exist(toolboxDir, 'dir')
            d = dir(fullfile(toolboxDir, [toolboxName '*']));
            for i = 1:numel(d)
                if d(i).isdir
                    versionedPath = fullfile(toolboxDir, d(i).name);
                    addpath(genpath(versionedPath));
                    fprintf('[startup] %s loaded (versioned system: %s)\n', toolboxName, versionedPath);
                    added = true;
                    return;
                end
            end
        end
        
        % Try hardcoded system-level paths (for non-standard installs)
        hardcodedPaths = {
            fullfile('c:', 'MATLAB', 'toolboxes', toolboxName),
            fullfile('c:', 'Program Files', 'MATLAB', 'toolboxes', toolboxName),
            fullfile('C:', 'Users', getenv('USERNAME'), 'MATLAB', 'toolboxes', toolboxName)
        };
        for i = 1:numel(hardcodedPaths)
            if exist(hardcodedPaths{i}, 'dir')
                addpath(genpath(hardcodedPaths{i}));
                fprintf('[startup] %s loaded (hardcoded: %s)\n', toolboxName, hardcodedPaths{i});
                added = true;
                return;
            end
        end
        
        % Try project-local candidates
        if nargin > 1
            for i = 1:numel(commonDirs)
                candidate = commonDirs{i};
                % Make relative paths absolute (isabs not in older MATLAB; check for ':')
                if ~(length(candidate) > 1 && candidate(2) == ':')  % Windows drive check
                    candidate = fullfile(pwd, candidate);
                end
                if exist(candidate, 'dir')
                    addpath(genpath(candidate));
                    fprintf('[startup] %s loaded (project: %s)\n', toolboxName, candidate);
                    added = true;
                    return;
                end
            end
        end
        
        if ~added
            fprintf('[startup] WARNING: %s not found in standard locations; may not be available.\n', toolboxName);
        end
    end

%% Add EEGLAB
addToolbox('eeglab', {'a/eeglab', 'scripts/lib/eeglab', 'scripts/utils/eeglab'});

%% Add AMICA
addToolbox('amica', {'a/amica', 'scripts/lib/amica', 'scripts/utils/amica'});

%% Add yamlmatlab or provide SimpleYAML fallback
addToolbox('yamlmatlab', {'scripts/utils/yamlmatlab', 'a/yamlmatlab'});

% If ReadYaml not available, ensure SimpleYAML is on path
if exist('ReadYaml', 'file') ~= 2 && exist('SimpleYAML', 'file') == 2
    fprintf('[startup] Using SimpleYAML as YAML fallback.\n');
end

%% Verify critical functions
fprintf('[startup] Verifying critical functions...\n');
criticalFunctions = {'eeglab', 'pop_loadset', 'pop_eegfiltnew'};
for i = 1:numel(criticalFunctions)
    func = criticalFunctions{i};
    if exist(func, 'file') == 2
        fprintf('[startup]   ✓ %s found\n', func);
    else
        fprintf('[startup]   ✗ %s NOT FOUND (may cause issues)\n', func);
    end
end

%% Save paths (optional; may fail in some batch/restricted environments)
try
    savepath;
    fprintf('[startup] Paths saved.\n');
catch ME
    fprintf('[startup] Note: savepath failed (expected in batch mode): %s\n', ME.message);
end

fprintf('[startup] VR-TSST pipeline initialization complete.\n');
