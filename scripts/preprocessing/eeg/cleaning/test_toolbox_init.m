% TEST_TOOLBOX_INIT - Verify all required functions are available
% Run this after startup.m to ensure toolbox initialization succeeded
% Usage: matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/test_toolbox_init.m')"

fprintf('\n=============================================================\n');
fprintf('TOOLBOX INITIALIZATION TEST\n');
fprintf('=============================================================\n\n');

%% Required functions for EEG cleaning pipeline
requiredFunctions = {
    'eeglab',           % EEGLAB init
    'pop_loadset',      % Load .set files
    'pop_saveset',      % Save .set files
    'pop_eegfiltnew',   % IIR filtering
    'clean_artifacts',  % ASR cleaning
    'runamica15',       % AMICA ICA
    'iclabel',          % IC Label classification
    'pop_subcomp',      % Remove components
    'pop_interp',       % Interpolate channels
    'pop_reref',        % Re-reference
    'pop_eegplot'       % Plot EEG
};

%% YAML functions (one should be available)
yamlFunctions = {
    'ReadYaml',         % yamlmatlab
    'SimpleYAML'        % Project-local fallback
};

fprintf('Required EEG functions:\n');
eeglab_ok = false;
missing_eeg = {};
for i = 1:numel(requiredFunctions)
    func = requiredFunctions{i};
    if exist(func, 'file') == 2
        fprintf('  ✓ %s\n', func);
        if strcmp(func, 'eeglab')
            eeglab_ok = true;
        end
    else
        fprintf('  ✗ %s NOT FOUND\n', func);
        missing_eeg{end+1} = func; %#ok<AGROW>
    end
end

fprintf('\nYAML reader (at least one required):\n');
yaml_ok = false;
for i = 1:numel(yamlFunctions)
    func = yamlFunctions{i};
    if exist(func, 'file') == 2
        fprintf('  ✓ %s\n', func);
        yaml_ok = true;
    else
        fprintf('  ○ %s not available\n', func);
    end
end

fprintf('\n=============================================================\n');
if eeglab_ok && yaml_ok && isempty(missing_eeg)
    fprintf('RESULT: ✓ ALL TESTS PASSED\n');
    fprintf('Pipeline is ready to run.\n');
    exitcode = 0;
else
    fprintf('RESULT: ✗ TESTS FAILED\n');
    if ~eeglab_ok
        fprintf('  - EEGLAB not initialized; check startup.m\n');
    end
    if ~yaml_ok
        fprintf('  - No YAML reader available; install yamlmatlab or verify SimpleYAML\n');
    end
    if ~isempty(missing_eeg)
        fprintf('  - Missing EEG functions: %s\n', strjoin(missing_eeg, ', '));
    end
    exitcode = 1;
end
fprintf('=============================================================\n\n');

exit(exitcode);
