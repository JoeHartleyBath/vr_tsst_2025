function setup_parallel_pool(config_feat)
% SETUP_PARALLEL_POOL Initialize parallel pool if enabled
%
% Inputs:
%   config_feat - Feature extraction configuration

    if ~config_feat.parallel.enabled
        fprintf('Parallel processing disabled - using single thread\n\n');
        return;
    end
    
    fprintf('Setting up parallel processing...\n');
    
    % Close existing pool
    pool = gcp('nocreate');
    if ~isempty(pool)
        fprintf('  Closing existing pool...\n');
        delete(pool);
    end
    
    % Start new pool
    num_workers = config_feat.parallel.num_workers;
    fprintf('  Starting pool with %d workers...\n', num_workers);
    parpool('local', num_workers);
    
    % Sync paths to workers
    fprintf('  Syncing toolbox paths to workers...\n');
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.eeglab '''));']);
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.entropy_hub '''));']);
    pctRunOnAll(['addpath(genpath(''' config_feat.toolbox_paths.utils '''));']);
    pctRunOnAll eeglab nogui;
    
    fprintf('  ✓ Parallel pool ready\n\n');
end
