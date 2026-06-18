function fix_p26_chanlocs()
    % FIX_P26_CHANLOCS - Add channel locations to raw P26.set file
    
    projectRoot = 'C:\vr_tsst_2025';
    
    % Load raw P26.set
    raw_set_path = fullfile(projectRoot, 'output', 'sets', 'P26.set');
    fprintf('Loading %s...\n', raw_set_path);
    
    EEG = pop_loadset('filename', 'P26.set', 'filepath', fullfile(projectRoot, 'output', 'sets'));
    
    fprintf('Loaded: %d channels, %d samples, %.1f Hz\n', ...
        EEG.nbchan, EEG.pnts, EEG.srate);
    
    % Assign ANT Neuro 128-channel equidistant layout
    chanlocs_file = fullfile(projectRoot, 'config', 'chanlocs', 'NA-271.elc');
    fprintf('Assigning channel locations from %s...\n', chanlocs_file);
    
    EEG = pop_chanedit(EEG, 'lookup', chanlocs_file);
    
    if isempty(EEG.chanlocs) || isempty(EEG.chanlocs(1).X)
        error('Failed to assign channel locations.');
    end
    
    fprintf('Channel locations assigned successfully.\n');
    fprintf('First 3 channels: %s, %s, %s\n', ...
        EEG.chanlocs(1).labels, EEG.chanlocs(2).labels, EEG.chanlocs(3).labels);
    
    % Save back to same location
    output_path = fullfile(projectRoot, 'output', 'sets', 'P26.set');
    fprintf('Saving fixed file to %s...\n', output_path);
    
    pop_saveset(EEG, 'filename', 'P26.set', 'filepath', fullfile(projectRoot, 'output', 'sets'));
    
    fprintf('Done! Raw P26.set now has channel locations.\n');
end
