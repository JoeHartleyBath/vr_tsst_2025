try
    % Robustly resolve project root and .set path
    % Directly use absolute path for robust test
    setpath = 'C:\vr_tsst_2025\output\cleaned_eeg\P02_cleaned.set';
    EEG = pop_loadset('filename', setpath);
    n = numel(EEG.chanlocs);
    hasXYZ = arrayfun(@(c) ~isempty(c.X) && ~isempty(c.Y) && ~isempty(c.Z) && all(~isnan([c.X c.Y c.Z])), EEG.chanlocs);
    hasTheta = arrayfun(@(c) isfield(c,'theta') && ~isempty(c.theta) && ~isnan(c.theta), EEG.chanlocs);
    if all(hasXYZ) && all(hasTheta) && n == EEG.nbchan
        fprintf('PASS: All %d channels have valid X/Y/Z and spherical coordinates. ASR will use location-aware cleaning.\n', n);
        try
            [EEG2, com] = clean_artifacts(EEG, 'FlatlineCriterion',5, 'ChannelCriterion',0.7, 'LineNoiseCriterion',4, 'BurstCriterion',50, 'WindowCriterion',0.6);
            fprintf('clean_artifacts ran successfully.\n');
            if isfield(EEG2, 'etc') && isfield(EEG2.etc, 'clean_channel_mask')
                fprintf('Channels retained after ASR: %d/%d\n', sum(EEG2.etc.clean_channel_mask), n);
            end
        catch ME2
            fprintf('ERROR: clean_artifacts failed: %s\n', ME2.message);
        end
    else
        fprintf('FAIL: Only %d/%d channels have valid X/Y/Z, %d/%d have theta.\n', sum(hasXYZ), n, sum(hasTheta), n);
        missing = find(~hasXYZ | ~hasTheta);
        if ~isempty(missing)
            disp('Sample missing:');
            disp({EEG.chanlocs(missing(1:min(10,end))).labels}');
        end
    end
catch ME
    disp(getReport(ME));
    fprintf('ERROR: Could not check channel locations.\n');
end
