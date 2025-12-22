try
    root = fileparts(mfilename('fullpath'));
    % paths
    input_set = fullfile(fileparts(root),'..','..','..','output','cleaned_eeg','P02_cleaned.set');
    out_folder = fullfile(fileparts(root),'..','..','..','output','cleaned_eeg');
    vis_folder = fullfile(fileparts(root),'..','..','..','output','vis','P02');
    qc_folder = fullfile(fileparts(root),'..','..','..','output','qc');

    if ~exist(vis_folder,'dir')
        mkdir(vis_folder);
    end
    if ~exist(qc_folder,'dir')
        mkdir(qc_folder);
    end

    disp(['Running clean_eeg on: ' input_set]);
    addpath(genpath(fileparts(root)));
    [EEG, qc] = clean_eeg(char(input_set), char(out_folder), 2, char(vis_folder), char(qc_folder), []);
    disp('clean_eeg completed successfully for P02.');
    if isfield(EEG,'icaweights')
        disp(['icaweights size: ' num2str(size(EEG.icaweights))]);
    else
        disp('EEG.icaweights missing');
    end
    if isfield(EEG,'icasphere')
        disp(['icasphere size: ' num2str(size(EEG.icasphere))]);
    else
        disp('EEG.icasphere missing');
    end
    if isfield(EEG,'etc') && isfield(EEG.etc,'ic_classification')
        disp('ICLabel classifications present in EEG.etc.ic_classification');
    else
        disp('ICLabel classifications NOT present');
    end
catch ME
    disp(getReport(ME));
    exit(1);
end
exit(0);
