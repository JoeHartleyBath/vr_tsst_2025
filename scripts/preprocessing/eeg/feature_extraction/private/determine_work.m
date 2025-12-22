function participants_to_process = determine_work(participants_requested, temp_folder, force_reprocess)
% DETERMINE_WORK Determine which participants need processing (resume logic)
%
% Inputs:
%   participants_requested - Vector of requested participant numbers
%   temp_folder           - Path to temp folder
%   force_reprocess       - Boolean, if true ignore existing temp files
%
% Outputs:
%   participants_to_process - Vector of participant numbers to process

    fprintf('Determining work...\n');
    fprintf('  Requested: %d participants (%s)\n', ...
            length(participants_requested), mat2str(participants_requested));
    
    if force_reprocess
        fprintf('  Force reprocess enabled - ignoring existing temp files\n');
        participants_to_process = participants_requested;
        return;
    end
    
    % Check for already-processed participants
    processed = [];
    for p = participants_requested
        temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
        if isfile(temp_file)
            processed(end+1) = p; %#ok<AGROW>
        end
    end
    
    if ~isempty(processed)
        fprintf('  Found %d already processed: %s\n', ...
                length(processed), mat2str(processed));
        fprintf('  → Skipping these (delete temp files or use force_reprocess to rerun)\n');
        participants_to_process = setdiff(participants_requested, processed);
    else
        fprintf('  No existing temp files found\n');
        participants_to_process = participants_requested;
    end
    
    fprintf('  To process: %d participants', length(participants_to_process));
    if ~isempty(participants_to_process)
        fprintf(' (%s)', mat2str(participants_to_process));
    end
    fprintf('\n\n');
end
