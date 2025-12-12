function cleanup_temp_files(participants_requested, temp_folder, keep_temp)
% CLEANUP_TEMP_FILES Delete temp files and folder after successful merge
%
% Inputs:
%   participants_requested - Vector of participant numbers
%   temp_folder           - Path to temp folder
%   keep_temp             - Boolean, if true keep temp files for debugging

    if keep_temp
        fprintf('Keeping temp files for debugging: %s\n', temp_folder);
        return;
    end
    
    fprintf('Cleaning up temp files...\n');
    
    deleted = 0;
    for p = participants_requested
        temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
        if isfile(temp_file)
            delete(temp_file);
            deleted = deleted + 1;
        end
    end
    
    % Remove temp folder if empty
    if isempty(dir(fullfile(temp_folder, '*.csv')))
        rmdir(temp_folder);
        fprintf('  ✓ Deleted %d temp files and removed temp folder\n', deleted);
    else
        fprintf('  ✓ Deleted %d temp files (folder contains other files)\n', deleted);
    end
end
