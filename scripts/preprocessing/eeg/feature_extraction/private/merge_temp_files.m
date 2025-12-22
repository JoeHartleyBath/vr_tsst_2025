function merged_count = merge_temp_files(participants_requested, temp_folder, output_csv)
% MERGE_TEMP_FILES Merge individual participant temp files into final CSV
%
% Inputs:
%   participants_requested - Vector of participant numbers (in processing order)
%   temp_folder           - Path to temp folder
%   output_csv            - Path to output CSV file
%
% Outputs:
%   merged_count - Number of participants successfully merged

    fprintf('=== MERGING RESULTS ===\n');
    
    % Open output file for appending (header already written)
    fid_out = fopen(output_csv, 'a');
    if fid_out == -1
        error('Could not open output file for writing: %s', output_csv);
    end
    
    merged_count = 0;
    missing = [];
    
    % Merge in order requested
    for p = participants_requested
        temp_file = fullfile(temp_folder, sprintf('P%02d_features.csv', p));
        
        if isfile(temp_file)
            % Read and append
            temp_data = fileread(temp_file);
            if ~isempty(temp_data)
                fprintf(fid_out, '%s', temp_data);
                merged_count = merged_count + 1;
            end
        else
            missing(end+1) = p; %#ok<AGROW>
        end
    end
    
    fclose(fid_out);
    
    % Report results
    fprintf('  ✓ Merged %d/%d participants\n', merged_count, length(participants_requested));
    if ~isempty(missing)
        fprintf('  ⚠ Missing: %s\n', mat2str(missing));
    end
    fprintf('  Output: %s\n\n', output_csv);
end
