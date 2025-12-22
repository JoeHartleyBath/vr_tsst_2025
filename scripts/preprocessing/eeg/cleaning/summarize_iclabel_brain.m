% summarize_iclabel_brain.m

% Summarize the number and proportion of ICs labeled as 'Brain' for all participants with ICLabel snapshots
iclabel_dir = 'c:/vr_tsst_2025/output/ica_weights';
files = dir(fullfile(iclabel_dir, '*_iclabel_snapshot.mat'));
if isempty(files)
    error('No ICLabel snapshot files found in %s', iclabel_dir);
end

for f = 1:length(files)
    infile = fullfile(iclabel_dir, files(f).name);
    [~, fname, ~] = fileparts(files(f).name);
    pid = regexprep(fname, '_iclabel_snapshot$', '');
    S = load(infile);
    if isfield(S, 'iclabel_results') && isfield(S.iclabel_results, 'classifications')
        labels = S.iclabel_results.classifications;
        [maxval, maxidx] = max(labels, [], 2);
        brain_idx = (maxidx == 1);
        n_brain = sum(brain_idx);
        n_total = size(labels, 1);
        fprintf('\nICLabel summary for %s:\n', pid);
        fprintf('  Total ICs: %d\n', n_total);
        fprintf('  Brain components: %d (%.1f%%)\n', n_brain, 100*n_brain/n_total);
        % Optionally, print the top 10 brain-labeled ICs with their probabilities
        brain_probs = labels(:,1);
        [sorted_probs, sorted_idx] = sort(brain_probs, 'descend');
        fprintf('  Top 10 ICs with highest Brain probability:\n');
        for i = 1:min(10, n_total)
            fprintf('    IC %d: %.3f\n', sorted_idx(i), sorted_probs(i));
        end
    else
        fprintf('\nICLabel summary for %s: [ERROR: No valid iclabel_results or classifications field]\n', pid);
    end
end
