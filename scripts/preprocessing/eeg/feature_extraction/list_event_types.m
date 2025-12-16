% list_event_types.m
% Lists all unique event/condition names in a participant's cleaned .set file

participant = 1; % Change this to the desired participant number
set_dir = fullfile('output', 'cleaned_eeg');
set_file = sprintf('P%02d_cleaned.set', participant);

EEG = pop_loadset('filename', set_file, 'filepath', set_dir);
event_types = {EEG.event.type};
unique_event_types = unique(event_types);

disp(['Event types in ', set_file, ':']);
disp(unique_event_types');
