% Wrapper script to run the pipeline
cd(fileparts(mfilename('fullpath')));
addpath(pwd());
run_clean_eeg_pipeline;
