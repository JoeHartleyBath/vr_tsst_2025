import subprocess
import itertools
from pathlib import Path
import concurrent.futures

# Define parameter grid
burst_criteria = [30, 50, 70, 90]
channel_criteria = [0.7, 0.8, 0.9]
window_criteria = [0.6, 0.7, 0.8]

# Participants to test (diverse subset)
participants = [1, 10, 25, 40]

# Build all parameter combinations
param_grid = list(itertools.product(burst_criteria, channel_criteria, window_criteria))

# Output directory
output_dir = Path('output/asr_sweep_qc')
output_dir.mkdir(parents=True, exist_ok=True)

# Function to run MATLAB quick_asr_check for a given participant and parameter set
def run_asr_check(participant, burst, channel, window):
    set_path = f'output/sets/P{participant:02d}.set'
    log_path = output_dir / f'P{participant:02d}_B{burst}_C{channel}_W{window}.txt'
    matlab_cmd = (
        f"addpath('scripts/preprocessing/eeg/cleaning'); "
        f"try, qc=quick_asr_check('{set_path}',{participant}, 'BurstCriterion',{burst}, 'ChannelCriterion',{channel}, 'WindowCriterion',{window}); "
        f"save('{log_path.with_suffix('.mat')}', 'qc'); "
        f"catch ME, disp(getReport(ME)), end; exit;"
    )
    cmd = [
        'matlab', '-batch', matlab_cmd
    ]
    with open(log_path, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

# Run in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    futures = []
    for participant in participants:
        for burst, channel, window in param_grid:
            futures.append(executor.submit(run_asr_check, participant, burst, channel, window))
    for f in concurrent.futures.as_completed(futures):
        pass  # Optionally, print(f.result())

print('ASR parameter sweep complete. Results saved in output/asr_sweep_qc/')
