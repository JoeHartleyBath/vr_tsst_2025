import subprocess
from pathlib import Path
import concurrent.futures

# Narrowed parameter grid
burst_criteria = [30, 50]
channel_criteria = [0.7, 0.8]
window_criteria = [0.6, 0.8]

participants = [10, 25]

output_dir = Path('output/asr_sweep_qc')
output_dir.mkdir(parents=True, exist_ok=True)

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

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = []
    for participant in participants:
        for burst in burst_criteria:
            for channel in channel_criteria:
                for window in window_criteria:
                    futures.append(executor.submit(run_asr_check, participant, burst, channel, window))
    for f in concurrent.futures.as_completed(futures):
        pass

print('Narrowed ASR parameter sweep complete for P10 and P25.')
