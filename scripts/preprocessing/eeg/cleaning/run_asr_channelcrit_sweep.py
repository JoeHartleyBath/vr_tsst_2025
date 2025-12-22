import subprocess
from pathlib import Path
import concurrent.futures

# ChannelCriterion values to test (reasonable range for 128ch, 1hr)
channel_criteria = [0.7, 0.75, 0.8, 0.85, 0.9]

# Legacy script parameters
burst_criterion = 20
flatline_criterion = 5
linenoise_criterion = 4
window_criterion = 0.8

participants = [10, 25, 40]

output_dir = Path('output/asr_sweep_qc')
output_dir.mkdir(parents=True, exist_ok=True)

def run_asr_check(participant, channel):
    set_path = f'output/sets/P{participant:02d}.set'
    log_path = output_dir / f'P{participant:02d}_C{channel}.txt'
    matlab_cmd = (
        f"addpath('scripts/preprocessing/eeg/cleaning'); "
        f"try, qc=quick_asr_check('{set_path}',{participant}, "
        f"'BurstCriterion',{burst_criterion}, 'ChannelCriterion',{channel}, "
        f"'FlatlineCriterion',{flatline_criterion}, 'LineNoiseCriterion',{linenoise_criterion}, 'WindowCriterion',{window_criterion}); "
        f"save('{log_path.with_suffix('.mat')}', 'qc'); "
        f"catch ME, disp(getReport(ME)), end; exit;"
    )
    cmd = [
        'matlab', '-batch', matlab_cmd
    ]
    with open(log_path, 'w') as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    futures = []
    for participant in participants:
        for channel in channel_criteria:
            futures.append(executor.submit(run_asr_check, participant, channel))
    for f in concurrent.futures.as_completed(futures):
        pass

print('ASR ChannelCriterion sweep complete for P10, P25, P40.')
