#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import logging
import pyxdf
import mne
import yaml


# Setup logging
logging.basicConfig(level=logging.DEBUG, filename='data_processing_debug.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')



def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# In[2]:


def Index_of_Closest(array, val):
    return np.argmin(np.abs(array - val))

def Do_Import(f_path, A="000496", B="000495", ch_count=64, include_other_streams=False, drop_time=0.0):
    data, header = pyxdf.load_xdf(f_path)

    eeg_streams = [None, None]
    other_streams = []

    for stream in data:
        s_type = stream['info']['type'][0]
        s_name = stream['info']['name'][0]
        if s_type == "EEG":
            if A in s_name:
                eeg_streams[0] = stream
            elif B in s_name:
                eeg_streams[1] = stream
            else:
                raise IndexError("EEG Stream should either be A or B")
        else:
            other_streams.append(stream)

    s_rate = float(eeg_streams[0]['info']['nominal_srate'][0])
    eeg_ts = eeg_streams[0]["time_stamps"].T
    eeg_data = eeg_streams[0]["time_series"][:, :ch_count].T

    if eeg_streams[1] is not None:
        b_ts = eeg_streams[1]["time_stamps"].T
        b_data = eeg_streams[1]["time_series"][:, :ch_count].T

        if b_ts.size > eeg_ts.size:
            gap = b_ts.size - eeg_ts.size
            idx = min(gap, Index_of_Closest(b_ts, eeg_ts[0]))
            b_ts = b_ts[idx:eeg_ts.size + idx]
            b_data = b_data[:, idx:eeg_ts.size + idx]
        else:
            gap = eeg_ts.size - b_ts.size
            idx = min(gap, Index_of_Closest(eeg_ts, b_ts[0]))
            eeg_ts = eeg_ts[idx:b_ts.size + idx]
            eeg_data = eeg_data[:, idx:b_ts.size + idx]

        eeg_data = np.concatenate((eeg_data, b_data), axis=0)

    if drop_time > 0.0:
        drop_samples = int(drop_time * s_rate)
        eeg_data = eeg_data[:, drop_samples:]
        eeg_ts = eeg_ts[drop_samples:]

    return s_rate, eeg_ts, eeg_data


# In[3]:


channel_names = [
    'Z2', 'Z4', 'Z6', 'Z8', 'Z10', 'Z12', 
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13', 'L14', 
    'LL1', 'LL2', 'LL3', 'LL4', 'LL5', 'LL6', 'LL7', 'LL8', 'LL9', 'LL10', 'LL11', 'LL12', 
    'Z1', 'Z3', 'Z5', 'Z9', 'Z11', 'Z13', 
    'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 
    'RR1', 'RR2', 'RR3', 'RR4', 'RR5', 'RR6', 'RR7', 'RR8', 'RR9', 'RR10', 'RR11', 'RR12', 
    'LL13', 'LL14', 
    'LA1', 'LA2', 'LA3', 'LA4', 'LA5', 
    'LB1', 'LB2', 'LB3', 'LB4', 'LB5', 'LB6', 
    'LC1', 'LC2', 'LC3', 'LC4', 'LC5', 'LC6', 'LC7', 
    'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'LD7', 
    'LE1', 'LE2', 'LE3', 'LE4', 
    'Lm', 
    'RR13', 'RR14', 
    'RA1', 'RA2', 'RA3', 'RA4', 'RA5', 
    'RB1', 'RB2', 'RB3', 'RB4', 'RB5', 'RB6', 
    'RC1', 'RC2', 'RC3', 'RC4', 'RC5', 'RC6', 'RC7', 
    'RD1', 'RD2', 'RD3', 'RD4', 'RD5', 'RD6', 'RD7', 
    'RE1', 'RE2', 'RE3', 'RE4', 
    'RM'
]

def load_data(pId, config):
    xdf_path = os.path.join(config['paths']['raw_data'], f"{pId}_EEG.xdf")
    csv_path = os.path.join(config['paths']['raw_data'], f"{pId}_RAW.csv")

    # Load EEG and LSL streams
    s_rate, eeg_ts, eeg_data = Do_Import(
        f_path=xdf_path,
        B="495",
        A="496",
        include_other_streams=True
    )

    # Create MNE Raw object
    ch_names = channel_names  # defined elsewhere or load separately
    info = mne.create_info(ch_names=ch_names, sfreq=s_rate, ch_types=['eeg'] * len(ch_names))
    raw = mne.io.RawArray(eeg_data, info)

    # Load physiological CSV
    df = pd.read_csv(csv_path, low_memory=False)


    return raw, df, eeg_ts, eeg_data, s_rate, ch_names


# In[4]:


def process_and_export_eeg(eeg_ts, eeg_data, s_rate, ch_names, raw, df_physio, pId, config):
    # Convert physio timestamps to datetime and set as index
    df_physio['LSL_Timestamp'] = pd.to_datetime(df_physio['LSL_Timestamp'], unit='s', origin='unix')
    df_physio.set_index('LSL_Timestamp', inplace=True)

    # Create datetime index for EEG
    first_eeg_timestamp = pd.to_datetime(eeg_ts[0], unit='s', origin='unix')
    eeg_time_deltas = pd.to_timedelta(np.arange(len(eeg_data[0])) / s_rate, unit='s')
    eeg_timestamps = first_eeg_timestamp + eeg_time_deltas

    # Align EEG to physio by computing and subtracting the gap
    gap = first_eeg_timestamp - df_physio.index[0]
    adjusted_eeg_timestamps = eeg_timestamps - gap

    # Build EEG dataframe (not essential but used for merge_asof)
    df_eeg = pd.DataFrame(eeg_data.T, columns=ch_names, index=adjusted_eeg_timestamps)
    df_eeg.sort_index(inplace=True)

    # Optional: Resample physio
    numeric_columns = df_physio.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df_physio.select_dtypes(exclude=[np.number]).columns

    df_numeric = df_physio[numeric_columns].resample('8ms').mean().interpolate(method='linear', limit_direction='both')
    df_non_numeric = df_physio[non_numeric_columns].resample('8ms').ffill()

    df_resampled = pd.concat([df_numeric, df_non_numeric], axis=1)

    # Optional: Merge EEG and physio for downstream use
    tolerance = pd.Timedelta(milliseconds=20)
    df_combined_resampled = pd.merge_asof(
        df_eeg,
        df_resampled,
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=tolerance
    )

    # Resample EEG Raw (MNE auto-filters)
    raw = raw.resample(125, npad="auto")

    # Export
    export_path = os.path.join(config['paths']['eeg_data'], f"P{pId:02d}_raw.set")
    try:
        raw.export(export_path, fmt='eeglab', overwrite=True)
        print(f"✅ EEG .set file saved: {export_path}")
    except Exception as e:
        print(f"❌ Failed to export EEG .set file for P{pId}: {e}")

    return adjusted_eeg_timestamps





# In[5]:


def assign_conditions_and_generate_events(df_physio, raw, adjusted_eeg_timestamps):
    print("Starting condition assignment and event generation.")

    # Print the data types of the inputs
    print(f"Data type of df_physio: {type(df_physio)}")
    print(f"Data type of raw: {type(raw)}")
    print(f"Data type of adjusted_eeg_timestamps: {type(adjusted_eeg_timestamps)}")

    # Print content of adjusted_eeg_timestamps
    print(f"Contents of adjusted_eeg_timestamps: {adjusted_eeg_timestamps}")

    # Define conditions and corresponding labels
    conditions = [
        #Calibrations
        (df_physio['Unity Scene'] == 'PrimaryCalibration'),
        (df_physio['Unity Scene'] == 'BlinkCalibration'),
        (df_physio['Unity Scene'] == 'BaseLine'),
        #Fixation cross scenes
        ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'Start Blank Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'Start Room Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'End Room Fixation')),
         ((df_physio['Unity Scene'] == 'Fixation') &
         (df_physio['Study_Phase'] == 'End Blank Fixation')),
        #Preambles
    #Stress 1022 preamble
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Stress 2043 preamble
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # Preamble for Stress Addition
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Calm 1022 preamble
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    #Calm 2043 preamble
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # Preamble for Calm Addition
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'].isin(['Instructions', 'Begin', 'TaskAction']))
    ),
    # TaskTime for high cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    # TaskTime for low cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    # Same for CalmRoom
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'TaskTime')
    ),
# Thanks Action for high cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    # Thanks Action for low cognitive effort
    (
        (df_physio['Shown_Scene'] == 'StressRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    # Same for CalmRoom
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction1022') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Subtraction2043') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),
    (
        (df_physio['Shown_Scene'] == 'CalmRoom') & 
        (df_physio['Arithmetic_Task'] == 'Addition') & 
        (df_physio['Participant_State'] == 'ThanksAction')
    ),

        (df_physio['Shown_Scene'] == 'Forest1'),
        (df_physio['Shown_Scene'] == 'Forest2'),
        (df_physio['Shown_Scene'] == 'Forest3'),
        (df_physio['Shown_Scene'] == 'Forest4'),
    ]
    choices = [
    'Primary_Calibrations',
    'Blink_Calibration',
    'Movement_Baseline',
    'Pre_Exposure_Blank_Fixation_Cross',
    'Pre_Exposure_Room_Fixation_Cross',
    'Post_Exposure_Blank_Fixation_Cross',
    'Post_Exposure_Room_Fixation_Cross',
    'HighStress_HighCog1022_Preamble',
    'HighStress_HighCog2043_Preamble',
    'HighStress_LowCog_Preamble',
    'LowStress_HighCog1022_Preamble',
    'LowStress_HighCog2043_Preamble',
    'LowStress_LowCog_Preamble',
    'HighStress_HighCog1022_Task',
    'HighStress_HighCog2043_Task',
    'HighStress_LowCog_Task',
    'LowStress_HighCog1022_Task',
    'LowStress_HighCog2043_Task',
    'LowStress_LowCog_Task',
    'HighStress_HighCog1022_Finish',
    'HighStress_HighCog2043_Finish',
    'HighStress_LowCog_Finish',
    'LowStress_HighCog1022_Finish',
    'LowStress_HighCog2043_Finish',
    'LowStress_LowCog_Finish',
    'Forest1',
    'Forest2',
    'Forest3',
    'Forest4',
];

    # Create the new column using numpy.select to apply conditions and choices
    df_physio['exposure_type'] = np.select(conditions, choices, default='no exposure')


    conditions_of_interest = [
                            'Primary_Calibrations',
                            'Blink_Calibration',
                            'Movement_Baseline',
                            'Pre_Exposure_Blank_Fixation_Cross',
                            'Pre_Exposure_Room_Fixation_Cross',
                            'Post_Exposure_Blank_Fixation_Cross',
                            'Post_Exposure_Room_Fixation_Cross',
                            'HighStress_HighCog1022_Preamble',
                            'HighStress_HighCog2043_Preamble',
                            'HighStress_LowCog_Preamble',
                            'LowStress_HighCog1022_Preamble',
                            'LowStress_HighCog2043_Preamble',
                            'LowStress_LowCog_Preamble',
                            'HighStress_HighCog1022_Task',
                            'HighStress_HighCog2043_Task',
                            'HighStress_LowCog_Task',
                            'LowStress_HighCog1022_Task',
                            'LowStress_HighCog2043_Task',
                            'LowStress_LowCog_Task',
                            'HighStress_HighCog1022_Finish',
                            'HighStress_HighCog2043_Finish',
                            'HighStress_LowCog_Finish',
                            'LowStress_HighCog1022_Finish',
                            'LowStress_HighCog2043_Finish',
                            'LowStress_LowCog_Finish',
                            'Forest1',
                            'Forest2',
                            'Forest3',
                            'Forest4',
                            ]
    # Create a dictionary to store the start and end sample index for each condition
    condition_sample_index = {condition: None for condition in conditions_of_interest}

    for condition in conditions_of_interest:
        condition_sample_index[condition] = []  # Initialize as an empty list
        condition_indices = df_physio.index[df_physio['exposure_type'] == condition]

        if not condition_indices.empty:
            print(f"Condition indices for {condition}: {condition_indices}")
            try:
                # Convert the first and last timestamp index to EEG data indices
                # Ensure subtraction is between compatible types, i.e., DatetimeIndex entries
                start_timestamp = condition_indices[0]
                end_timestamp = condition_indices[-1]

                start_idx = int((start_timestamp - adjusted_eeg_timestamps[0]).total_seconds() * raw.info['sfreq'])
                end_idx = int((end_timestamp - adjusted_eeg_timestamps[0]).total_seconds() * raw.info['sfreq'])

                print(f"Start index for {condition}: {start_idx}, End index: {end_idx}")
                condition_sample_index[condition].append((start_idx, end_idx))
            except Exception as e:
                print(f"Error processing indices for {condition}: {e}")


    # Map 'exposure_type' to unique integers for event markers
    unique_exposures = df_physio['exposure_type'].unique()
    event_id_dict = {etype: i + 1 for i, etype in enumerate(unique_exposures)}

    # Log the creation of event identifiers
    unique_exposures = df_physio['exposure_type'].unique()
    event_id_dict = {etype: i + 1 for i, etype in enumerate(unique_exposures)}
    logging.debug(f"Event ID dictionary created with items: {event_id_dict}")

    # Data preparation
    all_events = []
    sfreq = raw.info['sfreq']  # Sampling frequency
    logging.debug(f"Sampling frequency: {sfreq}")

    for condition in conditions_of_interest:
        for start_index, end_index in condition_sample_index[condition]:
            for sec in range(start_index, end_index + 1, int(sfreq)):  # Increment by sample rate to get one-second epochs
                all_events.append([sec, condition])  # sec is the sample index for each second


    # Create DataFrame
    events_df = pd.DataFrame(all_events, columns=['latency', 'type'])

    return df_physio, events_df




# In[ ]:


# -------- MAIN PIPELINE --------
def run_pipeline(pId, config):

    paths = config["paths"]
    raw_data_folder = paths["raw_data"].format(pid=pId)
    metadata_path   = paths["metadata"].format(pid=pId)
    events_path     = paths["events"].format(pid=pId)
    # Step 1: Load raw EEG and physiology data
    raw, df_physio, eeg_ts, eeg_data, s_rate, ch_names = load_data(pId, config)

    # Step 2: Process and export EEG .set file
    adjusted_eeg_timestamps = process_and_export_eeg(eeg_ts, eeg_data, s_rate, ch_names, raw, df_physio, pId, config)

    # Step 3: Generate events CSV
    df_physio, events_df = assign_conditions_and_generate_events(df_physio, raw, adjusted_eeg_timestamps)

    # Save the events file
    events_output = config["paths"]["events"]
    events_df.to_csv(os.path.join(events_output, f"P{pId}_events.csv"), index=False)

    print(f"✅ All done for participant {pId}")


# -------- RUN --------
if __name__ == "__main__":
    config = load_config()
    participant_id = 40  
    run_pipeline(participant_id, config)

