import pandas as pd
import numpy as np
import os

def calculate_demographics():
    # Path to QuestionPro data
    data_path = r"data/Pre_Study_Questionnaire.xlsx"
    
    print(f"Reading demographics from: {data_path}")
    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # Clean column names (strip whitespace)
    df.columns = df.columns.str.strip()
    
    print(f"Columns found: {df.columns.tolist()}")

    # Define exclusion list (QC failures + P14)
    # P02, P08, P46 from QC summary, plus P14
    excluded_ids = [2, 8, 14, 46]
    print(f"Excluding participants: {excluded_ids}")

    # Filter participants
    # Ensure Participant_ID is numeric
    df['Participant_ID'] = pd.to_numeric(df['Participant_ID'], errors='coerce')
    
    # Drop rows with invalid IDs
    df = df.dropna(subset=['Participant_ID'])
    
    # Apply exclusion
    subset = df[~df['Participant_ID'].isin(excluded_ids)].copy()
    
    print(f"Original N: {len(df)}")
    print(f"Final N (after exclusion): {len(subset)}")
    print(f"Participants retained: {sorted(subset['Participant_ID'].unique().astype(int))}")

    # Convert Age to numeric
    subset["Age"] = pd.to_numeric(subset["Age"], errors='coerce')

    # Mappings
    gender_map = {1: "Male", 2: "Female", 5: "Prefer not to say"}
    vr_map = {1: "Daily", 2: "Weekly", 3: "Occasionally", 4: "Never"}

    # Calculate Stats
    n_total = len(subset)
    age_mean = subset["Age"].mean()
    age_sd = subset["Age"].std()
    
    # Get counts with mapped labels
    gender_counts = subset["Gender"].map(gender_map).value_counts()
    vr_exp_counts = subset["VR_Experience"].map(vr_map).value_counts()

    # Output results
    output_file = r"results/demographics_summary.txt"
    csv_output_file = r"results/demographics.csv"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save clean CSV for merging
    subset_export = subset[["Participant_ID", "Age", "Gender", "VR_Experience"]].copy()
    subset_export.columns = ["participant_id", "age", "gender", "vr_experience"]
    
    subset_export.to_csv(csv_output_file, index=False)
    print(f"Demographics CSV written to {csv_output_file}")

    with open(output_file, "w") as f:
        f.write("--- Demographics Summary ---\n")
        f.write(f"Total Participants (N): {n_total}\n")
        f.write(f"Excluded: {excluded_ids}\n\n")
        
        f.write("--- Age ---\n")
        f.write(f"Mean: {age_mean:.2f}\n")
        f.write(f"SD: {age_sd:.2f}\n\n")
        
        f.write("--- Gender ---\n")
        f.write(gender_counts.to_string())
        f.write("\n\n")
        
        f.write("--- VR Experience ---\n")
        f.write(vr_exp_counts.to_string())
        f.write("\n")

    print(f"Demographics summary written to {output_file}")
    print("-" * 20)
    print(open(output_file).read())
    print("-" * 20)

if __name__ == "__main__":
    calculate_demographics()
