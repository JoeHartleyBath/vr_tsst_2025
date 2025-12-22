# Baseline Matching Fix - Root Cause Analysis

## Diagnosis Summary

Running `diagnose_baseline_matching.py` revealed THREE critical issues blocking baseline adjustment:

### Issue 1: Wrong Condition Names in CLASSIFICATION_TASKS ❌

**Current (WRONG):**
```python
CLASSIFICATION_TASKS = {
    'HighStress': ['HighStress1022_Speech', 'HighStress1022_Math', ...],
    'LowStress': ['LowStress1022_Speech', 'LowStress1022_Math', ...],
    ...
}
```

**Actual condition names in data:**
- `HighStress_HighCog1022_Task`
- `HighStress_LowCog_Task`
- `LowStress_HighCog1022_Task`
- `LowStress_LowCog_Task`
- `Forest1`, `Forest2`, `Forest3`, `Forest4`

**Impact:** Script never found any task data because names didn't match!

---

### Issue 2: Counterbalance Sheet Incompatible with get_baseline() ❌

**Current code assumes:**
- Columns like `HighStress_round`, `LowStress_round`, etc.
- Direct task-to-round mapping

**Actual counterbalance structure:**
```
Participant | Round 1           | Round 2        | Round 3           | Round 4
1          | Stress Subtraction | Stress Addition| Calm Subtraction  | Calm Addition
```

**Impact:** get_baseline() couldn't find the round columns it expected!

---

### Issue 3: Task Name Mapping Missing ❌

**Need to map round descriptions to condition names:**
- "Stress Subtraction" → `HighStress_HighCog1022_Task`
- "Stress Addition" → `HighStress_LowCog_Task`
- "Calm Subtraction" → `LowStress_HighCog1022_Task`
- "Calm Addition" → `LowStress_LowCog_Task`

---

## The Fix

### 1. Update CLASSIFICATION_TASKS Dictionary

Replace lines ~73-85 in `xgboost_multimodal_classification.py`:

```python
CLASSIFICATION_TASKS = {
    'HighStress': ['HighStress_HighCog1022_Task', 'HighStress_LowCog_Task'],
    'LowStress': ['LowStress_HighCog1022_Task', 'LowStress_LowCog_Task'],
    'HighCog': ['HighStress_HighCog1022_Task', 'LowStress_HighCog1022_Task'],
    'LowCog': ['HighStress_LowCog_Task', 'LowStress_LowCog_Task'],
}
```

### 2. Create Task-to-Round Mapping

Add after CLASSIFICATION_TASKS:

```python
# Mapping from condition names to counterbalance round descriptions
CONDITION_TO_ROUND_NAME = {
    'HighStress_HighCog1022_Task': 'Stress Subtraction',
    'HighStress_LowCog_Task': 'Stress Addition',
    'LowStress_HighCog1022_Task': 'Calm Subtraction',
    'LowStress_LowCog_Task': 'Calm Addition',
}
```

### 3. Fix Counterbalance File Path

Replace line ~137:

```python
COUNTERBALANCE_FILE = 'data/experimental_counterbalance.xlsx'  # NOT 'data/processed/VR-TSST...'
```

### 4. Rewrite get_baseline() Function

Replace the entire get_baseline() function (lines ~345-417) with:

```python
def get_baseline(df, participant_id, task_condition, duration):
    """
    Get baseline data for a participant's task.
    
    Args:
        df: Full dataframe with all conditions
        participant_id: Participant ID
        task_condition: Task condition name (e.g., 'HighStress_HighCog1022_Task')
        duration: Duration in seconds
        
    Returns:
        DataFrame with baseline windows for this participant/task
    """
    try:
        # Load counterbalance sheet
        cb_df = pd.read_excel(COUNTERBALANCE_FILE)
        
        # Get participant's counterbalance data
        participant_cb = cb_df[cb_df['Participant'] == participant_id]
        if len(participant_cb) == 0:
            return None
            
        # Map task condition to round description
        round_description = CONDITION_TO_ROUND_NAME.get(task_condition)
        if round_description is None:
            return None
        
        # Find which round (1-4) this task was performed in
        task_round = None
        for round_num in range(1, 5):
            round_col = f'Round {round_num}'
            if participant_cb[round_col].values[0] == round_description:
                task_round = round_num
                break
        
        if task_round is None:
            return None
        
        # Get corresponding Forest baseline
        baseline_condition = f'Forest{task_round}'
        
        # Get baseline data for this participant
        baseline_data = df[
            (df['Participant_ID'] == participant_id) & 
            (df['Condition'] == baseline_condition)
        ].copy()
        
        if len(baseline_data) == 0:
            return None
        
        # Apply duration windowing if needed
        if duration < baseline_data['Window_Index'].max():
            baseline_data = baseline_data[baseline_data['Window_Index'] <= duration]
        
        return baseline_data
        
    except Exception as e:
        print(f"Error getting baseline for P{participant_id}, {task_condition}: {e}")
        return None
```

---

## Expected Results After Fix

- **Task conditions properly recognized**: All 4 task conditions matched
- **Baseline matching works**: Each task condition paired with correct Forest baseline
- **Full preprocessing exploration**: 3 baselines × 3 normalizations × 4 durations = 36 strategies
- **Complete model comparison**: 36 × 8 models × 3 modalities × 2 tasks = **1,728 total runs**
- **Runtime**: ~4-6 hours with parallelization

---

## Test After Applying Fix

Run diagnostic again to verify:
```bash
python pipelines/08_xgboost_multimodal/diagnose_baseline_matching.py
```

Should show successful baseline matches for test participants.

Then run full pipeline:
```bash
python pipelines/08_xgboost_multimodal/xgboost_multimodal_classification.py
```

Expected output: "Total strategies tested: 1728" (not 504!)
