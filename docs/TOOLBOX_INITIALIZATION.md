# Toolbox Initialization - Permanent Fixes

## Problem
Previous setup had reactive fixes (guards, fallbacks) but toolbox/plugin discovery kept failing in batch mode because:
1. **yamlmatlab**: No vendored copy in repo; hardcoded system paths often wrong or missing
2. **EEGLAB, AMICA**: Versioned folders not detected; batch mode doesn't auto-discover paths
3. **Batch mode limitations**: `startup.m` runs but path initialization wasn't robust

## Solutions Implemented

### 1. **SimpleYAML** — Project-local YAML reader
**File**: `scripts/utils/SimpleYAML.m`
- Custom minimal YAML parser for simple key:value configs
- No external dependency on yamlmatlab
- Fallback if ReadYaml unavailable
- Always available after `addpath(fullfile(pwd, 'scripts', 'utils'))`

### 2. **Robust startup.m** — Intelligent toolbox discovery
**File**: `startup.m`
Key improvements:
- Uses `matlabroot()` to find system toolbox locations dynamically
- **Detects versioned folders** (e.g., `eeglab2025.1.0`) via wildcard matching
- Tries multiple fallback locations before giving up
- Works reliably in both interactive and batch modes
- Graceful degradation with warnings instead of hard errors
- Verifies critical EEGLAB functions after loading

**Path search order** (per toolbox):
1. Standard MATLAB system path: `{matlabroot}/toolbox/{name}`
2. Versioned system path: `c:\MATLAB\toolboxes\{name}*` (wildcard match)
3. Hardcoded standard locations: `c:\MATLAB\toolboxes`, `c:\Program Files\MATLAB\toolboxes`, etc.
4. Project-local: `a/{name}`, `scripts/lib/{name}`, `scripts/utils/{name}`

### 3. **test_toolbox_init.m** — Sanity check script
**File**: `scripts/preprocessing/eeg/cleaning/test_toolbox_init.m`
- Validates all required functions are on path after startup
- Checks for YAML reader (ReadYaml or SimpleYAML)
- Safe to run in batch mode; exits with code 0 (success) or 1 (failure)

**Usage**:
```matlab
matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/test_toolbox_init.m')"
```

### 4. **Updated run_clean_eeg_parallel.m** — Dual YAML support
**File**: `scripts/preprocessing/eeg/cleaning/run_clean_eeg_parallel.m`
- Tries ReadYaml first (if yamlmatlab installed)
- Falls back to SimpleYAML (project-local)
- Continues gracefully if config loading fails

## Testing

**Verify setup** (batch mode):
```powershell
matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/test_toolbox_init.m')"
```

Expected output:
```
[startup] eeglab loaded (versioned system: ...)
[startup] amica loaded (...)
[startup] yamlmatlab loaded (...)
[startup] Using SimpleYAML as YAML fallback.
...
RESULT: ✓ ALL TESTS PASSED
```

**Run cleaning**:
```powershell
matlab -batch "run('startup.m'); run('scripts/preprocessing/eeg/cleaning/run_clean_eeg_parallel.m')"
```

## Why These Fixes Are Permanent

1. **No external vendoring needed** — SimpleYAML is ~100 lines of MATLAB code, in the repo
2. **Self-healing discovery** — startup.m adapts to system changes (new MATLAB version, new folder structure)
3. **Batch mode compatible** — Uses only core MATLAB functions (no advanced API)
4. **Graceful degradation** — Warnings instead of crashes; fallbacks if toolbox missing
5. **Testable** — test_toolbox_init.m validates the entire chain before pipeline runs

## Future Maintenance

If MATLAB installation changes:
1. Run `test_toolbox_init.m` to diagnose issues
2. startup.m will auto-detect versioned folders
3. If custom toolbox location is needed, add to `commonDirs` in startup.m

No more manual path editing or batch mode surprises.
