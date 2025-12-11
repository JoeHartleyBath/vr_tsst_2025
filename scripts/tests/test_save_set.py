import os
import tempfile
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy.io import loadmat

from scripts.xdf_to_set.xdf_to_set import save_set


def test_save_set_writes_matfile_and_contains_EEG():
    # Build a minimal EEG-like dict
    eeg_struct = {
        "data": np.zeros((2, 10)),
        "srate": 500,
        "nbchan": 2,
        "pnts": 10,
        "trials": 1,
        "times": np.arange(10),
        "chanlocs": [{"labels": "C1"}, {"labels": "C2"}],
        "event": [{"latency": 1, "type": 10}],
        "urevent": [{"latency": 1, "type": 10}],
    }

    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "test_output.set")

        # Should not raise
        saved = save_set(eeg_struct, out_path)

        assert os.path.exists(saved)

        # Loading should succeed and contain 'EEG'
        mat = loadmat(saved)
        assert "EEG" in mat
