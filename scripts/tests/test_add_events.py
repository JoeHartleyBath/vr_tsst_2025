import numpy as np

from scripts.xdf_to_set.xdf_to_set import add_events_to_eeg_struct


def test_add_events_to_eeg_struct_basic():
    """Basic unit test for add_events_to_eeg_struct.

    Verifies that:
    - `event` and `urevent` are added
    - latencies are converted to 1-based indices
    - event types are preserved
    - the original EEG dict is not mutated
    """

    EEG = {
        "data": np.zeros((2, 10)),
        "srate": 500,
        "nbchan": 2,
        "pnts": 10,
        "trials": 1,
        "times": np.arange(10),
        "chanlocs": [{"labels": "C1"}, {"labels": "C2"}],
    }

    events = [
        {"latency": 0, "type": 10},
        {"latency": 4, "type": 20},
    ]

    # Keep a shallow copy to assert immutability of the original dict
    original_keys = set(EEG.keys())

    out = add_events_to_eeg_struct(EEG, events)

    assert "event" in out and "urevent" in out
    assert len(out["event"]) == 2

    # Latencies should be converted to 1-based indices
    assert out["event"][0]["latency"] == 1
    assert out["event"][1]["latency"] == 5

    # Types preserved
    assert out["event"][0]["type"] == 10
    assert out["urevent"][1]["type"] == 20

    # Original EEG must not be mutated (no 'event' key added)
    assert set(EEG.keys()) == original_keys
