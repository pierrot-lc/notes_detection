import numpy as np
import pandas as pd
import pytest

from src.dataset.label import SongLabels


@pytest.mark.parametrize(
    "filepath, timesteps",
    [
        ("./data/musicnet/train_labels/1727.csv", [0, 1000, 10000]),  # Base case.
        (
            "./data/musicnet/train_labels/1727.csv",
            [0, 272727, 1000000000, 1000000001, 1000000002],
        ),  # Extreme case with no labels.
        (
            "./data/musicnet/train_labels/1727.csv",
            [0, 9182, 62430],
        ),  # Test on exact start_time and end_time.
    ],
)
def test_from_timesteps(filepath: str, timesteps: list[int]):
    df = pd.read_csv(filepath)
    labels = SongLabels(df, df["note"].max(), df["instrument"].max())
    labels = labels.from_timesteps(np.array(timesteps))

    for timestep, label in zip(timesteps, labels):
        true_label = np.zeros_like(label)
        for start_time, end_time, instrument_id, note_id in df[
            ["start_time", "end_time", "instrument", "note"]
        ].values:
            if start_time <= timestep < end_time:
                true_label[instrument_id - 1, note_id - 1] = 1

        assert (true_label == label).all()
