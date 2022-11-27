import numpy as np
import pandas as pd
import pytest

from src.dataset.label import SongLabels


def load_song_label(filepath: str) -> SongLabels:
    df = pd.read_csv(filepath)
    n_pitches = df["note"].max()
    n_instruments = df["instrument"].max()
    return SongLabels(df, n_pitches, n_instruments)


@pytest.mark.parametrize(
    "filepath",
    ["./data/musicnet/train_labels/1775.csv", "./data/musicnet/test_labels/2628.csv"],
)
def test_len(filepath: str):
    song_label = load_song_label(filepath)
    assert len(song_label) == len(song_label.events)
