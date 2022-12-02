import numpy as np
import pytest

from src.midi.parser import frames_to_frame_counts


@pytest.mark.parametrize(
    "frames, frame_counts",
    [
        (
            np.array(
                [
                    [1, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [1, 0, 1, 0, 1],
                ]
            ),
            [
                (3, 0, 0),
                (3, 2, 2),
                (0, 0, 3),
                (2, 1, 3),
                (4, 3, 3),
            ],
        ),
        (
            np.array(
                [
                    [0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 0, 0],
                    [1, 0, 1, 1, 0],
                    [0, 0, 0, 0, 1],
                ]
            ),
            [
                (3, 0, 0),
                (0, 1, 3),
                (2, 1, 3),
                (3, 3, 3),
                (4, 4, 4),
            ],
        ),
        (np.array([[]]), []),
    ],
)
def test_frames_to_frame_counts(
    frames: np.ndarray, frame_counts: list[tuple[int, int, int]]
):
    assert frames_to_frame_counts(frames) == frame_counts
