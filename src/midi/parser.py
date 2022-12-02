import numpy as np


def frames_to_frame_counts(frames: np.ndarray) -> list[tuple[int, int, int]]:
    """Convert the frames to a list of events.
    Each event is of the form (pitch_id, start_frame, end_frame).

    ----
    Args
        frames: Array of pitches for each frames.
            Shape of [n_frames, n_notes].

    -------
    Returns
        frame_counts: List of events of length `n_frames`.
    """
    n_notes = frames.shape[1]
    durations = np.zeros(n_notes, dtype=int)
    onsets = np.zeros(n_notes, dtype=bool)
    frame_counts = []

    for frame_id, sample in enumerate(frames):
        for pitch_id in range(n_notes):
            # New event if the onset is changing and
            # if durations of the current pitch is not 0.
            # Durations can be 0 at the beginning of the samples.
            new_event = (
                onsets[pitch_id] != sample[pitch_id] and durations[pitch_id] != 0
            )

            # Add a new event if the note is "on".
            if new_event and onsets[pitch_id]:
                frame_counts.append(
                    (pitch_id, frame_id - durations[pitch_id], frame_id - 1)
                )  # (pitch_id, start_frame, end_frame)

        durations += 1
        durations[sample != onsets] = 1
        onsets = sample.astype("bool")

    # Add the pending notes.
    for pitch_id in range(n_notes):
        if onsets[pitch_id]:
            frame_counts.append(
                (pitch_id, len(frames) - durations[pitch_id], len(frames) - 1)
            )

    return frame_counts


def frames_to_timesteps(
    frames: np.ndarray, sample_rate: int
) -> list[tuple[int, float, float]]:
    """Convert the frames to a list of events.
    Each event is of the form (pitch_id, start_time, end_time).

    ----
    Args
        frames: Array of pitches for each frames.
            Shape of [n_frames, n_notes].
        sample_rate: Sample rate for the given frames (Hz).

    -------
    Returns
        timesteps: List of events of length `n_frames`.
    """
    # Go from frames to list of events with frame counts.
    frame_counts = frames_to_frame_counts(frames)

    # Go from frame counts to duration using sample rate.
    timesteps = [
        (pitch_id, start_frame / sample_rate, end_frame / sample_rate)
        for pitch_id, start_frame, end_frame in frame_counts
    ]

    return timesteps
