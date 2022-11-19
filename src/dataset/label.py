import numpy as np
import pandas as pd


class SongLabels:
    """Loader of a song's labels.

    The main method is `from_middle_points` that returns the exact pressed pitches
    for each timsteps given. This method end in `O(n)`, where `n` is the number of
    events (keys pressed/released) in the song.

    Parameters
    ----------
        labels: DataFrame containing the `start_time`, `end_time`, `note`
            and `instrument` information.
        n_pitches: Number of pitches.
        n_instruments: Number of instruments.
    """

    def __init__(
        self,
        labels: pd.DataFrame,
        n_pitches: int,
        n_instruments: int,
    ):
        self.n_pitches = n_pitches
        self.n_instruments = n_instruments
        self.events = []  # Tuples (time, pitch, instrument, is_pressed).

        for start, end, note, instrument in labels[
            ["start_time", "end_time", "note", "instrument"]
        ].values:
            self.events.append((start, note, instrument, True))
            self.events.append((end, note, instrument, False))

        self.events = list(sorted(self.events, key=lambda el: el[0]))  # Sort by time.

    def from_middle_points(self, middle_points: np.ndarray) -> np.ndarray:
        """Return the labels for each points given in that song.
        This is done in `O(n)` time, where `n` is `len(self)`.

        Input
        -----
            middle_points: Array of timesteps.
                Shape of [n_points,].

        Output
        ------
            labels: One-hot vectors for each timesteps.
                Shape of [n_points, n_instruments, n_pitches].
        """
        labels = np.zeros(
            (len(middle_points), self.n_instruments, self.n_pitches), dtype=np.uint8
        )

        point_id = 0
        previous_labels = np.zeros((self.n_instruments, self.n_pitches), dtype=np.uint8)
        iter_label = iter(self)
        curr_labels, curr_time = next(iter_label)

        try:
            # We go through all events and fill the labels until
            # all are processed or events are terminated.
            while point_id < len(middle_points):
                if curr_time >= middle_points[point_id]:
                    labels[point_id] = previous_labels
                    point_id += 1
                else:
                    previous_labels = curr_labels
                    curr_labels, curr_time = next(iter_label)
        except StopIteration:
            # The last middle points are after the end of the song,
            # so we assume the end of the song is containing no labels.
            while point_id < len(middle_points):
                labels[point_id] = np.zeros(
                    (self.n_instruments, self.n_pitches), dtype=int
                )
                point_id += 1

        return labels

    def __len__(self) -> int:
        """Return the number of events in this song.
        One event is a key being pressed on or off for a specific instrument.
        This is one row of the MusicNet dataframes.
        """
        return len(self.events)

    def __iter__(self):
        """Returns itself.
        It initializes some variables for a properly fresh iteration through all events.
        """
        self.curr_index = 0
        self.curr_labels = np.zeros(
            (self.n_instruments, self.n_pitches), dtype=np.uint8
        )
        self.first_iter = True
        return self

    def __next__(self) -> tuple[np.ndarray, int]:
        """Next iteration through the events.
        It allows the user to iterate in chronological order.
        If some events occurs at the same time, they will be gathered into one
        iteration.

        Output
        -----
            label: One-hot vector for the current time step.
                Shape of [n_instruments, n_pitches].
            time: Time step corresponding to this label. This time step is
                rescaled to take into account the `downsampling_factor`.
        """
        if self.curr_index >= len(self):
            raise StopIteration

        if self.first_iter:  # Return the labels for the first iteration
            self.first_iter = False
            return self.curr_labels.copy(), 0

        # Do update the labels while the time event does not change
        first_iter = True  # Do-while loop
        while first_iter or (
            self.curr_index < len(self)
            and (
                self.events[self.curr_index - 1][0] == self.events[self.curr_index][0]
            )  # Is the current event at the same time as the preceding one?
        ):
            time, note, instrument, is_pressed = self.events[self.curr_index]
            self.curr_labels[instrument - 1, note - 1] = int(is_pressed)

            self.curr_index += 1
            first_iter = False

        return self.curr_labels.copy(), time
