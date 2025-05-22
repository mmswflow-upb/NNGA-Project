# plot_utils.py

import time
import numpy as np
import matplotlib.pyplot as plt

# These will be set by the caller before first animate() call:
N = None            # board size (e.g. 8)
VIS_DELAY = None    # seconds between queen drops

# Internal state
_stop_requested = False
_fig = None
_ax  = None


def init_plot(board_size: int, vis_delay: float):
    """
    Initialize the plotting window and close-event handler.
    Must be called once before animate().
    """
    global N, VIS_DELAY, _fig, _ax
    N = board_size
    VIS_DELAY = vis_delay

    plt.ion()
    _fig, _ax = plt.subplots(figsize=(6, 6))
    _fig.canvas.mpl_connect("close_event", _on_close)


def _on_close(event):
    """Internal callback: user closed the figure."""
    global _stop_requested
    _stop_requested = True


def draw_blank():
    """Draw an empty N×N checkerboard on the current axes."""
    board = np.fromfunction(lambda i, j: (i + j) % 2, (N, N))
    _ax.imshow(board, cmap="binary", interpolation="nearest")
    _ax.set_xticks([])
    _ax.set_yticks([])


def animate(mat: np.ndarray, gen: int, fit: int, max_fit: int) -> bool:
    """
    Draw the given board matrix, placing queens one-by-one.

    Args:
      mat     : N×N binary matrix (1=queen, 0=empty)
      gen     : current generation number
      fit     : fitness of this board
      max_fit : theoretical maximum fitness

    Returns:
      False if the user has closed the figure (stop requested),
      True otherwise.
    """
    if _stop_requested or not plt.fignum_exists(_fig.number):
        return False

    _ax.clear()
    draw_blank()

    # sort by column so queens drop left→right
    coords = sorted(zip(*np.where(mat == 1)), key=lambda rc: rc[1])
    for k, (r, c) in enumerate(coords, 1):
        _ax.text(
            c, r, u"\u265B",
            ha='center', va='center',
            fontsize=24, color='red'
        )
        _ax.set_title(
            f"Gen {gen} | Fit {fit}/{max_fit}\n"
            f"Placing queen {k}/{N}"
        )
        _fig.canvas.draw()
        _fig.canvas.flush_events()
        time.sleep(VIS_DELAY)

        if _stop_requested:
            return False

    time.sleep(VIS_DELAY)
    return True


def finalize():
    """
    Turn off interactive mode and show the final plot.
    Call this at the very end of your GA.
    """
    plt.ioff()
    if _fig and plt.fignum_exists(_fig.number):
        plt.show()
