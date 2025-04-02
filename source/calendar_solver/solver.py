from itertools import chain
from typing import Iterable

import more_itertools
import numpy
import scipy.ndimage


def orientations(piece: numpy.ndarray) -> Iterable[numpy.ndarray]:
    """Return all orientations of a piece.

    This rotates a piece and its flipped version, and returns all unique
    orientations.
    """

    def rotated(p):
        "Get all rotations of a piece: 0, 90, 180, and 270 degrees."
        yield p
        yield numpy.rot90(p)
        yield numpy.rot90(p, 2)
        yield numpy.rot90(p, 3)

    return more_itertools.unique_everseen(
        chain(rotated(piece), rotated(numpy.flip(piece, axis=0))),
        key=lambda x: x.tobytes(),
    )


def count_zeros(arr: numpy.ndarray) -> int:
    """Count the number of zeros in an array."""
    return arr.size - numpy.count_nonzero(arr)


def solve(board, pieces):
    """Solve the board with the given buckets.

    This is the top-level solver function. It defers to a numba-based recursive implementation.

    Args:
        board: The 2D numpy array representing the board. It should have zeroes in any empty spaces, and any positive integer elsewhere.
        pieces: An iterable of 2D numpy arrays representing the pieces to place on the board. Each piece should have 1 on the squares it occupies, and 0 elsewhere.

    Returns: True if the board is solved, False otherwise. The `board` argument will contain the final state of the board, solved or not.
    """
    # We have to copy the oriented pieces into numba-specific lists to make them amenable to numba.
    buckets = [tuple(orientations(piece * index)) for index, piece in enumerate(pieces, start=2)]
    result = _solve(board, buckets)

    if result == 0:
        assert count_zeros(board) == 0, "The board should have no zeros if it's solved."

    return result == 0


def has_deadends(board, min_hole_size):
    """Check if the board has dead ends.

    A dead end is a cell that cannot be filled with any piece. This is a simple
    heuristic to prune the search space.
    """
    ones = numpy.ones_like(board, dtype=numpy.int8)
    ones = numpy.logical_xor(ones, board)
    inverse = numpy.array(ones, dtype=numpy.int8)
    label, num_features = scipy.ndimage.label(inverse)
    unique, counts = numpy.unique(label, return_counts=True)
    counts = dict(zip(unique, counts))

    for value in range(1, num_features + 1):
        if counts[value] < min_hole_size:
            # This means we have a dead end. We can prune this branch.
            return True

    return False


def _solve(board, buckets):
    """DFS solver for the board.

    Returns:
        True if the puzzle is solved, False otherwise.
    """
    if not buckets:
        return False

    for piece in buckets[0]:
        for i in range(0, board.shape[0] - piece.shape[0] + 1):
            for j in range(0, board.shape[1] - piece.shape[1] + 1):
                pre_zeros = count_zeros(board)
                expected_post_zeros = pre_zeros - numpy.count_nonzero(piece)

                # Place the piece
                board[i : i + piece.shape[0], j : j + piece.shape[1]] += piece
                post_zeros = count_zeros(board)
                if (post_zeros == expected_post_zeros) and (not has_deadends(board, 5)):
                    if post_zeros == 0:
                        return True

                    if _solve(board, buckets[1:]):
                        return True

                # Remove the piece
                board[i : i + piece.shape[0], j : j + piece.shape[1]] -= piece

    return False


def main():
    BOARD = numpy.array(
        [
            [0, 0, 0, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=numpy.int8,
    )

    PIECES = [
        numpy.array([[1, 1, 1, 1, 1]], dtype=numpy.int8),  # 2
        numpy.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=numpy.int8),  # 4
        numpy.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=numpy.int8),  # 8
        numpy.array([[1, 1, 1], [1, 1, 0]], dtype=numpy.int8),  # 8
        numpy.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=numpy.int8),  # 8
        numpy.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=numpy.int8),  # 4
        numpy.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=numpy.int8),  # 8
        numpy.array([[0, 0, 1, 1], [1, 1, 1, 0]], dtype=numpy.int8),  # 8
        numpy.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=numpy.int8),  # 8
        numpy.array([[1, 1, 1], [1, 0, 1]], dtype=numpy.int8),  # 4
    ]

    solved = solve(BOARD, PIECES)
    print("solved:", solved)
    print(BOARD)


if __name__ == "__main__":
    main()
