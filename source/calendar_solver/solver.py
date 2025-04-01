from itertools import chain
from typing import Iterable
import more_itertools
import numba
import numba.typed.typedlist
import numpy


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


@numba.njit
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
    buckets = numba.typed.typedlist.List()
    for index, piece in enumerate(pieces, start=2):
        bucket = numba.typed.typedlist.List()
        for orientation in orientations(piece * index):
            # This "copy" seems to be necessary to make sure all of our pieces have the same layout. I guess views are problematic for numba, but I'm not 100% sure.
            bucket.append(orientation.copy())
        buckets.append(bucket)

    result = _solve(board, buckets)

    if result == 0:
        assert count_zeros(board) == 0, "The board should have no zeros if it's solved."

    return result == 0


@numba.njit
def _solve(board, buckets, counter=0):
    """Numba-based recursive function to solve the board.

    Returns:
        0 if the board is solved, otherwise returns the number of total iterations.
    """
    if not buckets:
        # TODO: This is a bit wrong. If we're called initially with no buckets, we return 0 here indicating "solved" when in fact we've solved nothing.
        return counter

    for piece in buckets[0]:
        for i in range(0, board.shape[0] - piece.shape[0] + 1):
            for j in range(0, board.shape[1] - piece.shape[1] + 1):
                counter += 1
                if counter % 1_000_000 == 0:
                    print(counter)
                pre_zeros = count_zeros(board)
                expected_post_zeros = pre_zeros - numpy.count_nonzero(piece)

                # Place the piece
                board[i : i + piece.shape[0], j : j + piece.shape[1]] += piece
                post_zeros = count_zeros(board)
                if post_zeros == expected_post_zeros:
                    if post_zeros == 0:
                        return 0

                    counter = _solve(board, buckets[1:], counter)
                    if counter == 0:
                        return 0

                # Remove the piece
                board[i : i + piece.shape[0], j : j + piece.shape[1]] -= piece

    return counter


def main():
    BOARD = numpy.array(
        [
            [0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=numpy.int8,
    )

    PIECES = [
        numpy.array([[1, 1, 1, 1, 1]], dtype=numpy.int8),
        numpy.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 1, 1], [0, 1, 0, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 1], [1, 1, 0]], dtype=numpy.int8),
        numpy.array([[0, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=numpy.int8),
        numpy.array([[0, 0, 1, 1], [1, 1, 1, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=numpy.int8),
        numpy.array([[1, 1, 1], [1, 0, 1]], dtype=numpy.int8),
    ]

    solved = solve(BOARD, PIECES)
    print("solved:", solved)
    print(BOARD)


if __name__ == "__main__":
    main()
