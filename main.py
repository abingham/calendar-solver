import numba.typed.typedlist
import numpy
import more_itertools
from numba.typed.typedlist import List


def orientations(piece: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    """Return all orientations of a piece."""

    def rotated(p):
        return [
            p,
            numpy.rot90(p),
            numpy.rot90(p, 2),
            numpy.rot90(p, 3),
        ]

    return tuple(
        piece.copy()
        for piece in more_itertools.unique_everseen(
            rotated(piece) + rotated(numpy.flip(piece, axis=0)),
            key=lambda x: x.tobytes(),
        )
    )


@numba.njit
def count_zeros(arr: numpy.ndarray) -> int:
    """Count the number of zeros in an array."""
    return arr.size - numpy.count_nonzero(arr)


def solve(board, buckets):
    new_buckets = List()
    for bucket in buckets:
        new_bucket = List()
        for piece in bucket:
            new_bucket.append(piece)
        new_buckets.append(new_bucket)
    return _solve(board, new_buckets)


@numba.njit
def _solve(board, buckets, counter=0):
    """Solve the board with the given buckets."""
    if not buckets:
        return False

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

                    counter =_solve(board, buckets[1:], counter)
                    if counter == 0:
                        return 0

                # Remove the piece
                board[i : i + piece.shape[0], j : j + piece.shape[1]] -= piece

    return counter


BOARD = numpy.array(
    [
        [0, 0, 1, 0, 0, 0, 0, 1, 0],
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

BUCKETS = [orientations(piece * index) for index, piece in enumerate(PIECES, start=2)]

solved = solve(BOARD, BUCKETS)
print("solved:", solved)
print(BOARD)
