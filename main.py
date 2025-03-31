import numpy
import more_itertools

def orientations(piece: numpy.ndarray) -> tuple[numpy.ndarray, ...]:
    """Return all orientations of a piece."""

    def rotated(p):
        return [
            p,
            numpy.rot90(p),
            numpy.rot90(p, 2),
            numpy.rot90(p, 3),
        ]

    return tuple(more_itertools.unique_everseen(rotated(piece) + rotated(numpy.flip(piece, axis=0)), key=lambda x: x.tobytes()))

def count_zeros(arr: numpy.ndarray) -> int:
    """Count the number of zeros in an array."""
    return arr.size - numpy.count_nonzero(arr)

COUNTER = 0

def solve(board, buckets):
    """Solve the board with the given buckets."""
    global COUNTER

    if not buckets:
        return False

    for piece in buckets[0]:
        COUNTER += 1
        print(COUNTER, len(buckets))
        for i in range(0, board.shape[0] - piece.shape[0] + 1):
            for j in range(0, board.shape[1] - piece.shape[1] + 1):
                pre_zeros = count_zeros(board)
                expected_post_zeros = pre_zeros - numpy.count_nonzero(piece)

                # Place the piece
                board[i:i+piece.shape[0], j:j+piece.shape[1]] += piece
                post_zeros = count_zeros(board)
                if post_zeros == expected_post_zeros:
                    if post_zeros == 0:
                        return True

                    if solve(board, buckets[1:]):
                        # print(board)
                        return True
                
                # Remove the piece
                board[i:i+piece.shape[0], j:j+piece.shape[1]] -= piece

    return False

BOARD = numpy.array([
    [0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
])

PIECES = [
    numpy.array([[1, 1, 1, 1, 1]]),
    numpy.array([[1, 1, 1], 
                 [1, 0, 0], 
                 [1, 0, 0]]),
    numpy.array([[1, 1, 1, 1], 
                 [0, 1, 0, 0]]),
    numpy.array([[1, 1, 1], 
                 [1, 1, 0]]),
    numpy.array([[0, 0, 1], 
                 [1, 1, 1], 
                 [1, 0, 0]]),
    numpy.array([[1, 1, 1], 
                 [0, 1, 0], 
                 [0, 1, 0]]),
    numpy.array([[1, 1, 0], 
                 [0, 1, 1], 
                 [0, 1, 0]]),
    numpy.array([[0, 0, 1, 1], 
                 [1, 1, 1, 0]]),
    numpy.array([[1, 1, 1, 1], 
                 [1, 0, 0, 0]]),
    numpy.array([[1, 1, 1], 
                 [1, 0, 1]]),
]

BUCKETS = [
    orientations(piece * index)
    for index, piece in enumerate(PIECES, start=2)
]

solved = solve(BOARD, BUCKETS)
print("solved:", solved)
print(BOARD)