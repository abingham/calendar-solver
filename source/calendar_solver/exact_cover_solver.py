from dataclasses import dataclass
from typing import Iterable, Sequence
import more_itertools
import numpy
import exact_cover


@dataclass
class Tile:
    prototile_id: int
    arr: numpy.ndarray

    @property
    def cells(self) -> Iterable[tuple[int, int]]:
        "The cells occupied by this tile without reference to a board."
        return zip(*numpy.nonzero(self.arr))


@dataclass
class Placement:
    tile: Tile
    row: int
    col: int

    @property
    def cells(self) -> Iterable[tuple[int, int]]:
        """The cells occupied by this tile on the board.
        
        This are the tile's cells offset by the row and column of the placement.
        """
        for row, col in self.tile.cells:
            yield self.row + row, self.col + col


@dataclass
class Board:
    arr: numpy.ndarray

    @property
    def width(self) -> int:
        return self.arr.shape[1]

    @property
    def height(self) -> int:
        return self.arr.shape[0]

    @property
    def reserved_positions(self) -> Iterable[tuple[int, int]]:
        return zip(*numpy.nonzero(self.arr))

    def can_place(self, placement: Placement) -> bool:
        """Check if a tile can be placed on the board."""
        tile = placement.tile.arr
        row, col = placement.row, placement.col

        # Get slice of board where the tile will go
        board_slice = self.arr[row : row + tile.shape[0], col : col + tile.shape[1]]

        # If the slice and tile are different shapes, then the placement is off the board
        if board_slice.shape != tile.shape:
            return False

        # Check if the tile overlaps with "reserved" places on the board.
        if numpy.any(board_slice & tile):
            return False

        return True


def make_placements(prototiles: Iterable[numpy.ndarray], board: Board):
    """Generate all possible placements of prototiles on the board."""
    for prototile_id, prototile in enumerate(prototiles):
        for o in make_orientations(prototile):
            for row in range(0, (board.arr.shape[0] - o.shape[0]) + 1):
                for col in range(0, (board.arr.shape[1] - o.shape[1]) + 1):
                    placement = Placement(Tile(prototile_id, o), row, col)
                    if board.can_place(placement):
                        yield placement


def make_orientations(prototile: numpy.ndarray) -> Iterable[numpy.ndarray]:
    """Generate all unique orientations of a prototile."""

    def rotations():
        yield prototile
        yield numpy.rot90(prototile)
        yield numpy.rot90(prototile, k=2)
        yield numpy.rot90(prototile, k=3)

    return more_itertools.unique_everseen(
        rotations(),
        key=lambda tile: tile.tobytes() + str(tile.shape).encode("utf-8"),
    )


class SolutionMatrix:
    def __init__(self, board: Board, prototiles: Sequence[numpy.ndarray]):
        self.board = board
        self.num_prototiles = len(prototiles)
        self.placements = list(make_placements(prototiles, board))

        self._allocate_matrix(board, prototiles)

        # Initialize first row with reserved board locations
        self._initialize_reserved_row()

        # Set up the solution matrix row for each placement
        self._initialize_placement_rows()

    def solve(self):
        solution_row_indices = exact_cover.get_exact_cover(self.arr)
        return self._solution_rows_to_board(solution_row_indices)

    def all_solutions(self):
        for solution_row_indices in exact_cover.get_all_solutions(self.arr):
            yield self._solution_rows_to_board(solution_row_indices)

    def _solution_rows_to_board(self, solution_row_indices):
        result = self.board.arr.copy()
        for solution_row_index in solution_row_indices:
            prototile_id = self._get_prototile_id(solution_row_index)
            if prototile_id is not None:
                board_value = prototile_id + 2
            else:
                board_value = 1

            for row, col in self._get_board_coordinates(solution_row_index):
                result[row, col] = board_value

        return result

    def _allocate_matrix(self, board, prototiles):
        # Solution matrix shape:
        # - one row for "reserved" board locations
        # - one row for each placement
        # - one column for each prototile
        # - one column for each board location
        self.arr = numpy.zeros((len(self.placements) + 1, len(prototiles) + board.arr.size), dtype=numpy.int8)

    def _initialize_reserved_row(self):
        for row, col in self.board.reserved_positions:
            self.arr[0, self._coord_to_col(row, col)] = 1

    def _initialize_placement_rows(self):
        for i, placement in enumerate(self.placements):
            # Set the column for this piece
            self.arr[i + 1, placement.tile.prototile_id] = 1

            # Set the columns for the board locations covered by this piece
            for cell in placement.cells:
                self.arr[i + 1, self._coord_to_col(*cell)] = 1

    @property
    def num_placements(self):
        return len(self.placements)

    def _coord_to_col(self, row, col):
        return (row * self.board.width + col) + self.num_prototiles

    def _col_to_coord(self, solution_col):
        """Convert a column index in the solution matrix to a coordinate on the board.

        Args:
            solution_col: The column index in the solution matrix.

        Returns:
            A tuple (row, col) representing the coordinate.
        """
        solution_col -= self.num_prototiles
        row = solution_col // self.board.width
        col = solution_col % self.board.width
        return row, col

    def _get_prototile_id(self, solution_row_index) -> int | None:
        row = self.arr[solution_row_index, :self.num_prototiles]
        indices = numpy.nonzero(row)[0]
        if len(indices) == 0:
            return None

        assert len(indices) == 1, "Multiple prototile IDs found in the solution row."

        return int(indices[0])

    def _get_board_coordinates(self, solution_row_index):
        solution_row = self.arr[solution_row_index, self.num_prototiles:]
        position_cols = numpy.nonzero(solution_row)[0]
        for position_col in position_cols:
            board_row, board_col = self._col_to_coord(position_col + self.num_prototiles)
            yield board_row, board_col


def main():
    BOARD = numpy.array(
        [
            [0, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=numpy.int8,
    )

    PROTOTILES = [
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

    solver = SolutionMatrix(Board(BOARD), PROTOTILES) 
    solutions = tuple(solver.all_solutions())
    for solution in solutions:
        print(solution)
    print("Number of solutions:", len(solutions))



if __name__ == "__main__":
    main()
