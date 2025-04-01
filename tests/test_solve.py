import numpy
from calendar_solver.solver import solve
import unittest


class TestSolver(unittest.TestCase):
    def test_basic(self):
        BOARD = numpy.array(
            [
                [0, 0, 0,],
                [0, 1, 0,],
                [0, 0, 0,],
            ],
            dtype=numpy.int8,
        )

        PIECES = [
            numpy.array([
                [1, 1, 1,],
                [1, 0, 0,],
                [1, 0, 0,],
                ], dtype=numpy.int8),
            numpy.array([
                [1, 1,],
                [1, 0,],
                ], dtype=numpy.int8),

        ]

        solved = solve(BOARD, PIECES)
        self.assertTrue(solved)

        expected_board = numpy.array(
            [
                [2, 2, 2,],
                [2, 1, 3,],
                [2, 3, 3,],
            ],
            dtype=numpy.int8,
        )
        self.assertEqual(BOARD.tolist(), expected_board.tolist())