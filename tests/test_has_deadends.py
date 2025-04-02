import unittest
from calendar_solver.solver import has_deadends
import numpy


class TestHasDeadends(unittest.TestCase):
    def test_has_deadends(self):
        board = numpy.array([
            [0, 0, 0],
            [0, 2, 2],
            [1, 2, 2],
        ], dtype=numpy.int8)

        self.assertTrue(has_deadends(board))

    def test_has_no_deadends(self):
        board = numpy.array([
            [0, 0, 0],
            [0, 2, 2],
            [0, 2, 2],
        ], dtype=numpy.int8)

        self.assertFalse(has_deadends(board))