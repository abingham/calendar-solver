from calendar_solver.exact_cover_solver import col_to_coord, coord_to_col


def test_coord_to_col():
    assert coord_to_col((0, 0), 10, 5) == 10
    assert coord_to_col((0, 1), 10, 5) == 11
    assert coord_to_col((1, 0), 10, 5) == 15
    assert coord_to_col((1, 2), 10, 5) == 17


def test_col_to_coord():
    assert col_to_coord(10, 10, 5) ==  (0, 0)
    assert col_to_coord(11, 10, 5) ==  (0, 1)
    assert col_to_coord(15, 10, 5) ==  (1, 0)
    assert col_to_coord(17, 10, 5) ==  (1, 2)

