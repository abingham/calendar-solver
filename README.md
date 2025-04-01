=========================
Pentomino Calendar Solver
=========================

This is a pretty crude solver for the pentominoes calendar puzzle.

Installation
============

The preferred way is with [`uv`](https://docs.astral.sh/uv/):
```
uv sync
```

But of course `pip` can also work:
```
pip install .
```
But remember to manage your virtual environment if using `pip`.

Running the solver
==================

To solve the calendar:

1. Update the function `main()` in `source/calendar_solver/solver.py` so that the `BOARD` array matches the day's puzzle.
2. Run:
```
python -m calendar_solver.solver
```