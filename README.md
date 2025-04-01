# Pentomino Calendar Solver

This is a pretty crude solver for the pentominoes calendar puzzle.

## Installation

The preferred way is with [`uv`](https://docs.astral.sh/uv/):
```
uv sync
```

But of course `pip` can also work:
```
pip install .
```
But remember to manage your virtual environment if using `pip`.

## Running the solver

To solve the calendar:

1. Update the function `main()` in `source/calendar_solver/solver.py` so that the `BOARD` array matches the day's puzzle.
2. Run the `calendar-solver` command:
```
calendar-solver
```

Search space
============

It seems to be something like 5.8e24 (septillion) combinations of piece, orientation, and placement.
5_868_249_732_031_622_747_783_168