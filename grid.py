# grid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


Cell = tuple[int, int]


@dataclass(frozen=True)
class GridData:
    """
    Container for the simulated grid.

    Coordinates are 0-indexed in Python:
        (0, 0), ..., (L-1, L-1).

    The LaTeX writeup may use 1-indexed coordinates:
        (1, 1), ..., (L, L).
    """

    L: int
    is_parking: np.ndarray  # shape (L, L), bool
    all_cells: list[Cell]
    parking_cells: list[Cell]      # p_1, ..., p_K in the writeup, but 0-indexed in code
    destination_cells: list[Cell]  # g_1, ..., g_M in the writeup, but 0-indexed in code
    parking_index_by_cell: dict[Cell, int]
    destination_index_by_cell: dict[Cell, int]
    center: tuple[float, float]

    @property
    def K(self) -> int:
        """Number of parking cells."""
        return len(self.parking_cells)

    @property
    def M(self) -> int:
        """Number of destination/non-parking cells."""
        return len(self.destination_cells)

    def parking_cell(self, j: int) -> Cell:
        """Return parking cell p_j, using 0-indexed Python index j."""
        return self.parking_cells[j]

    def destination_cell(self, i: int) -> Cell:
        """Return destination cell g_i, using 0-indexed Python index i."""
        return self.destination_cells[i]


def all_grid_cells(L: int) -> list[Cell]:
    """Return all cells in an L x L grid."""
    return [(r, c) for r in range(L) for c in range(L)]


def manhattan(u: Cell | tuple[float, float], v: Cell | tuple[float, float]) -> float:
    """Manhattan distance between two cells/points."""
    return abs(u[0] - v[0]) + abs(u[1] - v[1])


def make_path(
    start: Cell,
    end: Cell,
    *,
    horizontal_first: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> list[Cell]:
    """
    Ordered sequence of cells traversed from start to end.

    The path excludes the starting cell and includes the ending cell.
    If start == end, returns [start], matching the convention in the writeup.

    Movement is Manhattan: one adjacent grid cell per step.

    If rng is provided, the required north/south and east/west moves are
    randomly permuted. If rng is not provided, the path is deterministic:
    horizontal moves first if horizontal_first=True, otherwise vertical moves first.
    """
    if start == end:
        return [start]

    r0, c0 = start
    r1, c1 = end

    row_step = 1 if r1 > r0 else -1
    col_step = 1 if c1 > c0 else -1

    row_moves = [("row", row_step)] * abs(r1 - r0)
    col_moves = [("col", col_step)] * abs(c1 - c0)

    if rng is not None:
        moves = row_moves + col_moves
        rng.shuffle(moves)
    else:
        if horizontal_first:
            moves = col_moves + row_moves
        else:
            moves = row_moves + col_moves

    path: list[Cell] = []
    cur_r, cur_c = r0, c0

    for move_type, step in moves:
        if move_type == "row":
            cur_r += step
        else:
            cur_c += step
        path.append((cur_r, cur_c))

    return path


def generate_grid(config: Any, rng: Optional[np.random.Generator] = None) -> GridData:
    """
    Generate parking and non-parking cells.

    Expected config fields:
        L: int
        parking_probability: float
        use_ring_parking: bool
        ring_specs: optional list of objects with
            inner_size, outer_size, parking_probability

    Returns:
        GridData with parking cells indexed as p_j and destination cells indexed as g_i.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    L = int(config.L)
    if L <= 0:
        raise ValueError("L must be positive.")

    prob_map = _parking_probability_map(config)
    is_parking = rng.random((L, L)) < prob_map

    # Ensure we do not accidentally get all-parking or no-parking grid.
    _ensure_valid_parking_split(is_parking, rng)

    all_cells = all_grid_cells(L)
    parking_cells = [u for u in all_cells if is_parking[u]]
    destination_cells = [u for u in all_cells if not is_parking[u]]

    parking_index_by_cell = {cell: j for j, cell in enumerate(parking_cells)}
    destination_index_by_cell = {cell: i for i, cell in enumerate(destination_cells)}

    center = ((L - 1) / 2.0, (L - 1) / 2.0)

    return GridData(
        L=L,
        is_parking=is_parking,
        all_cells=all_cells,
        parking_cells=parking_cells,
        destination_cells=destination_cells,
        parking_index_by_cell=parking_index_by_cell,
        destination_index_by_cell=destination_index_by_cell,
        center=center,
    )


def get_parking_cells(grid: GridData) -> list[Cell]:
    return grid.parking_cells


def get_destination_cells(grid: GridData) -> list[Cell]:
    return grid.destination_cells


def is_parking_cell(grid: GridData, cell: Cell) -> bool:
    return bool(grid.is_parking[cell])


def parking_index(grid: GridData, cell: Cell) -> int:
    return grid.parking_index_by_cell[cell]


def destination_index(grid: GridData, cell: Cell) -> int:
    return grid.destination_index_by_cell[cell]


def _parking_probability_map(config: Any) -> np.ndarray:
    """
    Return an L x L array of parking probabilities.

    If config.use_ring_parking is False, every cell has probability config.parking_probability.

    If config.use_ring_parking is True, probabilities are assigned using centered square rings.
    Example for L=30:
        inner_size=0,  outer_size=10 -> central 10 x 10 square
        inner_size=10, outer_size=20 -> surrounding 20 x 20 ring
        inner_size=20, outer_size=30 -> surrounding 30 x 30 ring
    """
    L = int(config.L)

    if not getattr(config, "use_ring_parking", False):
        return np.full((L, L), float(config.parking_probability))

    prob_map = np.full((L, L), float(config.parking_probability))

    for spec in config.ring_specs:
        inner = int(spec.inner_size)
        outer = int(spec.outer_size)
        p = float(spec.parking_probability)

        if inner < 0 or outer <= 0 or inner >= outer:
            raise ValueError(f"Invalid ring spec: inner_size={inner}, outer_size={outer}.")
        if outer > L:
            raise ValueError(f"Ring outer_size={outer} exceeds grid size L={L}.")
        if not (0.0 <= p <= 1.0):
            raise ValueError("Ring parking_probability must be in [0, 1].")

        outer_mask = _centered_square_mask(L, outer)
        inner_mask = _centered_square_mask(L, inner) if inner > 0 else np.zeros((L, L), dtype=bool)
        ring_mask = outer_mask & ~inner_mask
        prob_map[ring_mask] = p

    return prob_map


def _centered_square_mask(L: int, size: int) -> np.ndarray:
    """Boolean mask for a centered size x size square inside an L x L grid."""
    if size < 0 or size > L:
        raise ValueError(f"size must be between 0 and L; got size={size}, L={L}.")

    mask = np.zeros((L, L), dtype=bool)
    if size == 0:
        return mask

    start = (L - size) // 2
    end = start + size
    mask[start:end, start:end] = True
    return mask


def _ensure_valid_parking_split(is_parking: np.ndarray, rng: np.random.Generator) -> None:
    """
    Mutates is_parking if needed so there is at least one parking cell
    and at least one destination cell.
    """
    L = is_parking.shape[0]

    if is_parking.all():
        r = int(rng.integers(0, L))
        c = int(rng.integers(0, L))
        is_parking[r, c] = False

    if not is_parking.any():
        r = int(rng.integers(0, L))
        c = int(rng.integers(0, L))
        is_parking[r, c] = True


def grid_summary(grid: GridData) -> str:
    parking_share = grid.K / (grid.L * grid.L)
    return (
        f"GridData(L={grid.L}, K={grid.K}, M={grid.M}, "
        f"parking_share={parking_share:.3f})"
    )