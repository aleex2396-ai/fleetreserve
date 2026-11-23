"""A sparse linear programme built in blocks of variables and rows, solved through `lp.solve_lp`.

Variables are added in blocks (`variables`), each with its own bounds; rows are added in blocks of
coordinate entries (`rows`), each block a set of new rows `sum_j coefficient * x_j (<=, ==) rhs`.
Nothing is solved here except through `lp.solve_lp`, which carries the time limit.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from fleetreserve.lp import LpResult, solve_lp


@dataclass(frozen=True)
class Sizes:
    """The size of a programme as it was solved."""

    variables: int
    rows: int
    nonzeros: int


class SparseProgramme:
    """Blocks of variables and rows; `maximise` builds the sparse matrices and solves."""

    def __init__(self):
        self.n_variables = 0
        self._low: list[np.ndarray] = []
        self._high: list[np.ndarray] = []
        self._blocks = {"le": [], "eq": []}  # (row, column, value, rhs) per block
        self._n_rows = {"le": 0, "eq": 0}

    def variables(self, shape, low=-np.inf, high=np.inf) -> np.ndarray:
        """A new block of variables of `shape`; returns their indices in that shape."""
        count = int(np.prod(shape))
        self._low.append(np.broadcast_to(np.asarray(low, float), shape).reshape(-1).copy())
        self._high.append(np.broadcast_to(np.asarray(high, float), shape).reshape(-1).copy())
        index = np.arange(self.n_variables, self.n_variables + count).reshape(shape)
        self.n_variables += count
        return index

    def rows(self, sense: str, row, column, value, rhs) -> np.ndarray:
        """New rows: entry `(row[k], column[k], value[k])`, with `row` counted from 0 within this
        block and `rhs[r]` the right-hand side of row `r`. `sense` is "le" or "eq". Returns the new
        rows' indices among the rows of that sense."""
        rhs = np.asarray(rhs, float).reshape(-1)
        row = np.asarray(row, int).reshape(-1)
        column = np.asarray(column, int).reshape(-1)
        value = np.broadcast_to(np.asarray(value, float), row.shape).copy()
        if row.size and (row.min() < 0 or row.max() >= rhs.size):
            raise ValueError("a row index is outside the block")
        if column.size and (column.min() < 0 or column.max() >= self.n_variables):
            raise ValueError("a column index is not a variable")
        start = self._n_rows[sense]
        self._blocks[sense].append((row + start, column, value, rhs))
        self._n_rows[sense] += rhs.size
        return np.arange(start, start + rhs.size)

    def _matrix(self, sense):
        blocks = self._blocks[sense]
        if not blocks:
            return None, None
        row = np.concatenate([b[0] for b in blocks])
        column = np.concatenate([b[1] for b in blocks])
        value = np.concatenate([b[2] for b in blocks])
        rhs = np.concatenate([b[3] for b in blocks])
        matrix = sp.csr_matrix((value, (row, column)), shape=(rhs.size, self.n_variables))
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix, rhs

    def maximise(self, column, value, time_limit) -> tuple[LpResult, Sizes]:
        """Maximise `sum_k value[k] * x[column[k]]` with the interior-point method (crossover on).

        Returns the solver's result (its objective is the minimised negative) and the sizes."""
        cost = np.zeros(self.n_variables)
        np.add.at(cost, np.asarray(column, int).reshape(-1), -np.asarray(value, float).reshape(-1))
        a_ub, b_ub = self._matrix("le")
        a_eq, b_eq = self._matrix("eq")
        bounds = np.column_stack([np.concatenate(self._low), np.concatenate(self._high)])
        result = solve_lp(
            cost,
            A_ub=a_ub,
            b_ub=b_ub,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=bounds,
            time_limit=time_limit,
            method="ipm",
        )
        nonzeros = sum(0 if a is None else a.nnz for a in (a_ub, a_eq))
        rows = sum(0 if b is None else b.size for b in (b_ub, b_eq))
        return result, Sizes(self.n_variables, rows, nonzeros)
