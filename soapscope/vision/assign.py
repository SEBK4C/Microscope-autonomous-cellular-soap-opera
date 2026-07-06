"""Optimal linear-sum (Hungarian) assignment — pure numpy, no scipy.

``linear_sum_assignment(cost)`` returns row/column index arrays minimising total
cost, matching ``scipy.optimize.linear_sum_assignment`` for rectangular inputs
(it assigns every row when rows <= cols, else every column). Used by the tracker
to match tracks to detections optimally (fewer ID switches on crossings) and by
the MOTA metric to match tracks to ground truth.

Implementation: the O(n^2 m) Jonker–Volgenant shortest-augmenting-path method.
Sizes here are tiny (a handful of microbes), so this is plenty fast.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_INF = 1e18


def linear_sum_assignment(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cost = np.asarray(cost, dtype=float)
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape                       # n <= m: assign every row
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=int)          # p[j] = row (1-indexed) assigned to column j
    way = np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, _INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = _INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j] > 0:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    if transposed:
        return cols, rows
    return rows, cols
