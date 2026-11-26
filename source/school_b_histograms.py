"""A compact NumPy histogram regressor for the B residual continuation."""
from heapq import heappop, heappush

import numpy as np


def fit_histogram_boost(x, target, weights, *, stages=150, leaves=12, min_leaf=45, rate=.04):
    """Fit squared-error residual increments; all thresholds come from fitting rows."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or len(target) != len(x) or len(weights) != len(x):
        raise ValueError("invalid B residual fitting shapes")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(target)) and np.all(np.isfinite(weights) & (weights > 0))):
        raise ValueError("invalid B residual fitting values")
    quantiles = np.linspace(0, 1, 41)[1:-1]
    cutoffs = [np.unique(np.quantile(col, quantiles)) for col in x.T]
    bins = np.array([np.searchsorted(edges, column, side="right") for edges, column in zip(cutoffs, x.T, strict=True)])
    anchor = float(np.average(target, weights=weights))
    predicted = np.full(len(x), anchor)
    trees = []

    def leaf_candidate(indices, residual):
        if len(indices) < 2 * min_leaf:
            return None
        w = weights[indices]
        r = w * residual[indices]
        bias, total = r.sum(), w.sum()
        old_error = bias ** 2 / total
        best = None
        for i, edges in enumerate(cutoffs):
            if not len(edges):
                continue
            trace = bins[i, indices]
            k = len(edges) + 1
            size = np.cumsum(np.bincount(trace, minlength=k))[:-1]
            fit_w = np.cumsum(np.bincount(trace, weights=w, minlength=k))[:-1]
            fit_r = np.cumsum(np.bincount(trace, weights=r, minlength=k))[:-1]
            valid = np.flatnonzero((size >= min_leaf) & (size <= len(indices) - min_leaf))
            if not len(valid):
                continue
            gains = (fit_r[valid] ** 2 / fit_w[valid]
                     + (bias - fit_r[valid]) ** 2 / (total - fit_w[valid]) - old_error)
            first = int(gains.argmax())
            gain = float(gains[first])
            if gain > 1e-10 and (best is None or gain > best[0]):
                best = (gain, i, int(valid[first]))
        return best

    for _ in range(stages):
        residual = target - predicted
        nodes, active, queue = [None], {0: np.arange(len(x))}, []

        def enqueue(node_id):
            split = leaf_candidate(active[node_id], residual)
            if split is not None:
                gain, feature, cutoff = split
                heappush(queue, (-gain, node_id, feature, cutoff))

        enqueue(0)
        for _ in range(leaves - 1):
            if not queue:
                break
            _, node, feature, cutoff = heappop(queue)
            indices = active.pop(node)
            left_mask = bins[feature, indices] <= cutoff
            left, right = len(nodes), len(nodes) + 1
            nodes[node] = (feature, float(cutoffs[feature][cutoff]), left, right)
            nodes.extend((None, None))
            active[left], active[right] = indices[left_mask], indices[~left_mask]
            enqueue(left)
            enqueue(right)
        for key, indices in active.items():
            shift = rate * float(np.average(residual[indices], weights=weights[indices]))
            nodes[key] = (-1, shift, 0, 0)
            predicted[indices] += shift
        trees.append(nodes)
    return {"anchor": anchor, "trees": trees}


def predict_histogram_boost(model, x):
    if len(x) == 1:
        total = float(model["anchor"])
        row = x[0]
        for tree in model["trees"]:
            node = 0
            while tree[node][0] >= 0:
                feature, cutoff, left, right = tree[node]
                node = left if row[feature] < cutoff else right
            total += tree[node][1]
        return np.array([total])
    values = np.full(len(x), float(model["anchor"]))
    for tree in model["trees"]:
        def visit(node, members):
            feature, value, left, right = tree[node]
            if feature < 0:
                values[members] += value
                return
            choice = x[members, feature] < value
            if np.any(choice):
                visit(left, members[choice])
            if not np.all(choice):
                visit(right, members[~choice])

        visit(0, np.arange(len(x)))
    return values
