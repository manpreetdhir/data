
import os
import sys
import math
import time
from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from collections import deque

# -------------------------------
# Config (adjust if needed)
# -------------------------------
GRID_SIZE = 500            # expected grid size (rows = cols = 500)
WINDOW = 5                 # 5x5 site
MAX_SLOPE = 22.0           # slope threshold between adjacent cells (assumed as |dz| in meters per cell)
MOVE_8 = False             # use 4-neighborhood by default; set True for 8-neighborhood (assumption not specified)
TOP_K_HABITAT = 40         # number of top habitat candidates to consider
TOP_K_MINING = 40          # number of top mining candidates to consider
PENALTY_PER_CELL = 0.001   # path length penalty
RANDOM_SEED = 42           # for deterministic tie-breaking

# Notes on assumptions:
# - "Adjacent cells" interpreted as 4-neighborhood (N,E,S,W). Set MOVE_8=True to allow diagonals.
# - Slope constraint interpreted as |elev(next) - elev(curr)| <= MAX_SLOPE (in meters).
# - Site "Coordinates (row, col)" reported as the TOP-LEFT of the 5x5 site (0-indexed in code; converted to 0-index in output).
#   If you want 1-indexed for human readability, flip the flag below.
OUTPUT_ONE_INDEXED = False

# -------------------------------
# IO helpers
# -------------------------------
def load_csv_grid(path: str) -> np.ndarray:
    """Load a 2D CSV as float32 numpy array, robust to weird headers."""
    df = pd.read_csv(path)
    arr = df.to_numpy(dtype=np.float32)
    # If the CSV is 500x500 but ended up as (499,500) because of header row used as data,
    # try reading with header=None
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        try:
            df2 = pd.read_csv(path, header=None)
            arr2 = df2.to_numpy(dtype=np.float32)
            if arr2.shape == (GRID_SIZE, GRID_SIZE):
                return arr2
        except Exception:
            pass
    return arr

def save_result(
    score: float,
    habitat_top_left: Tuple[int, int],
    habitat_illum_avg: float,
    habitat_roughness: float,
    mining_top_left: Tuple[int, int],
    mining_water_avg: float,
    mining_roughness: float,
    path_len_cells: int,
    out_path: str = "result.txt",
) -> None:
    r_h, c_h = habitat_top_left
    r_m, c_m = mining_top_left

    if OUTPUT_ONE_INDEXED:
        r_h, c_h = r_h + 1, c_h + 1
        r_m, c_m = r_m + 1, c_m + 1

    contents = f"""Optimal Pair Found with Combined Score: {score:.4f}
--- Optimal Habitat Site ---
> Coordinates (row, col): ({r_h}, {c_h})
> Avg Illumination: {habitat_illum_avg*100:.2f}%
> Terrain Roughness (Std Dev): {habitat_roughness:.4f} m
--- Optimal Mining Site ---
> Coordinates (row, col): ({r_m}, {c_m})
> Avg Water-Ice Probability: {mining_water_avg:.4f}
> Terrain Roughness (Std Dev): {mining_roughness:.4f} m
--- Power Cable Path ---
> Path Length: {path_len_cells} cells ({path_len_cells*100} m)
"""
    with open(out_path, "w") as f:
        f.write(contents)

# -------------------------------
# Windowed statistics via integral images
# -------------------------------
def integral_image(a: np.ndarray) -> np.ndarray:
    """2D integral (cumulative) image with zero padding convention."""
    return np.pad(a, ((1,0),(1,0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

def window_sum(integral: np.ndarray, r0: int, c0: int, h: int, w: int) -> float:
    """Sum of a[h,w] window whose top-left is (r0,c0)."""
    r1, c1 = r0 + h, c0 + w
    return integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]

def rolling_mean_and_std(
    data: np.ndarray, window: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rolling WINDOWxWINDOW mean and std for the entire grid (valid positions only)."""
    H, W = data.shape
    assert window <= H and window <= W
    # means
    integ = integral_image(data)
    means = np.empty((H - window + 1, W - window + 1), dtype=np.float32)
    # variances via E[x^2] - (E[x])^2
    sq = data**2
    integ_sq = integral_image(sq)
    variances = np.empty_like(means)

    win_area = window * window
    for r in range(H - window + 1):
        r1 = r + window
        for c in range(W - window + 1):
            c1 = c + window
            s = integ[r1, c1] - integ[r, c1] - integ[r1, c] - (-integ[r, c])
            # Equivalent clearer call:
            # s = window_sum(integ, r, c, window, window)
            # But keep inline for speed
            ss = integ_sq[r1, c1] - integ_sq[r, c1] - integ_sq[r1, c] - (-integ_sq[r, c])
            m = s / win_area
            means[r, c] = m
            variances[r, c] = max(0.0, ss / win_area - m * m)

    stds = np.sqrt(variances, dtype=np.float32)
    return means, stds

# -------------------------------
# Candidate selection
# -------------------------------
def non_overlapping_top_k(
    score_map: np.ndarray,
    roughness_map: np.ndarray,
    k: int,
    window: int,
    prefer_low_roughness: bool = True,
) -> List[Tuple[int, int]]:
    """Greedy selection of top-k WINDOWxWINDOW sites, avoiding overlap.
    score_map and roughness_map are defined on top-left coordinates of valid windows.
    """
    H, W = score_map.shape
    idxs = [(score_map[r,c], roughness_map[r,c], r, c) for r in range(H) for c in range(W)]
    # primary: high score; secondary: low roughness
    idxs.sort(key=lambda x: (-x[0], x[1] if prefer_low_roughness else 0.0))
    chosen: List[Tuple[int,int]] = []
    occupied = np.zeros((H, W), dtype=bool)  # marking top-left slots inside a WINDOW area

    for _, _, r, c in idxs:
        if len(chosen) >= k:
            break
        # Check overlap: windows shouldn't overlap with already chosen ones
        overlap = False
        for (rr, cc) in chosen:
            if not (r + window <= rr or rr + window <= r or c + window <= cc or cc + window <= c):
                overlap = True
                break
        if not overlap:
            chosen.append((r, c))
    return chosen

# -------------------------------
# Pathfinding (multi-source BFS under slope constraint)
# -------------------------------
def neighbors(r: int, c: int, H: int, W: int, move8: bool = False):
    if move8:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    yield rr, cc
    else:
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

def shortest_path_len_under_slope(
    elev: np.ndarray,
    src_cells: List[Tuple[int,int]],
    dst_mask: np.ndarray,
    max_slope: float,
    move8: bool = False
) -> Optional[int]:
    """Return shortest path length in cells, or None if no path exists."""
    H, W = elev.shape
    q = deque()
    dist = -np.ones((H, W), dtype=np.int32)

    for (r, c) in src_cells:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        if dst_mask[r, c]:
            return int(dist[r, c])
        z = elev[r, c]
        dnext = dist[r, c] + 1
        for rr, cc in neighbors(r, c, H, W, move8):
            if dist[rr, cc] != -1:
                continue
            if abs(elev[rr, cc] - z) <= max_slope:
                dist[rr, cc] = dnext
                q.append((rr, cc))
    return None

# -------------------------------
# Scoring
# -------------------------------
def combined_score(illum_avg: float, water_avg: float, path_len: int) -> float:
    # Normalize illumination from [0,100] to [0,1] if needed
    illum_norm = illum_avg / 100.0 if illum_avg > 1.0 else illum_avg
    return 0.5 * illum_norm + 0.5 * water_avg - PENALTY_PER_CELL * float(path_len)

# -------------------------------
# Main solver
# -------------------------------
def solve(
    data_dir: str = ".",
    elevation_file: str = "elevation.csv",
    illumination_file: str = "illumination.csv",
    water_file: str = "water_ice.csv",
    signal_file: str = "signal_occultation.csv",
    out_path: str = "result.txt",
    verbose: bool = True,
) -> Dict:
    np.random.seed(RANDOM_SEED)

    elev_path = os.path.join(data_dir, elevation_file)
    illum_path = os.path.join(data_dir, illumination_file)
    water_path = os.path.join(data_dir, water_file)
    signal_path = os.path.join(data_dir, signal_file)

    if not os.path.exists(elev_path):
        raise FileNotFoundError(f"Missing {elevation_file} in {data_dir}")
    if not os.path.exists(illum_path) or not os.path.exists(water_path):
        if verbose:
            print("Warning: Missing illumination or water data. The solver needs both to compute the final answer.")
        # We still proceed to compute rolling stats for what we have.

    elev = load_csv_grid(elev_path)
    if elev.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"elevation grid shape {elev.shape} != {(GRID_SIZE, GRID_SIZE)}")

    # Load optional grids
    illum = load_csv_grid(illum_path) if os.path.exists(illum_path) else None
    water = load_csv_grid(water_path) if os.path.exists(water_path) else None
    signal = load_csv_grid(signal_path) if os.path.exists(signal_path) else None

    # Compute rolling stats
    if verbose: print("Computing rolling 5x5 stats (this can take ~10-20s without SciPy)...")
    hab_score_map = None
    mining_score_map = None

    # Roughness = std of elevation within the window
    _, elev_std = rolling_mean_and_std(elev, WINDOW)

    if illum is not None:
        illum_mean, _ = rolling_mean_and_std(illum, WINDOW)
        hab_score_map = illum_mean.copy()  # higher is better
    else:
        illum_mean = None

    if water is not None:
        water_mean, _ = rolling_mean_and_std(water, WINDOW)
        mining_score_map = water_mean.copy()  # higher is better
    else:
        water_mean = None

    # Candidate lists
    if hab_score_map is not None:
        habitat_candidates = non_overlapping_top_k(hab_score_map, elev_std, TOP_K_HABITAT, WINDOW, True)
    else:
        habitat_candidates = []

    if mining_score_map is not None:
        mining_candidates = non_overlapping_top_k(mining_score_map, elev_std, TOP_K_MINING, WINDOW, True)
    else:
        mining_candidates = []

    if verbose:
        print(f"Habitat candidates: {len(habitat_candidates)}; Mining candidates: {len(mining_candidates)}")

    # Prepare site cell sets (all 25 cells within each 5x5) for pathfinding
    def site_cells(top_left: Tuple[int,int]) -> List[Tuple[int,int]]:
        r0, c0 = top_left
        return [(r0 + dr, c0 + dc) for dr in range(WINDOW) for dc in range(WINDOW)]

    best = {
        "score": -1e9,
        "habitat": None,
        "mining": None,
        "hab_illum_avg": None,
        "hab_rough": None,
        "min_water_avg": None,
        "min_rough": None,
        "path_len": None,
    }

    # Masks for BFS targets for speed; we'll rebuild per mining candidate
    H, W = elev.shape

    if illum_mean is None or water_mean is None:
        if verbose:
            print("Cannot compute final combined score without illumination and water grids.")
        # Provide partial outputs
        return {
            "partial": True,
            "habitat_candidates": habitat_candidates,
            "mining_candidates": mining_candidates,
        }

    # Evaluate pairs
    t0 = time.time()
    for (rh, ch) in habitat_candidates:
        srcs = site_cells((rh, ch))

        for (rm, cm) in mining_candidates:
            dst_mask = np.zeros((H, W), dtype=bool)
            for (r, c) in site_cells((rm, cm)):
                dst_mask[r, c] = True

            path_len = shortest_path_len_under_slope(elev, srcs, dst_mask, MAX_SLOPE, move8=MOVE_8)
            if path_len is None:
                continue

            hab_illum_avg = float(illum_mean[rh, ch])
            hab_rough = float(elev_std[rh, ch])
            min_water_avg = float(water_mean[rm, cm])
            min_rough = float(elev_std[rm, cm])

            score = combined_score(hab_illum_avg, min_water_avg, path_len)

            if score > best["score"]:
                best.update({
                    "score": score,
                    "habitat": (rh, ch),
                    "mining": (rm, cm),
                    "hab_illum_avg": hab_illum_avg,
                    "hab_rough": hab_rough,
                    "min_water_avg": min_water_avg,
                    "min_rough": min_rough,
                    "path_len": int(path_len),
                })

    if best["habitat"] is None:
        if verbose:
            print("No valid path satisfying slope constraint was found among candidate pairs.")
        return {
            "partial": False,
            "solution_found": False,
        }

    # Save result.txt
    save_result(
        best["score"],
        best["habitat"],
        best["hab_illum_avg"],
        best["hab_rough"],
        best["mining"],
        best["min_water_avg"],
        best["min_rough"],
        best["path_len"],
        out_path=out_path,
    )

    if verbose:
        print(f"Done in {time.time()-t0:.1f}s. Best score: {best['score']:.4f}, path {best['path_len']} cells.")
    best["partial"] = False
    best["solution_found"] = True
    return best

if __name__ == "__main__":
    data_dir = "."
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    res = solve(data_dir=data_dir, out_path="result.txt", verbose=True)
    # Print minimal summary so you can see it in stdout when running as a script
    print(res)
