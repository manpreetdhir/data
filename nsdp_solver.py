
#!/usr/bin/env python3
"""
NSpD 2025 — H5 group (Self coded)
Author: Manpreet , Rudraksh CET 2025-29

Usage:
    python nsdp_solver.py --data . --out result.txt --plots


Output:
    result.txt  — saved by cet students manpreet , rudraksh
    (optional) plots/*.png when --plots is passed
"""
import os
import sys
import argparse
from typing import Tuple, List, Optional, Dict
from collections import deque

import numpy as np
import pandas as pd


GRID_SIZE = 500
WINDOW = 5
MAX_SLOPE = 22.0         
MOVE_8 = False           
TOP_K_HABITAT = 60       
TOP_K_MINING = 60
PENALTY_PER_CELL = 0.001
OUTPUT_ONE_INDEXED = False
RANDOM_SEED = 42

# -------------------------------
# IO helpers
# -------------------------------
def load_csv_grid(path: str) -> np.ndarray:
    """Load a 500x500 CSV into float32 numpy array. Be robust to headers."""
    # Try header=None first (most robust for raw numeric matrices)
    try:
        arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
        if arr.shape == (GRID_SIZE, GRID_SIZE):
            return arr
    except Exception:
        pass
    # Fallback: default header detection
    df = pd.read_csv(path)
    arr2 = df.to_numpy(dtype=np.float32)
    if arr2.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"{os.path.basename(path)} has shape {arr2.shape}, expected {(GRID_SIZE, GRID_SIZE)}")
    return arr2

def ensure_exists(path: str, name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {name} at {path}")

def save_result(
    score: float,
    habitat_top_left: Tuple[int, int],
    habitat_illum_avg: float,
    habitat_roughness: float,
    mining_top_left: Tuple[int, int],
    mining_water_avg: float,
    mining_roughness: float,
    path_len_cells: int,
    out_path: str,
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
# Windowed stats via integral images
# -------------------------------
def integral_image(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((1,0),(1,0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

def rolling_mean_and_std(a: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    H, W = a.shape
    if window > H or window > W:
        raise ValueError("window larger than array")
    integ = integral_image(a)
    sq = a * a
    integ_sq = integral_image(sq)
    out_h = H - window + 1
    out_w = W - window + 1
    means = np.empty((out_h, out_w), dtype=np.float32)
    vars_ = np.empty_like(means)
    win_area = window * window
    # vectorized via sliding sums using integral images
    # build all corners
    r = np.arange(out_h)[:, None]
    c = np.arange(out_w)[None, :]
    r1 = r + window
    c1 = c + window
    S  = integ[r1, c1] - integ[r, c1] - integ[r1, c] + integ[r, c]
    SS = integ_sq[r1, c1] - integ_sq[r, c1] - integ_sq[r1, c] + integ_sq[r, c]
    means = S / win_area
    vars_ = SS / win_area - means * means
    np.maximum(vars_, 0.0, out=vars_)
    stds = np.sqrt(vars_, dtype=np.float32)
    return means.astype(np.float32), stds.astype(np.float32)

# -------------------------------
# Candidate selection
# -------------------------------
def non_overlapping_top_k(score_map: np.ndarray, roughness_map: np.ndarray, k: int, window: int) -> List[Tuple[int,int]]:
    H, W = score_map.shape
    flat = []
    for r in range(H):
        for c in range(W):
            flat.append((score_map[r,c], roughness_map[r,c], r, c))
    flat.sort(key=lambda x: (-x[0], x[1]))  # prefer high score, then low roughness
    chosen: List[Tuple[int,int]] = []
    for _, _, r, c in flat:
        if len(chosen) >= k:
            break
        overlap = False
        for rr, cc in chosen:
            if not (r + window <= rr or rr + window <= r or c + window <= cc or cc + window <= c):
                overlap = True
                break
        if not overlap:
            chosen.append((r, c))
    return chosen

# -------------------------------
# Pathfinding
# -------------------------------
def neighbors(r: int, c: int, H: int, W: int, move8: bool):
    if move8:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0: continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    yield rr, cc
    else:
        for dr, dc in ((-1,0), (1,0), (0,-1), (0,1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

def shortest_path_len_under_slope(elev: np.ndarray, src_cells: List[Tuple[int,int]], dst_mask: np.ndarray, max_slope: float, move8: bool) -> Optional[int]:
    H, W = elev.shape
    from collections import deque
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
    illum_norm = illum_avg / 100.0 if illum_avg > 1.0 else illum_avg
    return 0.5 * illum_norm + 0.5 * water_avg - PENALTY_PER_CELL * float(path_len)

# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="NSpD 2025 Full Integrated Solver")
    parser.add_argument("--data", default=".", help="Folder containing CSVs")
    parser.add_argument("--out", default="result.txt", help="Output result.txt path")
    parser.add_argument("--plots", action="store_true", help="Save helpful PNGs to ./plots")
    args = parser.parse_args()

    data_dir = args.data
    elev_path = os.path.join(data_dir, "elevation.csv")
    illum_path = os.path.join(data_dir, "illumination.csv")
    water_path = os.path.join(data_dir, "water_ice.csv")
    sig_path   = os.path.join(data_dir, "signal_occultation.csv")  # not used

    ensure_exists(elev_path, "elevation.csv")
    ensure_exists(illum_path, "illumination.csv")
    ensure_exists(water_path, "water_ice.csv")

    print("[1/6] Loading grids...")
    elev = load_csv_grid(elev_path)
    illum = load_csv_grid(illum_path)
    water = load_csv_grid(water_path)
    H, W = elev.shape

    print("[2/6] Computing window stats (5x5 mean/std)...")
    # roughness from elevation std
    _, elev_std = rolling_mean_and_std(elev, WINDOW)
    illum_mean, _ = rolling_mean_and_std(illum, WINDOW)
    water_mean, _ = rolling_mean_and_std(water, WINDOW)

    print("[3/6] Selecting top candidates (non-overlapping)...")
    habitat_cands = non_overlapping_top_k(illum_mean, elev_std, TOP_K_HABITAT, WINDOW)
    mining_cands  = non_overlapping_top_k(water_mean, elev_std, TOP_K_MINING, WINDOW)
    if not habitat_cands or not mining_cands:
        raise RuntimeError("Failed to produce candidates. Check input files.")

    def site_cells(tl: Tuple[int,int]) -> List[Tuple[int,int]]:
        r0, c0 = tl
        return [(r0 + dr, c0 + dc) for dr in range(WINDOW) for dc in range(WINDOW)]

    print("[4/6] Evaluating pairs with slope-constrained BFS (this is the heavy part)...")
    best = dict(score=-1e9, habitat=None, mining=None, hab_illum=None, hab_rough=None, min_water=None, min_rough=None, path=None)
    for (rh, ch) in habitat_cands:
        srcs = site_cells((rh, ch))
        for (rm, cm) in mining_cands:
            dst_mask = np.zeros((H, W), dtype=bool)
            for (r, c) in site_cells((rm, cm)): dst_mask[r, c] = True
            pl = shortest_path_len_under_slope(elev, srcs, dst_mask, MAX_SLOPE, MOVE_8)
            if pl is None: 
                continue
            hab_illum = float(illum_mean[rh, ch])
            hab_rough = float(elev_std[rh, ch])
            min_water = float(water_mean[rm, cm])
            min_rough = float(elev_std[rm, cm])
            score = combined_score(hab_illum, min_water, pl)
            if score > best["score"]:
                best.update(dict(score=score, habitat=(rh, ch), mining=(rm, cm),
                                 hab_illum=hab_illum, hab_rough=hab_rough,
                                 min_water=min_water, min_rough=min_rough, path=pl))

    if best["habitat"] is None:
        raise RuntimeError("No valid path found that respects MAX_SLOPE. Consider increasing MAX_SLOPE or enabling MOVE_8.")

    print("[5/6] Writing result.txt...")
    save_result(best["score"], best["habitat"], best["hab_illum"], best["hab_rough"],
                best["mining"], best["min_water"], best["min_rough"], int(best["path"]), args.out)

    print(f"Done. Best score {best['score']:.4f}; path length {best['path']} cells;")
    print(f"Habitat TL {best['habitat']}, Mining TL {best['mining']} -> result saved to {args.out}")

    if args.plots:
        os.makedirs("plots", exist_ok=True)
        # Quick heatmaps (no special styling to keep dependencies minimal)
        import matplotlib.pyplot as plt
        def hm(data, title, fname):
            plt.imshow(data)
            plt.title(title)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join("plots", fname), dpi=160)
            plt.close()
        hm(elev, "Elevation", "elevation.png")
        hm(illum, "Illumination", "illumination.png")
        hm(water, "Water-Ice Probability", "water.png")
        hm(illum_mean, "Illumination 5x5 Mean", "illum_mean.png")
        hm(water_mean, "Water 5x5 Mean", "water_mean.png")
        hm(elev_std, "Elevation 5x5 Std (Roughness)", "elev_std.png")
        print("[6/6] Saved plots to ./plots")

if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    main()

