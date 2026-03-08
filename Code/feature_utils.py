import numpy as np
from collections import deque

# Global distance cache: key = (r1, c1, r2, c2) value = shortest path length in steps
_DISTANCE_CACHE = {}


def _obs_to_pixel(map_img, x_obs, y_obs):
    Nrow, Ncol = map_img.shape[:2]
    col = int(round(x_obs * Ncol))
    row = int(round(y_obs * Ncol))
    col = max(0, min(Ncol - 1, col))
    row = max(0, min(Nrow - 1, row))
    return row, col


def _is_white(map_img, r, c):
    # Check if pixel (r, c) is traversable (white = road)
    if r < 0 or r >= map_img.shape[0] or c < 0 or c >= map_img.shape[1]:
        return False
    val = np.asarray(map_img[r, c])
    return float(val.flat[0]) > 0.5


def compute_shortest_distance(map_img, x1, y1, x2, y2):
    global _DISTANCE_CACHE
    Nrow, Ncol = map_img.shape[0], map_img.shape[1]
    r1, c1 = _obs_to_pixel(map_img, x1, y1)
    r2, c2 = _obs_to_pixel(map_img, x2, y2)
    # Cache key (symmetric: (r1,c1,r2,c2) and (r2,c2,r1,c1) same distance)
    key = (min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2))

    if key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[key]
    map_2d = np.asarray(map_img)

    if map_2d.ndim > 2:
        map_2d = map_2d[:, :, 0] if map_2d.shape[2] >= 1 else map_2d[:, :, 0]
    white = (map_2d.astype(float) > 0.5)

    if not white[r1, c1] or not white[r2, c2]:
        _DISTANCE_CACHE[key] = np.inf
        return np.inf

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    queue = deque([(r1, c1)])
    visited = np.zeros((Nrow, Ncol), dtype=bool)
    visited[r1, c1] = True
    steps = np.zeros((Nrow, Ncol), dtype=float)
    steps[r1, c1] = 0
    found = False

    while queue:
        r, c = queue.popleft()
        if (r, c) == (r2, c2):
            found = True
            break
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < Nrow and 0 <= nc < Ncol and white[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                steps[nr, nc] = steps[r, c] + 1
                queue.append((nr, nc))

    dist = float(steps[r2, c2]) if found else np.inf
    _DISTANCE_CACHE[key] = dist
    return dist


def extract_features(context, map_img, max_drivers=10):
    c_passenger, c_drivers = context
    c_passenger = np.asarray(c_passenger).ravel()
    x_orig, y_orig = float(c_passenger[0]), float(c_passenger[1])
    x_dest, y_dest = float(c_passenger[2]), float(c_passenger[3])
    passenger_alpha = float(c_passenger[4])
    Nrow, Ncol = map_img.shape[0], map_img.shape[1]
    map_diagonal = np.sqrt(float(Nrow ** 2 + Ncol ** 2))
    if map_diagonal <= 0:
        map_diagonal = 1.0
    # Trip distance
    trip_dist_steps = compute_shortest_distance(map_img, x_orig, y_orig, x_dest, y_dest)
    trip_dist = trip_dist_steps / map_diagonal
    if not np.isfinite(trip_dist):
        trip_dist = 0.0
    # Driver distances and alphas
    driver_dists = []
    driver_alphas = []
    if c_drivers is not None and len(c_drivers) > 0:
        for d in c_drivers:
            d = np.asarray(d).ravel()
            if len(d) >= 3:
                x_d, y_d, alpha_d = float(d[0]), float(d[1]), float(d[2])
                dist_steps = compute_shortest_distance(map_img, x_d, y_d, x_orig, y_orig)
                if np.isfinite(dist_steps):
                    driver_dists.append(dist_steps / map_diagonal)
                driver_alphas.append(alpha_d)

    if not driver_dists:
        min_driver_dist = 0.0
        mean_driver_dist = 0.0
    else:
        min_driver_dist = float(np.min(driver_dists))
        mean_driver_dist = float(np.mean(driver_dists))

    if not driver_alphas:
        mean_driver_alpha = 0.0
        min_driver_alpha = 0.0
    else:
        mean_driver_alpha = float(np.mean(driver_alphas))
        min_driver_alpha = float(np.min(driver_alphas))

    num_drivers = len(driver_alphas) if driver_alphas else 0
    num_drivers_normalized = num_drivers / max(max_drivers, 1)

    # Avoid negative or NaN from bad data
    passenger_alpha = max(0.0, passenger_alpha)
    mean_driver_alpha = max(0.0, mean_driver_alpha)
    min_driver_alpha = max(0.0, min_driver_alpha)

    phi = np.array([
        trip_dist,
        min_driver_dist,
        mean_driver_dist,
        passenger_alpha,
        mean_driver_alpha,
        min_driver_alpha,
        num_drivers_normalized
    ], dtype=np.float64)
    return phi


def clear_distance_cache():
    global _DISTANCE_CACHE
    _DISTANCE_CACHE = {}