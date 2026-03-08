import numpy as np
from collections import deque



"""
feature_utils.py exists because both lin_greedy.py and lin_ucb.py need the exact same feature extraction and BFS distance logic. Instead of copy-pasting the same code into both files, we put it in one shared utility file and just import it wherever needed.
pythonfrom feature_utils import extract_features
That's the only reason — avoid code duplication. If we ever needed to change how features are computed, we'd only have to change it in one place instead of two.
policy_gradient.py doesn't import it because PG uses Euclidean distance instead of BFS, so it has its own simpler extract_features_fast function defined directly inside the file.
"""


# Dictionary to store already-computed BFS distances so we don't recompute them
# Key is a tuple of pixel coordinates, value is the distance in steps
_DISTANCE_CACHE = {}


def _obs_to_pixel(map_img, x_obs, y_obs):
    # Get the number of rows and columns in the map image
    Nrow, Ncol = map_img.shape[:2]

    # Convert the observation coordinate (0 to 1 float) to a pixel column index
    col = int(round(x_obs * Ncol))

    # Convert the observation coordinate (0 to 1 float) to a pixel row index
    # Note: uses Ncol for both, which keeps the scaling consistent
    row = int(round(y_obs * Ncol))

    # Clamp col to stay within image bounds (0 to Ncol-1)
    col = max(0, min(Ncol - 1, col))

    # Clamp row to stay within image bounds (0 to Nrow-1)
    row = max(0, min(Nrow - 1, row))

    # Return the pixel (row, col) corresponding to the given observation coordinate
    return row, col


def _is_white(map_img, r, c):
    # Return False if the pixel is outside the image boundaries
    if r < 0 or r >= map_img.shape[0] or c < 0 or c >= map_img.shape[1]:
        return False

    # Get the pixel value at (r, c)
    val = np.asarray(map_img[r, c])

    # If the pixel value is greater than 0.5, it's white (road/traversable), else black (wall)
    return float(val.flat[0]) > 0.5


def compute_shortest_distance(map_img, x1, y1, x2, y2):
    global _DISTANCE_CACHE

    # Get map dimensions
    Nrow, Ncol = map_img.shape[0], map_img.shape[1]

    # Convert both observation coordinates to pixel coordinates
    r1, c1 = _obs_to_pixel(map_img, x1, y1)
    r2, c2 = _obs_to_pixel(map_img, x2, y2)

    # Build a symmetric cache key so (A->B) and (B->A) map to the same entry
    # This avoids computing the same path twice in opposite directions
    key = (min(r1, r2), min(c1, c2), max(r1, r2), max(c1, c2))

    # If we already computed this distance before, just return the cached result
    if key in _DISTANCE_CACHE:
        return _DISTANCE_CACHE[key]

    # Convert map image to numpy array for processing
    map_2d = np.asarray(map_img)

    # If the image has multiple channels (e.g. RGB), just take the first channel
    if map_2d.ndim > 2:
        map_2d = map_2d[:, :, 0] if map_2d.shape[2] >= 1 else map_2d[:, :, 0]

    # Create a boolean mask: True = white pixel (road), False = black pixel (wall)
    white = (map_2d.astype(float) > 0.5)

    # If either the start or end point falls on a wall, path is impossible — return infinity
    if not white[r1, c1] or not white[r2, c2]:
        _DISTANCE_CACHE[key] = np.inf
        return np.inf

    # BFS uses 4-directional movement: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Initialize the BFS queue with the starting pixel
    queue = deque([(r1, c1)])

    # Track which pixels have already been visited to avoid revisiting
    visited = np.zeros((Nrow, Ncol), dtype=bool)
    visited[r1, c1] = True

    # Track the number of steps taken to reach each pixel
    steps = np.zeros((Nrow, Ncol), dtype=float)
    steps[r1, c1] = 0

    # Flag to track whether we found the destination
    found = False

    # BFS loop — keeps expanding until queue is empty or destination is found
    while queue:
        # Pop the next pixel to process from the front of the queue
        r, c = queue.popleft()

        # If we reached the destination pixel, stop BFS
        if (r, c) == (r2, c2):
            found = True
            break

        # Check all 4 neighbours of the current pixel
        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            # Only visit the neighbour if it's within bounds, is a road pixel, and hasn't been visited
            if 0 <= nr < Nrow and 0 <= nc < Ncol and white[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True

                # Distance to neighbour is one more step than current pixel
                steps[nr, nc] = steps[r, c] + 1

                # Add neighbour to queue for further exploration
                queue.append((nr, nc))

    # If destination was reached, get the step count; otherwise return infinity
    dist = float(steps[r2, c2]) if found else np.inf

    # Store result in cache so we don't recompute this pair again
    _DISTANCE_CACHE[key] = dist
    return dist


def extract_features(context, map_img, max_drivers=10):
    # Unpack the context tuple into passenger info and driver info
    c_passenger, c_drivers = context

    # Flatten passenger array to 1D just in case it has extra dimensions
    c_passenger = np.asarray(c_passenger).ravel()

    # Extract passenger origin coordinates
    x_orig, y_orig = float(c_passenger[0]), float(c_passenger[1])

    # Extract passenger destination coordinates
    x_dest, y_dest = float(c_passenger[2]), float(c_passenger[3])

    # Extract passenger price sensitivity
    passenger_alpha = float(c_passenger[4])

    # Get map dimensions to compute the diagonal for normalisation
    Nrow, Ncol = map_img.shape[0], map_img.shape[1]

    # Compute the map diagonal — used to normalise all distances to [0, 1] range
    map_diagonal = np.sqrt(float(Nrow ** 2 + Ncol ** 2))

    # Fallback in case map_diagonal is somehow 0 (shouldn't happen)
    if map_diagonal <= 0:
        map_diagonal = 1.0

    # Compute BFS shortest path from passenger origin to destination
    trip_dist_steps = compute_shortest_distance(map_img, x_orig, y_orig, x_dest, y_dest)

    # Normalise trip distance by the map diagonal
    trip_dist = trip_dist_steps / map_diagonal

    # If BFS couldn't find a path (e.g. point on wall), default to 0
    if not np.isfinite(trip_dist):
        trip_dist = 0.0

    # Lists to collect distances and alphas for all nearby drivers
    driver_dists = []
    driver_alphas = []

    # Loop through each nearby driver in the context
    if c_drivers is not None and len(c_drivers) > 0:
        for d in c_drivers:
            # Flatten driver array to 1D
            d = np.asarray(d).ravel()

            # Only process if driver array has at least 3 values (x, y, alpha)
            if len(d) >= 3:
                x_d, y_d, alpha_d = float(d[0]), float(d[1]), float(d[2])

                # Compute BFS distance from this driver to the passenger origin
                dist_steps = compute_shortest_distance(map_img, x_d, y_d, x_orig, y_orig)

                # Only add to list if a valid path was found
                if np.isfinite(dist_steps):
                    driver_dists.append(dist_steps / map_diagonal)

                # Always collect driver alpha even if distance was invalid
                driver_alphas.append(alpha_d)

    # If no valid driver distances, default both to 0
    if not driver_dists:
        min_driver_dist = 0.0
        mean_driver_dist = 0.0
    else:
        # Closest driver distance
        min_driver_dist = float(np.min(driver_dists))
        # Average driver distance
        mean_driver_dist = float(np.mean(driver_dists))

    # If no driver alphas collected, default both to 0
    if not driver_alphas:
        mean_driver_alpha = 0.0
        min_driver_alpha = 0.0
    else:
        # Average driver price sensitivity
        mean_driver_alpha = float(np.mean(driver_alphas))
        # Most willing driver (lowest alpha = most eager to accept)
        min_driver_alpha = float(np.min(driver_alphas))

    # Count how many drivers were present this time step
    num_drivers = len(driver_alphas) if driver_alphas else 0

    # Normalise driver count by the maximum possible (10) to get a value in [0, 1]
    num_drivers_normalized = num_drivers / max(max_drivers, 1)

    # Clip any negative values caused by bad data (shouldn't happen but just in case)
    passenger_alpha = max(0.0, passenger_alpha)
    mean_driver_alpha = max(0.0, mean_driver_alpha)
    min_driver_alpha = max(0.0, min_driver_alpha)

    # Assemble the final 7-dimensional feature vector
    phi = np.array([
        trip_dist,           # Feature 1: normalised trip distance (origin to destination)
        min_driver_dist,     # Feature 2: normalised distance of closest driver
        mean_driver_dist,    # Feature 3: normalised mean distance of all drivers
        passenger_alpha,     # Feature 4: passenger price sensitivity
        mean_driver_alpha,   # Feature 5: mean driver price sensitivity
        min_driver_alpha,    # Feature 6: min driver price sensitivity (most willing driver)
        num_drivers_normalized  # Feature 7: supply proxy (normalised driver count)
    ], dtype=np.float64)

    return phi


def clear_distance_cache():
    global _DISTANCE_CACHE
    # Wipe the cache — useful if you want to reset between runs
    _DISTANCE_CACHE = {}