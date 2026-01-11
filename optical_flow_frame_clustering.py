import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from pathlib import Path

# Resolve project root dynamically so it works regardless of where the repo is cloned
ROOT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = ROOT_DIR / "Accidents"

video_paths = sorted(VIDEO_DIR.glob("*.mp4"))


# How many clusters (representative frame groups) you want per video
N_CLUSTERS = 5

# How aggressively to downsample time (skip frames to reduce compute)
SKIP = 5

def normalize_to_360(arr):
    """Scale array to [0, 360] using max value in the array (avoid divide-by-zero)."""
    arr = np.asarray(arr, dtype=np.float32)
    m = arr.max()
    if m <= 1e-8:
        return arr  # all zeros
    return (arr / m) * 360.0

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if not ret or frame1 is None:
        cap.release()
        continue

    # Convert first frame to grayscale for optical flow
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    features = []        # each entry = 32-dim vector (16 mag bins + 16 ang bins)
    frame_indices = []   # which "time index" each feature corresponds to

    t = 0
    while True:
        # Skip frames to reduce redundancy / compute
        frame2 = None
        for _ in range(SKIP):
            ret, frame2 = cap.read()
            t += 1
            if not ret:
                frame2 = None
                break

        if frame2 is None:
            break  # end of video

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Dense optical flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Convert flow vectors to magnitude + angle per pixel
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Flatten to 1D and normalize to [0, 360] (your original idea)
        mag = normalize_to_360(mag.flatten())
        ang = normalize_to_360(ang.flatten())

        # Motion "signature" for the frame transition:
        # histogram of magnitude and angle (global summary of motion)
        mag_hist, _ = np.histogram(mag, bins=16, range=(0, 360))
        ang_hist, _ = np.histogram(ang, bins=16, range=(0, 360))

        vector = np.concatenate([mag_hist, ang_hist]).astype(np.float32)

        features.append(vector)
        frame_indices.append(t)  # index in the video stream

        # update previous frame
        prev_gray = next_gray

    cap.release()

    if len(features) < N_CLUSTERS:
        print(f"Not enough samples for clustering in {video_path}")
        continue

    X = np.vstack(features)  # shape: [num_samples, 32]

    # KMeans clustering of motion signatures
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0, n_init="auto")
    kmeans.fit(X)

    centers = kmeans.cluster_centers_

    # Pick the samples closest to each cluster center = representative frames
    closest, _ = pairwise_distances_argmin_min(centers, X)
    closest = np.sort(closest)

    keyframes = [frame_indices[idx] for idx in closest]
    print(video_path)
    print("Representative frame indices:", keyframes)

