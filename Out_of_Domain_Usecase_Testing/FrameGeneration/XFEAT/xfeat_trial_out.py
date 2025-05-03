# interpolate_video_xfeat.py
# Frame interpolation using XFeat for feature matching.

import cv2 as cv
import numpy as np
import torch
import time
import sys
import math
import argparse
from typing import List, Tuple, Iterable, Optional, Dict

# print("Python Path:", sys.path)
# sys.path.append('/home2/raghuv_aditya/swiftframes/accelerated_features')
# print("Python Path:", sys.path)
# import xfeat
# --- Try Importing XFeat ---
try:
    from accelerated_features.modules.xfeat import XFeat 
except ImportError:
    print("Error: XFeat module not found.")
    print("Please ensure the XFeat code (https://github.com/verlab/accelerated_features)")
    print("is cloned and the 'modules' directory is in your Python path, or install it.")
    sys.exit(1)

# --- Class Definition (from motion_feature.py - slightly adapted) ---

class MotionFeature:
    """Represents the estimated motion and appearance of a feature point."""
    def __init__(self, pt: np.ndarray, query_pt: Optional[np.ndarray], train_pt: Optional[np.ndarray], size: float, quality: float):
        self.pt = pt # Interpolated point position (midpoint)
        self.query_pt = query_pt # Corresponding point in the query (earlier) frame
        self.train_pt = train_pt # Corresponding point in the train (later) frame
        self.size = size # Interpolated feature size (often arbitrary for methods like XFeat)
        self.quality = quality # Confidence/Quality of this motion feature

    def sq_dist(self, pos: np.ndarray) -> float:
        """Calculates squared distance from this feature's point to a given position."""
        diff = pos - self.pt
        return np.dot(diff, diff)

    def query_at(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """Estimates where the pixel at 'pos' came from in the query frame."""
        if self.query_pt is None:
            return None
        return pos + self.query_pt - self.pt

    def query_pixel_at(self, pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Gets the estimated integer pixel coordinates in the query frame."""
        pix = self.query_at(pos)
        return (int(round(pix[0])), int(round(pix[1]))) if pix is not None else None

    def train_at(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """Estimates where the pixel at 'pos' will go in the train frame."""
        if self.train_pt is None:
            return None
        return pos + self.train_pt - self.pt

    def train_pixel_at(self, pos: np.ndarray) -> Optional[Tuple[int, int]]:
        """Gets the estimated integer pixel coordinates in the train frame."""
        pix = self.train_at(pos)
        return (int(round(pix[0])), int(round(pix[1]))) if pix is not None else None

# --- Video I/O Functions (Simplified) ---

def read_video_frames(fileName: str) -> Tuple[Iterable[np.ndarray], Tuple[int, int], float, int]:
    """Reads frames from a video file."""
    cap = cv.VideoCapture(fileName)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {fileName}")
    fps = cap.get(cv.CAP_PROP_FPS)
    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    print(f"Input Video: {width}x{height} @ {fps:.2f} FPS, {length} frames")
    def frame_generator():
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            yield frame
            count += 1
            print(f"\rReading input frame: {count}/{length}", end="")
        print("\nFinished reading input.")
        cap.release()
    return frame_generator(), (height, width), fps, length

def write_video_frames(fileName: str, shape: Tuple[int, int], frames: Iterable[np.ndarray], fps: float, total_frames: int) -> None:
    """Writes frames to a video file."""
    height, width = shape
    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Use mp4v for better compatibility
    out = cv.VideoWriter(fileName, fourcc, fps, (width, height))
    if not out.isOpened(): raise IOError(f"Cannot open video writer for: {fileName}")
    print(f"\nWriting Output Video: {fileName} ({width}x{height} @ {fps:.2f} FPS)")
    frame_count = 0
    start_time = time.time()
    for frame in frames:
        if frame.shape[0] != height or frame.shape[1] != width: # Ensure correct size
             frame = cv.resize(frame, (width, height))
        out.write(frame)
        frame_count += 1
        elapsed = time.time() - start_time
        eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
        print(f"\rWriting output frame: {frame_count}/{total_frames}, Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s", end="")
    print(f"\nFinished writing {frame_count} frames.")
    out.release()

# --- XFeat Motion Feature Generation ---

def numpy_to_torch(img: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert NumPy image (H, W, C) BGR to Torch tensor (1, C, H, W) RGB / 255.0"""
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

def get_xfeat_motion_features(
    xfeat_model: XFeat,
    feats0: Dict[str, torch.Tensor],
    feats1: Dict[str, torch.Tensor],
    match_threshold: float = 0.82 # Default from XFeat demo
) -> List[MotionFeature]:
    """Generates MotionFeature list from two frames' XFeat features."""

    motion_features = []
    if feats0 is None or feats1 is None or 'keypoints' not in feats0 or 'keypoints' not in feats1:
        return motion_features # Cannot proceed if features are missing

    kpts0, descs0 = feats0['keypoints'], feats0['descriptors']
    kpts1, descs1 = feats1['keypoints'], feats1['descriptors']

    if kpts0.shape[0] == 0 or kpts1.shape[0] == 0:
        return motion_features # No keypoints found in one or both frames

    # Match descriptors using XFeat's built-in matcher
    with torch.no_grad():
        idx0, idx1 = xfeat_model.match(descs0, descs1, match_threshold)

    matched_kpts0 = kpts0[idx0].cpu().numpy()
    matched_kpts1 = kpts1[idx1].cpu().numpy()
    matched_descs0 = descs0[idx0]
    matched_descs1 = descs1[idx1]

    # Calculate L2 distance between matched descriptors as a quality proxy
    # Normalize descriptors first for stable distance calculation
    matched_descs0_norm = torch.nn.functional.normalize(matched_descs0, p=2, dim=-1)
    matched_descs1_norm = torch.nn.functional.normalize(matched_descs1, p=2, dim=-1)
    distances = torch.norm(matched_descs0_norm - matched_descs1_norm, p=2, dim=-1).cpu().numpy()

    # Convert distance to quality (lower distance -> higher quality)
    # Simple inversion - might need tuning. Clamp distance to avoid division by near zero.
    max_dist_for_quality = 1.0 # Assume distances are roughly in [0, sqrt(2)] after normalization
    qualities = 1.0 / (1.0 + np.maximum(distances, 1e-6) / max_dist_for_quality)

    # Create MotionFeature objects
    for i in range(len(matched_kpts0)):
        pt0 = matched_kpts0[i]
        pt1 = matched_kpts1[i]
        quality = qualities[i]
        # Interpolate position (midpoint)
        pt_interp = (pt0 + pt1) / 2.0
        # XFeat doesn't provide size, use a default small value
        size = 5.0
        motion_features.append(MotionFeature(pt=pt_interp, query_pt=pt0, train_pt=pt1, size=size, quality=quality))

    return motion_features

# --- Interpolation Function (Adapted from previous script) ---

def gen_inter_frame(img_query: np.ndarray, img_train: np.ndarray, motionFeatures: List[MotionFeature]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generates an intermediate frame using motion features."""
    height, width = img_query.shape[:2]
    outImg = np.zeros_like(img_query, dtype=np.float32) # Use float for accumulation
    weightSum = np.zeros((height, width, 1), dtype=np.float32) + 1e-6 # Avoid division by zero

    if not motionFeatures:
        # Fallback: Simple 50/50 blend if no features exist
        return cv.addWeighted(img_query, 0.5, img_train, 0.5, 0.0), None

    # --- Feature Weighting and Blending ---
    print(f"\n  Interpolating frame based on {len(motionFeatures)} motion features...", end="")
    start_interp_time = time.time()

    # --- Optional: Grid-based acceleration (same as before) ---
    use_grid = True # Set to False to disable grid acceleration
    grid_size = 32
    feature_grid = defaultdict(list)
    if use_grid:
        grid_cols = math.ceil(width / grid_size)
        grid_rows = math.ceil(height / grid_size)
        for mf in motionFeatures:
            col = int(mf.pt[0] / grid_size)
            row = int(mf.pt[1] / grid_size)
            # Clamp grid indices
            col = max(0, min(col, grid_cols - 1))
            row = max(0, min(row, grid_rows - 1))
            feature_grid[(row, col)].append(mf)
        search_radius = 1
    # --- End Grid ---

    for y in range(height):
        for x in range(width):
            pos = np.array([x, y], dtype=np.float32)
            target_pixel_contributors = []

            if use_grid:
                # Find nearby features using the grid
                center_col = int(x / grid_size)
                center_row = int(y / grid_size)
                for r_offset in range(-search_radius, search_radius + 1):
                    for c_offset in range(-search_radius, search_radius + 1):
                        grid_r, grid_c = center_row + r_offset, center_col + c_offset
                        if 0 <= grid_r < grid_rows and 0 <= grid_c < grid_cols:
                            target_pixel_contributors.extend(feature_grid[(grid_r, grid_c)])
                if not target_pixel_contributors:
                     target_pixel_contributors = motionFeatures # Fallback to all if grid cell empty
            else:
                 target_pixel_contributors = motionFeatures # Use all features if grid is disabled

            # Calculate weights and sample pixel values
            total_weight = 0.0
            pixel_sum = np.zeros(3, dtype=np.float32)
            processed = False

            # Sort by distance to process closest features first (optional optimization)
            # target_pixel_contributors.sort(key=lambda mf: mf.sq_dist(pos))

            for mf in target_pixel_contributors:
                sq_dist = mf.sq_dist(pos)
                weight = mf.quality / (sq_dist + 1e-6) # Inverse distance weighting combined with feature quality
                if weight <= 1e-6: continue

                q_coords = mf.query_pixel_at(pos)
                t_coords = mf.train_pixel_at(pos)
                pixel_val = np.zeros(3, dtype=np.float32)
                contrib_count = 0

                # Sample from query frame if coordinates are valid
                if q_coords and 0 <= q_coords[1] < height and 0 <= q_coords[0] < width:
                    pixel_val += img_query[q_coords[1], q_coords[0]].astype(np.float32)
                    contrib_count += 1
                # Sample from train frame if coordinates are valid
                if t_coords and 0 <= t_coords[1] < height and 0 <= t_coords[0] < width:
                    pixel_val += img_train[t_coords[1], t_coords[0]].astype(np.float32)
                    contrib_count += 1

                if contrib_count > 0:
                    pixel_sum += (pixel_val / contrib_count) * weight
                    total_weight += weight
                    processed = True

            # Assign weighted average or fallback blend
            if processed and total_weight > 1e-6:
                outImg[y, x] = pixel_sum / total_weight
                weightSum[y, x] = 1.0 # Mark as processed using weighted average
            else:
                 # Fallback blend for pixels with no feature influence
                 outImg[y, x] = (img_query[y, x].astype(np.float32) + img_train[y, x].astype(np.float32)) * 0.5
                 # Keep weightSum low to indicate fallback was used (optional)

        # Progress Update (can be slow inside inner loop, maybe update every N rows)
        if y % 10 == 0:
             elapsed = time.time() - start_interp_time
             perc = (y+1) / height
             eta = (elapsed / perc) * (1 - perc) if perc > 0 else 0
             print(f"\r  Interpolating frame: {perc*100:.1f}%, ETA: {eta:.1f}s", end="")

    print(f"\r  Finished interpolation. Time: {time.time() - start_interp_time:.2f}s")

    # Clip and convert back to uint8
    outImg = np.clip(outImg, 0, 255).astype(np.uint8)

    # Optional: Segmentation map generation (can be slow) - currently disabled
    featureSeg = None

    return outImg, featureSeg

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Motion-Compensated Frame Interpolation using XFeat. Doubles the frame rate.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_video", help="Path for the output interpolated video file (e.g., output.mp4).")
    parser.add_argument("--max_kpts", type=int, default=4096, help="Maximum keypoints per frame for XFeat. Default: 4096")
    parser.add_argument("--match_thresh", type=float, default=0.82, help="Matching threshold for XFeat (lower is stricter). Default: 0.82")
    parser.add_argument("--device", type=str, default="auto", help="Device for PyTorch ('cpu', 'cuda', 'mps', 'auto'). Default: auto")

    args = parser.parse_args()

    # --- Setup Device ---
    if args.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Initialize XFeat ---
    try:
        xfeat_model = XFeat(top_k=args.max_kpts).to(device).eval()
        print(f"XFeat model initialized with top_k={args.max_kpts}")
    except Exception as e:
        print(f"Error initializing XFeat model: {e}")
        sys.exit(1)

    # --- Read Input Video ---
    try:
        frame_gen, shape, fps, length = read_video_frames(args.input_video)
        if length == 0: raise ValueError("Input video has no frames.")
    except Exception as e:
        print(f"Error reading video file: {e}")
        sys.exit(1)

    # --- Generator for Output Frames ---
    def generate_output_frames():
        previous_frame = None
        previous_features = None
        frame_index = 0

        for current_frame in frame_gen:
            print(f"\n--- Processing original frame {frame_index+1}/{length} ---")

            # --- Detect & Describe with XFeat ---
            start_feat_time = time.time()
            with torch.no_grad():
                current_features = xfeat_model.detectAndCompute(numpy_to_torch(current_frame, device), top_k=args.max_kpts)[0]
            print(f"  Feature detection time: {time.time() - start_feat_time:.3f}s", end="")
            if current_features and 'keypoints' in current_features:
                print(f" (Found {current_features['keypoints'].shape[0]} keypoints)")
            else:
                print(" (No keypoints found)")

            # --- Interpolate if we have a previous frame ---
            if previous_frame is not None and previous_features is not None:
                print(f"  Matching features between frame {frame_index} and {frame_index+1}...")
                start_match_time = time.time()
                motion_features = get_xfeat_motion_features(
                    xfeat_model, previous_features, current_features, args.match_thresh
                )
                print(f"  Matching time: {time.time() - start_match_time:.3f}s ({len(motion_features)} matches)")

                # Generate the interpolated frame
                inter_frame, _ = gen_inter_frame(previous_frame, current_frame, motion_features)
                yield inter_frame

            # Yield the current original frame
            yield current_frame

            # Update for next iteration
            previous_frame = current_frame
            previous_features = current_features
            frame_index += 1

        # Estimate total output frames (N originals -> N interpolated + N originals = 2N - 1 ?)
        # Let's refine this: N originals -> (N-1) interpolated + N originals = 2N - 1 frames
        estimated_output_frames = max(0, 2 * length - 1) if length > 0 else 0
        return estimated_output_frames # Return total frame count for progress bar in writer

    # --- Write Output Video ---
    try:
        output_frame_generator = generate_output_frames()
        # Get estimated frame count first
        estimated_total_frames = next(iter(output_frame_generator), 0)
        # Pass the generator itself to the writer
        write_video_frames(args.output_video, shape, output_frame_generator, fps * 2, estimated_total_frames)
        print(f"\nInterpolated video saved to: {args.output_video}")
    except Exception as e:
        print(f"\nError writing output video: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nProcessing complete.")