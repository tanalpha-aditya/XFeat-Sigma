import cv2
import numpy as np
import os
from glob import glob
import argparse
from pathlib import Path
import sys
import tensorflow as tf
import traceback # Import traceback for detailed error reporting

# --- ZippyPoint Imports ---
# Add the path to the ZippyPoint directory if necessary
# Example: sys.path.insert(0, '/path/to/your/ZippyPoint_folder')
# Adjust the path below based on where your ZippyPoint code resides relative to this script
try:
    # Assuming ZippyPoint is a sibling directory or installed
    from ZippyPoint.models.zippypoint import load_ZippyPoint
    from ZippyPoint.models.postprocessing import PostProcessing
    from ZippyPoint.utils.utils import pre_process
except ImportError:
    print("Error: Cannot find ZippyPoint modules.")
    print("Please ensure ZippyPoint is installed or add its path to sys.path.")
    # Example:
    # zippy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ZippyPoint'))
    # sys.path.insert(0, zippy_path)
    # print(f"Attempted to add ZippyPoint path: {zippy_path}")
    # try again:
    #     from ZippyPoint.models.zippypoint import load_ZippyPoint
    #     from ZippyPoint.models.postprocessing import PostProcessing
    #     from ZippyPoint.utils.utils import pre_process
    # except ImportError:
    sys.exit("Exiting due to missing ZippyPoint modules.")


# Suppress excessive TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def compute_homography_accuracy(H_gt, H_est, corners):
    """
    Computes the Mean Homography Accuracy (MHA) as described in the paper.
    Assumes corners are in (x, y) format.
    """
    # Reshape corners for perspectiveTransform: (N, 1, 2)
    corners_reshaped = corners.reshape(-1, 1, 2).astype(np.float32)

    # Check if H_gt and H_est are valid 3x3 matrices
    if H_gt is None or H_gt.shape != (3, 3):
        # print("  Error: Ground truth homography H_gt is invalid.") # Keep logs minimal
        return np.inf * np.ones(4), False, False, False
    if H_est is None or H_est.shape != (3, 3):
        # Return infinite error if H_est couldn't be estimated (e.g., < 4 inliers)
        # print("  Note: Estimated homography H_est is invalid or None.")
        return np.inf * np.ones(4), False, False, False

    try:
        warped_corners_gt = cv2.perspectiveTransform(corners_reshaped, H_gt)
        warped_corners_est = cv2.perspectiveTransform(corners_reshaped, H_est)
    except cv2.error as e:
        # print(f"  Error during perspectiveTransform: {e}") # Keep logs minimal
        # print(f"  H_gt shape: {H_gt.shape if H_gt is not None else 'None'}, H_est shape: {H_est.shape if H_est is not None else 'None'}")
        return np.inf * np.ones(4), False, False, False

    if warped_corners_gt is None or warped_corners_est is None:
        # print("  Error: perspectiveTransform returned None.") # Keep logs minimal
        return np.inf * np.ones(4), False, False, False

    # Calculate Euclidean distance between corresponding warped corners
    corner_errors = np.sqrt(np.sum((warped_corners_gt - warped_corners_est) ** 2, axis=2)).flatten()

    # Check for NaN or Inf values in errors
    if np.any(np.isnan(corner_errors)) or np.any(np.isinf(corner_errors)):
        # print("  Warning: NaN or Inf detected in corner errors. Setting high error.") # Keep logs minimal
        mean_corner_error = np.inf
    else:
        mean_corner_error = np.mean(corner_errors)

    accuracy_at_3 = mean_corner_error < 3
    accuracy_at_5 = mean_corner_error < 5
    accuracy_at_7 = mean_corner_error < 7

    return corner_errors, accuracy_at_3, accuracy_at_5, accuracy_at_7

def extract_features_zippypoint(image, zippy_model, post_processor, resize_dim_cv2):
    """
    Extract features using ZippyPoint.
    Input is a cv2 image (BGR).
    resize_dim_cv2 is the target size for cv2.resize (W, H) or [-1] or [max_dim].
    Returns a list of cv2.KeyPoint objects and a numpy array of descriptors.
    """
    if image is None:
        return None, None

    # 1. Resize (optional) and Convert Color (BGR -> RGB)
    try:
        img_resized = image
        if isinstance(resize_dim_cv2, list) and len(resize_dim_cv2) == 2 :
             # Resize to specific WxH for cv2.resize
            target_w, target_h = resize_dim_cv2
            img_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # print(f"Debug: Resized to {img_resized.shape[1]}x{img_resized.shape[0]}")
        elif isinstance(resize_dim_cv2, list) and len(resize_dim_cv2) == 1 and resize_dim_cv2[0] > 0:
            # Resize max dimension
            h, w = image.shape[:2]
            max_dim = resize_dim_cv2[0]
            scale = max_dim / max(h, w)
            target_w = int(round(w * scale))
            target_h = int(round(h * scale))
            img_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            # print(f"Debug: Resized max dim to {img_resized.shape[1]}x{img_resized.shape[0]}")
        # Else: if resize_dim_cv2 is [-1], do nothing (img_resized remains original image)

        image_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        # print(f"  Error during image resize/color conversion: {e}") # Keep logs minimal
        return None, None

    # 2. Preprocess for ZippyPoint (Padding, Normalization, Tensor conversion)
    try:
        frame_tensor, img_pad = pre_process(image_rgb) # Expects RGB numpy array
    except Exception as e:
        # print(f"  Error during ZippyPoint pre_process: {e}") # Keep logs minimal
        return None, None

    # 3. Inference
    try:
        # Ensure model is called in inference mode if necessary (depends on ZippyPoint impl.)
        # The demo uses `ZippyPoint(frame_tensor, False)` -> training=False
        scores_tf, keypoints_tf, descriptors_tf = zippy_model(frame_tensor, training=False)
    except Exception as e:
        # print(f"  Error during ZippyPoint model inference: {e}") # Keep logs minimal
        return None, None

    # 4. Postprocessing (NMS, Thresholding, Top-K)
    try:
        scores_tf, keypoints_tf, descriptors_tf = post_processor(scores_tf, keypoints_tf, descriptors_tf)
    except Exception as e:
        # print(f"  Error during ZippyPoint post_processing: {e}") # Keep logs minimal
        return None, None

    # 5. Padding Correction
    # ZippyPoint demo uses: keypoints -= tf.constant([img_pad[2][0], img_pad[1][0]], dtype=tf.float32)
    # This subtracts (top_pad, left_pad). Assuming keypoints_tf are (y, x).
    # pre_process returns padding: ((top, bottom), (left, right), (0, 0))
    try:
        # Check if tensors are empty before proceeding
        if tf.size(keypoints_tf) == 0:
             # print("Debug: No keypoints after post-processing.")
             return [], np.array([]) # Return empty if no keypoints found

        keypoints_tf -= tf.constant([img_pad[0][0], img_pad[1][0]], dtype=tf.float32) # (top, left) padding offset

        # Convert to NumPy arrays (remove batch dimension)
        keypoints_np = keypoints_tf[0].numpy() # Shape (N, 2) -> likely (y, x)
        descriptors_np = descriptors_tf[0].numpy() # Shape (N, D)
        # scores_np = scores_tf[0].numpy() # Shape (N,) - not used directly for matching here
    except Exception as e:
        # print(f"  Error during padding correction or tensor conversion: {e}") # Keep logs minimal
        return None, None

    if keypoints_np.shape[0] == 0:
      # print("Debug: Keypoints became empty after numpy conversion (should not happen if check above works).")
      return [], np.array([]) # Return empty lists/arrays if no keypoints

    # 6. Create cv2.KeyPoint objects
    # IMPORTANT: cv2.KeyPoint expects (x, y) coordinates.
    # ZippyPoint keypoints after correction are likely still (y, x). We need to swap them.
    keypoints_cv = []
    for kp in keypoints_np:
        # kp[1] is x, kp[0] is y
        keypoints_cv.append(cv2.KeyPoint(float(kp[1]), float(kp[0]), 1)) # size=1 is arbitrary

    # Ensure descriptors are float32 for BFMatcher
    if descriptors_np.dtype != np.float32:
        descriptors_np = descriptors_np.astype(np.float32)

    return keypoints_cv, descriptors_np


def match_features(desc1, desc2):
    """
    Matches features between two sets of descriptors using BFMatcher with L2 norm.
    Returns a list of cv2.DMatch objects.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        # print("  Warning: Empty descriptors received for matching.")
        return []
    # Use L2 norm as ZippyPoint descriptors are likely normalized for it.
    # crossCheck=True means Lowe's ratio test is not applied, but enforces 1-to-1 matching.
    try:
        # Ensure descriptors are np.float32
        if desc1.dtype != np.float32: desc1 = desc1.astype(np.float32)
        if desc2.dtype != np.float32: desc2 = desc2.astype(np.float32)

        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(desc1, desc2)
        # Sort them in the order of their distance (optional but good practice)
        matches = sorted(matches, key=lambda x: x.distance)
    except cv2.error as e:
        # print(f"  Error during feature matching: {e}") # Keep logs minimal
        # print(f"  Descriptor 1 shape: {desc1.shape}, dtype: {desc1.dtype}")
        # print(f"  Descriptor 2 shape: {desc2.shape}, dtype: {desc2.dtype}")
        return []
    return matches

def run_hpatches_experiment(hpatches_path, zippy_model, post_processor, resize_dim_cv2, output_file=None):
    """
    Runs the HPatches homography estimation experiment using ZippyPoint.
    resize_dim_cv2 is the argument passed to extract_features_zippypoint.
    """
    if output_file:
        try:
            out_f = open(output_file, 'w')
            out_f.write("Sequence,Img1,Img2,NumInliers,Acc@3,Acc@5,Acc@7\n") # Header
        except IOError as e:
            print(f"Error opening output file {output_file}: {e}")
            output_file = None # Disable writing if file cannot be opened
            out_f = None
    else:
        out_f = None

    illumination_sequences = sorted(glob(os.path.join(hpatches_path, 'i_*')))
    viewpoint_sequences = sorted(glob(os.path.join(hpatches_path, 'v_*')))
    results = {
        'illumination': {'MHA@3': [], 'MHA@5': [], 'MHA@7': []},
        'viewpoint': {'MHA@3': [], 'MHA@5': [], 'MHA@7': []}
    }
    all_sequences = illumination_sequences + viewpoint_sequences
    total_pairs = len(all_sequences) * 5 # 5 pairs per sequence (1-2, 1-3, ..., 1-6)
    processed_pairs = 0

    for seq_path in all_sequences:
        seq_name = os.path.basename(seq_path)
        # print(f"Processing sequence: {seq_name}...")

        # Load reference image (image 1)
        image1_path_ppm = os.path.join(seq_path, '1.ppm')
        image1_path_png = os.path.join(seq_path, '1.png')
        image1 = cv2.imread(image1_path_ppm)
        if image1 is None:
            image1 = cv2.imread(image1_path_png)
        if image1 is None:
            # print(f"  ERROR: Could not read reference image 1 in {seq_name}. Skipping sequence.") # Keep logs minimal
            processed_pairs += 5 # Skip all pairs for this sequence
            # Record failure for all pairs in this sequence in the results
            seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'
            for img_idx in range(2, 7):
                results[seq_type]['MHA@3'].append(False)
                results[seq_type]['MHA@5'].append(False)
                results[seq_type]['MHA@7'].append(False)
                if out_f: out_f.write(f"{seq_name},1,{img_idx},-1,False,False,False\n") # Indicate img1 load failure
            continue

        h_orig, w_orig = image1.shape[:2]
        corners = np.array([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]]) # Original image corners

        # Extract features for reference image
        keypoints1, descriptors1 = extract_features_zippypoint(image1, zippy_model, post_processor, resize_dim_cv2)
        if keypoints1 is None or descriptors1 is None:
            # print(f"  ERROR: Feature extraction failed for image 1 in {seq_name}. Skipping sequence.") # Keep logs minimal
            processed_pairs += 5
            seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'
            for img_idx in range(2, 7):
                results[seq_type]['MHA@3'].append(False)
                results[seq_type]['MHA@5'].append(False)
                results[seq_type]['MHA@7'].append(False)
                if out_f: out_f.write(f"{seq_name},1,{img_idx},-1,False,False,False\n") # Indicate feat1 extract failure
            continue
        if not keypoints1:
            # print(f"  Warning: No keypoints found for image 1 in {seq_name}. Skipping sequence.") # Keep logs minimal
            processed_pairs += 5
            seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'
            for img_idx in range(2, 7):
                results[seq_type]['MHA@3'].append(False)
                results[seq_type]['MHA@5'].append(False)
                results[seq_type]['MHA@7'].append(False)
                if out_f: out_f.write(f"{seq_name},1,{img_idx},0,False,False,False\n") # 0 kpts -> 0 inliers
            continue


        seq_type = 'illumination' if seq_name.startswith('i') else 'viewpoint'

        for i in range(2, 7):
            pair_id = f"{seq_name}_1-{i}"
            # print(f"  Processing pair 1-{i}...")
            processed_pairs += 1
            print(f"Progress: {processed_pairs}/{total_pairs} ({pair_id})", end='\r') # Show progress inline

            # --- Default values for this pair (failure case) ---
            num_inliers = 0
            acc_3, acc_5, acc_7 = False, False, False
            H_est = None

            # Load target image (image i)
            image2_path_ppm = os.path.join(seq_path, f'{i}.ppm')
            image2_path_png = os.path.join(seq_path, f'{i}.png')
            image2 = cv2.imread(image2_path_ppm)
            if image2 is None:
                image2 = cv2.imread(image2_path_png)
            if image2 is None:
                # print(f"\n  ERROR: Could not read image {i} in {seq_name}. Skipping pair.") # Keep logs minimal
                num_inliers = -1 # Indicate image load failure specifically
                # Write result and continue to next image
                if out_f: out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")
                results[seq_type]['MHA@3'].append(acc_3)
                results[seq_type]['MHA@5'].append(acc_5)
                results[seq_type]['MHA@7'].append(acc_7)
                continue

            # Extract features for target image
            keypoints2, descriptors2 = extract_features_zippypoint(image2, zippy_model, post_processor, resize_dim_cv2)
            if keypoints2 is None or descriptors2 is None:
                # print(f"\n  ERROR: Feature extraction failed for image {i} in {seq_name}. Skipping pair.") # Keep logs minimal
                num_inliers = -1 # Indicate feature extraction failure
                if out_f: out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")
                results[seq_type]['MHA@3'].append(acc_3)
                results[seq_type]['MHA@5'].append(acc_5)
                results[seq_type]['MHA@7'].append(acc_7)
                continue
            if not keypoints2:
                # print(f"\n  Warning: No keypoints found for image {i} in {seq_name}. Skipping pair.")
                num_inliers = 0 # No keypoints -> 0 inliers
                if out_f: out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")
                results[seq_type]['MHA@3'].append(acc_3)
                results[seq_type]['MHA@5'].append(acc_5)
                results[seq_type]['MHA@7'].append(acc_7)
                continue

            # Match features
            matches = match_features(descriptors1, descriptors2)
            if len(matches) < 4: # Need at least 4 matches for findHomography
                # print(f"\n  Warning: Not enough matches found ({len(matches)}) for {pair_id}. Skipping pair.")
                num_inliers = 0 # 0 inliers if < 4 matches
                if out_f: out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")
                results[seq_type]['MHA@3'].append(acc_3)
                results[seq_type]['MHA@5'].append(acc_5)
                results[seq_type]['MHA@7'].append(acc_7)
                continue
            # print(f"    Found {len(matches)} raw matches.")

            # Prepare points for findHomography
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate Homography using RANSAC
            H_est, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,
                                             ransacReprojThreshold=3.0, # RANSAC reprojection threshold (pixels)
                                             maxIters=2000,
                                             confidence=0.995)

            if H_est is not None and mask is not None:
                num_inliers = int(np.sum(mask))
                # print(f"    Estimated H with {num_inliers} inliers.")
                if num_inliers < 4:
                     # print(f"\n    Warning: Not enough inliers ({num_inliers}) after RANSAC for {pair_id}. Treating as failure.")
                     H_est = None # Treat as failure if < 4 inliers
                     num_inliers = 0 # Record 0 inliers if RANSAC fails effectively
            else:
                # print(f"\n    Warning: Homography estimation failed for {pair_id}.")
                H_est = None
                num_inliers = 0 # Record 0 inliers if findHomography returns None

            # Load Ground Truth Homography
            H_gt_file = os.path.join(seq_path, f'H_1_{i}')
            try:
                H_gt = np.loadtxt(H_gt_file)
            except IOError:
                # print(f"\n  ERROR: Ground truth homography not found: {H_gt_file}. Skipping accuracy calculation for pair.") # Keep logs minimal
                # Cannot compute accuracy, record failure for this pair's accuracy metrics
                num_inliers = -1 # Indicate GT missing
                acc_3, acc_5, acc_7 = False, False, False # Treat as failure
                if out_f: out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")
                results[seq_type]['MHA@3'].append(acc_3)
                results[seq_type]['MHA@5'].append(acc_5)
                results[seq_type]['MHA@7'].append(acc_7)
                continue

            # Compute Homography Accuracy only if H_est was successfully found
            if H_est is not None:
                corner_errors, acc_3, acc_5, acc_7 = compute_homography_accuracy(H_gt, H_est, corners)
                mean_err = np.mean(corner_errors) if H_est is not None else float('inf')
                # print(f"    Mean Corner Error: {mean_err:.2f} | Acc@3: {acc_3}, Acc@5: {acc_5}, Acc@7: {acc_7}")
            # else: H_est is None, accuracy remains False (default)

            # Record results for the pair
            results[seq_type]['MHA@3'].append(acc_3)
            results[seq_type]['MHA@5'].append(acc_5)
            results[seq_type]['MHA@7'].append(acc_7)
            if out_f:
                out_f.write(f"{seq_name},1,{i},{num_inliers},{acc_3},{acc_5},{acc_7}\n")

        # print(f"Finished sequence {seq_name}.")

    print(f"\nProcessed {processed_pairs}/{total_pairs} pairs.") # Final progress update

    # Calculate final MHA percentages and print in the desired format
    summary = {}
    print("\nResults:") # <-- Changed Header
    if out_f:
        out_f.write("\nResults:\n") # <-- Changed Header for file

    for seq_type in ['illumination', 'viewpoint']:
        summary[seq_type] = {}
        count = len(results[seq_type]['MHA@3']) # Should be same for @5, @7
        print(f"  {seq_type}:") # <-- Changed format
        if out_f:
             out_f.write(f"  {seq_type}:\n") # <-- Changed format for file

        if count == 0:
            print("    (No results)") # Indicate no results with indent
            if out_f: out_f.write("    (No results)\n")
            for threshold_val in [3, 5, 7]:
                summary[seq_type][f'MHA@{threshold_val}'] = 0.0
            continue

        for threshold_val in [3, 5, 7]:
            threshold_key = f'MHA@{threshold_val}'
            acc_list = results[seq_type][threshold_key]
            # Ensure we only count valid boolean results (handle potential Nones/errors if logic changes)
            valid_results = [r for r in acc_list if isinstance(r, bool)]
            mha_percent = (np.sum(valid_results) / count) * 100 if count > 0 else 0.0
            summary[seq_type][threshold_key] = mha_percent
            print(f"    {threshold_key}: {mha_percent:.2f}") # <-- Changed format (indent, no %, name)
            if out_f:
                out_f.write(f"    {threshold_key}: {mha_percent:.2f}\n") # <-- Changed format for file

    if out_f:
        out_f.close()
        print(f"\nDetailed results saved to: {output_file}")

    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run HPatches homography estimation experiment with ZippyPoint.')
    parser.add_argument('hpatches_path', type=str, help='Path to the hpatches-sequences-release directory')
    parser.add_argument('--output', type=str, default='zippypoint_hpatches_results.csv', help='Path to the output CSV file')
    # ZippyPoint specific args (matching demo defaults)
    parser.add_argument('--resize', type=int, nargs='+', default=[480, 640], # Default resize HxW
                        help='Resize HxW (e.g., 480 640). Single value for max dimension (e.g., 800). Use -1 to disable.')
    parser.add_argument('--max_keypoints', type=int, default=1024, help='Maximum number of keypoints (-1 keeps all)')
    parser.add_argument('--keypoint_threshold', type=float, default=0.005, help='Keypoint detector confidence threshold') # Might need tuning
    parser.add_argument('--nms_window', type=int, default=3, help='NMS window size (e.g., 3)')
    parser.add_argument('--weights_path', type=str, default=None,
                        help='(Required) Path to ZippyPoint weights directory (e.g., ./ZippyPoint/models/weights)')

    args = parser.parse_args()

    # Validate resize argument and prepare args for model loading and cv2 resize
    # --- MODIFIED: Provide default for model load, keep -1 for extraction ---
    resize_arg_for_model = [480, 640] # Default shape HxW for model loading initialization
    resize_arg_for_cv2 = [-1]         # Default: No resize during feature extraction
    # --- END MODIFIED ---

    if isinstance(args.resize, list):
        if len(args.resize) == 2:
            resize_h, resize_w = args.resize # User provides H, W
            resize_arg_for_model = [resize_h, resize_w]
            resize_arg_for_cv2 = [resize_w, resize_h] # cv2 needs W, H
            print(f"Resizing images to HxW: {resize_h}x{resize_w}")
        elif len(args.resize) == 1:
            if args.resize[0] == -1:
                # --- MODIFIED: Keep default model shape, ensure no extraction resize ---
                resize_arg_for_cv2 = [-1]
                print("Not resizing images during feature extraction (using default shape for model load).")
                # --- END MODIFIED ---
            elif args.resize[0] > 0:
                max_dim = args.resize[0]
                # --- MODIFIED: Keep default model shape ---
                # resize_arg_for_model = None # OLD
                # --- END MODIFIED ---
                resize_arg_for_cv2 = [max_dim]
                print(f"Resizing max dimension to : {max_dim} (using default shape for model load).")
            else:
                 raise ValueError("Invalid single value for resize argument. Use a positive value for max dimension or -1.")
        else:
             raise ValueError("Invalid resize argument. Use H W, single value for max dim, or -1.")
    else:
         raise ValueError("Resize argument must be a list (e.g., --resize 480 640 or --resize -1).")


    # --- Load ZippyPoint Model ---
    print("Loading ZippyPoint model...")

    if args.weights_path is None:
        print("\nERROR: Path to ZippyPoint weights directory is required.")
        print("Please specify the path using the --weights_path argument.")
        print("Example: --weights_path /path/to/your/ZippyPoint/models/weights\n")
        sys.exit(1)

    weights_dir = Path(args.weights_path)
    print(f"Using weights from: {weights_dir}")

    if not weights_dir.exists() or not weights_dir.is_dir():
         print(f"\nERROR: ZippyPoint weights directory not found or not a directory at: {weights_dir}")
         print("Please ensure the path provided via --weights_path is correct.\n")
         sys.exit(1)

    try:
        # load_ZippyPoint now receives a valid shape even if --resize is -1
        zippy_model = load_ZippyPoint(weights_dir, input_shape=resize_arg_for_model)
        post_processor = PostProcessing(nms_window=args.nms_window,
                                        max_keypoints=args.max_keypoints,
                                        keypoint_threshold=args.keypoint_threshold)
    except Exception as e:
        print(f"\nERROR: Failed to load ZippyPoint model or initialize post-processor: {e}")
        print("Ensure TensorFlow is installed correctly and the weights path is valid.")
        print("\n--- Traceback ---")
        traceback.print_exc()
        print("-----------------\n")
        sys.exit(1)
    print("ZippyPoint model loaded.")
    # --- End Model Loading ---

    # Run the experiment
    run_hpatches_experiment(args.hpatches_path,
                            zippy_model,
                            post_processor,
                            resize_arg_for_cv2, # Pass the cv2-specific resize arg (-1 if no resize during extraction)
                            args.output)

    print("\nHPatches evaluation finished.")
