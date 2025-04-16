import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from xfeat_wrapper import XFeatWrapper
from accelerated_features.third_party import alike_wrapper as alike


def visualize_comparisons(image1, image2, p1, p2, original_p1, original_p2):
    """
    Visualize two sets of correspondences (p1-p2 and original_p1-original_p2)
    between two images in a stacked layout.

    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).
        p1: List of points in the first image (from your method).
        p2: List of corresponding points in the second image (from your method).
        original_p1: List of original points in the first image (from xfeat's method).
        original_p2: List of original corresponding points in the second image.
    """
    def create_combined_canvas(image1, image2, points1, points2, color_points=(0, 255, 0), color_lines=(255, 0, 0)):
        """
        Helper function to create a combined canvas with correspondences.
        """
        # Ensure images are BGR for drawing
        img1_bgr = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_bgr = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        h1, w1 = img1_bgr.shape[:2]
        h2, w2 = img2_bgr.shape[:2]
        height = max(h1, h2)
        width = w1 + w2

        # Create canvas large enough for both images side-by-side
        combined_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Place images onto the canvas
        combined_image[:h1, :w1, :] = img1_bgr
        combined_image[:h2, w1:w1+w2, :] = img2_bgr # Place image 2 next to image 1

        offset_x = w1  # Offset for points in the second image

        # Convert points to integers for drawing
        points1_int = np.int32(points1)
        points2_int = np.int32(points2)

        for pt1, pt2 in zip(points1_int, points2_int):
            # Draw points
            cv2.circle(combined_image, tuple(pt1), 5, color_points, -1)  # Points in image1
            cv2.circle(combined_image, (pt2[0] + offset_x, pt2[1]), 5, color_points, -1)  # Points in image2
            # Draw connecting lines
            cv2.line(combined_image, tuple(pt1), (pt2[0] + offset_x, pt2[1]), color_lines, 1) # Thinner line

        return combined_image

    # Create two combined canvases: one for the inferred points, one for the original points
    canvas1 = create_combined_canvas(image1, image2, p1, p2, color_points=(0, 255, 0), color_lines=(255, 0, 0)) # Method 1: Green pts, Red lines
    canvas2 = create_combined_canvas(image1, image2, original_p1, original_p2, color_points=(0, 0, 255), color_lines=(0, 255, 255)) # Method 2: Red pts, Cyan lines

    # Stack the two canvases vertically
    # Ensure canvases have the same width before stacking
    h_c1, w_c1 = canvas1.shape[:2]
    h_c2, w_c2 = canvas2.shape[:2]
    max_w = max(w_c1, w_c2)

    # Resize if widths differ (shouldn't happen with create_combined_canvas logic, but good practice)
    if w_c1 != max_w:
        canvas1 = cv2.resize(canvas1, (max_w, h_c1))
    if w_c2 != max_w:
        canvas2 = cv2.resize(canvas2, (max_w, h_c2))

    stacked_canvas = np.vstack((canvas1, canvas2))

    # Display the result
    plt.figure(figsize=(15, 15)) # Adjust figure size for better visibility
    plt.imshow(cv2.cvtColor(stacked_canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Top: {len(p1)} matches (Green pts, Red lines)\nBottom: {len(original_p1)} matches (Red pts, Cyan lines)")
    plt.tight_layout()
    plt.show()


def visualize_correspondences(image1, image2, p1, p2):
    """
    Visualize two images side by side with corresponding points linked by segments.

    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).
        p1: List of points in the first image [(x1, y1), (x2, y2), ...].
        p2: List of corresponding points in the second image [(x1', y1'), (x2', y2'), ...].
    """
    # Ensure images are BGR for drawing
    img1_bgr = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    img2_bgr = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1_bgr.shape[:2]
    h2, w2 = img2_bgr.shape[:2]
    height = max(h1, h2)
    width = w1 + w2

    # Create canvas large enough for both images side-by-side
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Place images onto the canvas
    combined_image[:h1, :w1, :] = img1_bgr
    combined_image[:h2, w1:w1+w2, :] = img2_bgr # Place image 2 next to image 1

    # Offset for the second image
    offset_x = w1

    # Convert points to integers for drawing
    p1_int = np.int32(p1)
    p2_int = np.int32(p2)

    # Plot the points and lines
    for pt1, pt2 in zip(p1_int, p2_int):
        # Draw points
        cv2.circle(combined_image, tuple(pt1), 5, (0, 0, 255), -1)  # Red points in image1
        cv2.circle(combined_image, (pt2[0] + offset_x, pt2[1]), 5, (0, 0, 255), -1)  # Red points in image2

        # Draw lines linking the points
        cv2.line(combined_image, tuple(pt1), (pt2[0] + offset_x, pt2[1]), (0, 255, 0), 1)  # Green line

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Point Correspondences ({len(p1)} matches)")
    plt.tight_layout()
    plt.show()


def get_points(matcher_fn, image1=None, image2=None, top_k=4092, trasformations=None, min_cossim=None, method='homography', threshold=2.5):
    '''
    Get the points from the specified matcher function.
    Passes necessary arguments based on the function's name.
    '''
    fn_name = matcher_fn.__name__

    if fn_name in ['match_xfeat_star_original', 'match_xfeat_original']:
        src_pts, dst_pts = matcher_fn(image1=image1, image2=image2, top_k=top_k)
    elif fn_name == 'match_alike':
        # Assuming alike doesn't use top_k directly in this interface call
        src_pts, dst_pts = matcher_fn(image1=image1, image2=image2)
    elif fn_name == 'match_xfeat_trasformed':
        src_pts, dst_pts = matcher_fn(image1=image1, image2=image2, top_k=top_k, trasformations=trasformations, min_cossim=min_cossim, merge=True) # Defaulting merge=True for intersection
    elif fn_name == 'match_xfeat_star_trasformed':
         src_pts, dst_pts = matcher_fn(imset1=image1, imset2=image2, top_k=top_k, trasformations=trasformations) # merge is handled internally
    elif fn_name in ['match_xfeat_refined', 'match_xfeat_star_refined']:
        # Use the threshold passed to get_points, which defaults to args.ransac_thr
        src_pts, dst_pts = matcher_fn(imset1=image1, imset2=image2, top_k=top_k, method=method, threshold=threshold, iterations=1) # Defaulting iterations=1
    else:
        raise ValueError(f"Invalid or unknown matcher function: {fn_name}")

    # Ensure output is numpy array, even if empty
    if not isinstance(src_pts, np.ndarray): src_pts = np.array(src_pts)
    if not isinstance(dst_pts, np.ndarray): dst_pts = np.array(dst_pts)

    # Reshape to (N, 2) if necessary, handle empty case
    if src_pts.size == 0: src_pts = np.empty((0, 2))
    if dst_pts.size == 0: dst_pts = np.empty((0, 2))

    return src_pts, dst_pts


def call_matcher(modality, args, xfeat_instance, image1, image2, trasformation=None):
    '''
    Call the correct matcher function based on the modality string.
    '''
    matcher_fn = None
    kwargs = {'image1': image1, 'image2': image2} # Base arguments

    if modality == 'xfeat':
        print("Running benchmark for XFeat Original...")
        matcher_fn = xfeat_instance.match_xfeat_original
        kwargs['top_k'] = 4096
    elif modality == 'xfeat-star':
        print("Running benchmark for XFeat* Original...")
        matcher_fn = xfeat_instance.match_xfeat_star_original
        kwargs['top_k'] = 10000
    elif modality == 'alike':
        print("Running benchmark for ALike...")
        matcher_fn = alike.match_alike
        # No top_k for alike in this interface
    elif modality == 'xfeat-trasformed':
        print("Running benchmark for XFeat Transformed...")
        matcher_fn = xfeat_instance.match_xfeat_trasformed
        kwargs['top_k'] = 4096
        kwargs['trasformations'] = trasformation
        kwargs['min_cossim'] = 0.8 # Example default cossim, adjust as needed
        kwargs['merge'] = True # Explicitly set merge strategy if needed
    elif modality == 'xfeat-star-trasformed':
        print("Running benchmark for XFeat* Transformed...")
        matcher_fn = xfeat_instance.match_xfeat_star_trasformed
        # Note: star methods typically take 'imset1', 'imset2'
        kwargs = {'imset1': image1, 'imset2': image2}
        kwargs['top_k'] = 10000
        kwargs['trasformations'] = trasformation
    elif modality == 'xfeat-refined':
        print("Running benchmark for XFeat Refined...")
        matcher_fn = xfeat_instance.match_xfeat_refined
        kwargs = {'imset1': image1, 'imset2': image2} # Refined methods take imset
        kwargs['top_k'] = 4096
        kwargs['method'] = args.method
        kwargs['threshold'] = args.ransac_thr # Use ransac_thr from args
    elif modality == 'xfeat-star-refined':
        print("Running benchmark for XFeat* Refined...")
        matcher_fn = xfeat_instance.match_xfeat_star_refined
        kwargs = {'imset1': image1, 'imset2': image2} # Refined methods take imset
        kwargs['top_k'] = 10000
        kwargs['method'] = args.method
        kwargs['threshold'] = args.ransac_thr # Use ransac_thr from args
    else:
        raise ValueError(f"Invalid matcher modality specified: {modality}")

    if matcher_fn:
        # Adjust args for get_points call based on matcher type
        get_points_kwargs = {
            'matcher_fn': matcher_fn,
            'image1': image1,
            'image2': image2,
            'top_k': kwargs.get('top_k'), # Pass top_k if exists
            'trasformations': kwargs.get('trasformations'), # Pass trasformations if exists
            'min_cossim': kwargs.get('min_cossim'), # Pass min_cossim if exists
            'method': kwargs.get('method', args.method), # Pass method if exists, else default
            'threshold': kwargs.get('threshold', args.ransac_thr) # Pass threshold if exists, else default
        }
        # Clean None values from get_points_kwargs as get_points handles defaults
        get_points_kwargs = {k: v for k, v in get_points_kwargs.items() if v is not None}

        return get_points(**get_points_kwargs)
    else:
         # Should not be reached due to the ValueError check, but good practice
        return np.empty((0, 2)), np.empty((0, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Qualitative comparison of feature matchers")
    parser.add_argument('--dataset-dir', type=str, required=False,
                        default='data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images',
                        help="Path to directory containing images")
    parser.add_argument('--image1-name', type=str, default='2429046426_eddd69687b_o.jpg',
                        help="Filename of the first image (relative to dataset-dir)")
    parser.add_argument('--matcher-1', type=str,
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined"],
                        default='xfeat-star-refined', # Example: Default to a refined method
                        help="Matcher for the top comparison panel.")
    parser.add_argument('--matcher-2', type=str,
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed", "xfeat-refined", "xfeat-star-refined"],
                        default='xfeat-star', # Example: Default to the base method for comparison
                        help="Matcher for the bottom comparison panel.")
    parser.add_argument('--ransac-thr', type=float, default=2.5,
                        help="RANSAC threshold value in pixels (used for 'refined' methods)")
    parser.add_argument('--method', type=str,
                        choices=['homography', 'fundamental' ],
                        default='homography',
                        help="Geometric model for 'refined' methods (homography or fundamental)")
    parser.add_argument('--max-pairs', type=int, default=5,
                        help="Maximum number of image pairs to process from the directory (0 for all)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Construct the full path for the first image
    PATH_IMAGE1 = os.path.join(args.dataset_dir, args.image1_name)

    if not os.path.isfile(PATH_IMAGE1):
        print(f"Error: Image 1 not found at {PATH_IMAGE1}")
        exit()

    xfeat_instance = XFeatWrapper() # Initialize the wrapper

    # Load the first image
    image1 = cv2.imread(PATH_IMAGE1)
    if image1 is None:
        print(f"Error: Could not load image 1 from {PATH_IMAGE1}")
        exit()
    print(f"Loaded Image 1: {PATH_IMAGE1}")

    # Define the transformations to be used (only relevant for 'trasformed' matchers)
    # This list will be passed to call_matcher if the modality requires it.
    transformations = [
        {'type': "rotation", 'angle': 15},
        {'type': "rotation", 'angle': -15},
        # {'type': "rotation", 'angle': 45},
        # {'type': "rotation", 'angle': 90},
        # {'type': "rotation", 'angle': 180}
        # Add more transformations if desired (e.g., translation, scaling)
    ]

    processed_count = 0
    # Iterate through other images in the directory
    for filename in os.listdir(args.dataset_dir):
        # Skip the first image file itself
        if filename == args.image1_name:
            continue

        path_image2 = os.path.join(args.dataset_dir, filename)

        # Check if it's a file and potentially filter by extension
        if os.path.isfile(path_image2) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"\nProcessing Pair: Image 1 vs {filename}")

            # Load the second image
            image2 = cv2.imread(path_image2)
            if image2 is None:
                print(f"Warning: Could not load image {filename}. Skipping.")
                continue

            try:
                # Run the first specified matcher
                print(f"--- Running Matcher 1: {args.matcher_1} ---")
                p1, p2 = call_matcher(args.matcher_1, args, xfeat_instance, image1, image2, trasformation=transformations)
                print(f"Matcher 1 found {len(p1)} points.")

                # Run the second specified matcher
                print(f"--- Running Matcher 2: {args.matcher_2} ---")
                p1o, p2o = call_matcher(args.matcher_2, args, xfeat_instance, image1, image2, trasformation=transformations)
                print(f"Matcher 2 found {len(p1o)} points.")

                # Visualize the comparison
                if len(p1) > 0 or len(p1o) > 0: # Only visualize if at least one method found points
                    visualize_comparisons(image1, image2, p1, p2, p1o, p2o)
                else:
                    print("Both matchers found 0 points. Skipping visualization.")

            except Exception as e:
                print(f"Error processing pair with {filename}: {e}")
                # Optionally continue to the next pair or re-raise the exception
                # raise e # Uncomment to stop on error

            processed_count += 1
            if args.max_pairs > 0 and processed_count >= args.max_pairs:
                print(f"\nReached maximum number of pairs ({args.max_pairs}). Stopping.")
                break

    print("\nFinished processing.")
