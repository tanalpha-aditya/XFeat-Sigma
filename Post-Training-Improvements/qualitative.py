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
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        height = max(h1, h2)
        # Ensure canvas has 3 channels even if input is grayscale
        canvas1 = np.zeros((height, w1, 3), dtype=np.uint8)
        canvas2 = np.zeros((height, w2, 3), dtype=np.uint8)

        img1_bgr = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
        img2_bgr = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

        canvas1[:h1, :w1, :] = img1_bgr
        canvas2[:h2, :w2, :] = img2_bgr


        combined_image = np.hstack((canvas1, canvas2))
        offset_x = w1  # Offset for points in the second image

        # Check if points are numpy arrays and have data
        valid_points = isinstance(points1, np.ndarray) and isinstance(points2, np.ndarray) and \
                       points1.shape[0] > 0 and points1.shape[0] == points2.shape[0]

        if valid_points:
            for (x1, y1), (x2, y2) in zip(points1, points2):
                 # Ensure points are valid numbers before drawing
                 if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
                    # Draw points
                    cv2.circle(combined_image, (int(round(x1)), int(round(y1))), 5, color_points, -1)  # Points in image1
                    cv2.circle(combined_image, (int(round(x2)) + offset_x, int(round(y2))), 5, color_points, -1)  # Points in image2
                    # Draw connecting lines
                    cv2.line(combined_image, (int(round(x1)), int(round(y1))), (int(round(x2)) + offset_x, int(round(y2))), color_lines, 2)
        else:
             print("Warning: Invalid or empty points provided to create_combined_canvas.")


        return combined_image

    # Create two combined canvases: one for the inferred points, one for the original points
    # Handle potential errors if p1/p2 or p1o/p2o are not valid np arrays or are empty
    try:
        canvas1 = create_combined_canvas(image1, image2, p1, p2, color_points=(0, 255, 0), color_lines=(255, 0, 0))
    except Exception as e:
        print(f"Error creating canvas 1 (method): {e}")
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        height = max(h1,h2)
        canvas1 = np.zeros((height, w1+w2, 3), dtype=np.uint8) # Placeholder

    try:
        canvas2 = create_combined_canvas(image1, image2, original_p1, original_p2, color_points=(0, 0, 255), color_lines=(0, 255, 255))
    except Exception as e:
        print(f"Error creating canvas 2 (original): {e}")
        h1, w1 = image1.shape[:2]
        h2, w2 = image2.shape[:2]
        height = max(h1,h2)
        canvas2 = np.zeros((height, w1+w2, 3), dtype=np.uint8) # Placeholder


    # Stack the two canvases vertically
    try:
        stacked_canvas = np.vstack((canvas1, canvas2))
    except ValueError as e:
        print(f"Error stacking canvases: {e}. Canvas shapes: {canvas1.shape}, {canvas2.shape}")
        # Handle mismatch, e.g., by showing separately or creating a default
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(canvas1, cv2.COLOR_BGR2RGB))
        plt.title("Method Correspondences (Stacking Failed)")
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(canvas2, cv2.COLOR_BGR2RGB))
        plt.title("Original Correspondences (Stacking Failed)")
        plt.show()
        return # Exit function early


    # Display the result
    plt.figure(figsize=(10, 10)) # Adjust figsize as needed
    plt.imshow(cv2.cvtColor(stacked_canvas, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Comparison of Correspondences (Top: Method, Bottom: Original)")
    plt.tight_layout() # Adjust layout
    plt.show()


# visualize_correspondences function remains the same as provided in the original code
def visualize_correspondences(image1, image2, p1, p2):
    """
    Visualize two images side by side with corresponding points linked by segments.

    Args:
        image1: The first image (numpy array).
        image2: The second image (numpy array).
        p1: List of points in the first image [(x1, y1), (x2, y2), ...].
        p2: List of corresponding points in the second image [(x1', y1'), (x2', y2'), ...].
    """
    # Ensure the images are the same height
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    height = max(h1, h2)
    canvas1 = np.zeros((height, w1, 3), dtype=np.uint8)
    canvas2 = np.zeros((height, w2, 3), dtype=np.uint8)

    canvas1[:h1, :w1, :] = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    canvas2[:h2, :w2, :] = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # Combine images side by side
    combined_image = np.hstack((canvas1, canvas2))

    # Offset for the second image
    offset_x = w1

    # Plot the points and lines
    if isinstance(p1, np.ndarray) and isinstance(p2, np.ndarray) and p1.shape[0] == p2.shape[0] and p1.shape[0] > 0:
        for (x1, y1), (x2, y2) in zip(p1, p2):
             if np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2):
                # Draw points
                cv2.circle(combined_image, (int(round(x1)), int(round(y1))), 5, (0, 0, 255), -1)  # Red points
                cv2.circle(combined_image, (int(round(x2)) + offset_x, int(round(y2))), 5, (0, 0, 255), -1) # Red points
                # Draw lines linking the points
                cv2.line(combined_image, (int(round(x1)), int(round(y1))), (int(round(x2)) + offset_x, int(round(y2))), (0, 255, 0), 1)  # Green line
    else:
        print("Warning: Invalid or empty points passed to visualize_correspondences.")


    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Point Correspondences")
    plt.tight_layout()
    plt.show()


def get_points(matcher_fn, image1=None, image2=None, top_k=4092, trasformations=None, min_cossim=0.9):
    '''
    Get the points from the matcher function.
    Only includes base XFeat, ALike, and XFeat Transformed methods.
    '''
    matcher_name = matcher_fn.__name__
    print(f"Calling get_points with matcher: {matcher_name}") # Debug print

    if matcher_name == 'match_xfeat_star_original' or matcher_name == 'match_xfeat_original':
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k)
    elif matcher_name == 'match_alike':
        src_pts, dst_pts = matcher_fn(image1, image2) # ALike might not use top_k
    elif matcher_name == 'match_xfeat_trasformed':
        # Ensure transformations is not None if required by the function
        if trasformations is None:
            print("Warning: 'trasformations' is None for match_xfeat_trasformed. Using empty list.")
            trasformations = []
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, trasformations=trasformations, min_cossim=min_cossim)
    elif matcher_name == 'match_xfeat_star_trasformed': # Corrected check
         # Ensure transformations is not None if required by the function
        if trasformations is None:
            print("Warning: 'trasformations' is None for match_xfeat_star_trasformed. Using empty list.")
            trasformations = []
        # Assuming match_xfeat_star_trasformed takes transformations but maybe not min_cossim directly in its call here
        src_pts, dst_pts = matcher_fn(image1, image2, top_k=top_k, trasformations=trasformations)
    # Removed elif conditions for _refined and _clustering methods
    else:
        raise ValueError(f"Invalid or unsupported matcher function: {matcher_name}")

    # Ensure output is numpy array, handle potential None or empty returns
    if src_pts is None or dst_pts is None:
        print(f"Warning: Matcher {matcher_name} returned None. Returning empty arrays.")
        return np.empty((0, 2)), np.empty((0, 2))
    if not isinstance(src_pts, np.ndarray): src_pts = np.array(src_pts)
    if not isinstance(dst_pts, np.ndarray): dst_pts = np.array(dst_pts)
    if src_pts.size == 0 or dst_pts.size == 0 :
         # print(f"Matcher {matcher_name} returned empty points.") # Less verbose
         return np.empty((0, 2)), np.empty((0, 2))
    if src_pts.shape[0] != dst_pts.shape[0]:
        print(f"Warning: Mismatch in number of points from {matcher_name}: {src_pts.shape[0]} vs {dst_pts.shape[0]}. Returning empty.")
        return np.empty((0, 2)), np.empty((0, 2))


    return src_pts, dst_pts


def call_matcher(modality, args, xfeat_instance, image1, image2, trasformation=None):
    '''
    Call the matcher function based on the modality.
    Only includes base XFeat, ALike, and XFeat Transformed methods.
    '''

    if modality == 'xfeat':
        print("Running matcher: XFeat Original..")
        # Pass top_k from args? Or keep default? Using default 4092 here.
        return get_points(matcher_fn = xfeat_instance.match_xfeat_original, image1=image1, image2=image2, top_k=4092)
    elif modality == 'xfeat-star':
        print("Running matcher: XFeat* Original..")
        # Pass top_k from args? Or keep default? Using default 10000 here.
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_original, image1=image1, image2=image2,  top_k=10000)
    elif modality == 'alike':
        print("Running matcher: ALike..")
        # ALike wrapper might have its own internal logic, pass None for top_k
        return get_points(matcher_fn = alike.match_alike, image1=image1, image2=image2, top_k=None)
    elif modality == 'xfeat-trasformed':
        print("Running matcher: XFeat Transformed..")
        # Ensure trasformation is passed, use default top_k and specific min_cossim
        return get_points(matcher_fn = xfeat_instance.match_xfeat_trasformed, image1=image1, image2=image2, top_k=4092, trasformations=trasformation, min_cossim=0.5)
    elif modality == 'xfeat-star-trasformed':
        print("Running matcher: XFeat* Transformed..")
         # Ensure trasformation is passed, use default top_k
        return get_points(matcher_fn = xfeat_instance.match_xfeat_star_trasformed, image1=image1, image2=image2, top_k=10000, trasformations=trasformation) # Removed min_cossim=0.5 as it's likely internal
    # Removed elif conditions for -refined and -clustering modalities
    else:
        print(f"Invalid matcher modality specified: {modality}")
        # Return empty points to avoid crashing the loop
        return np.empty((0, 2)), np.empty((0, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run qualitative comparison for feature matchers")
    parser.add_argument('--dataset-dir', type=str, required=False,
                        default='data/Mega1500/megadepth_test_1500/Undistorted_SfM/0015/images',
                        help="Path to image directory for comparison")
    parser.add_argument('--image1', type=str, default='2429046426_eddd69687b_o.jpg',
                        help="Filename of the first image within dataset-dir")
    parser.add_argument('--matcher-1', type=str,
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed"], 
                        default='xfeat-star-trasformed', # Changed default to test transformation
                        help="Matcher 1 (method to test) to use")
    parser.add_argument('--matcher-2', type=str,
                        choices=['xfeat', 'xfeat-star', 'alike', "xfeat-trasformed", "xfeat-star-trasformed"], 
                        default='xfeat-star', # Changed default to compare against original
                        help="Matcher 2 (baseline/comparison) to use")
    # Removed --method argument as it's not used in this version
    # Keep --ransac-thr even if unused by matchers here, maybe needed elsewhere? Removed for clarity.
    # parser.add_argument('--ransac-thr', type=float, default=2.5,
    #                     help="RANSAC threshold value (might not be used by selected matchers)")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Construct the full path for image 1
    PATH_IMAGE1 = os.path.join(args.dataset_dir, args.image1)

    if not os.path.exists(args.dataset_dir):
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        exit()
    if not os.path.isfile(PATH_IMAGE1):
         print(f"Error: Image 1 not found: {PATH_IMAGE1}")
         exit()


    # Initialize the wrapper (adjust device if GPU available/needed)
    # xfeat_instance = XFeatWrapper(device='cuda' if torch.cuda.is_available() else 'cpu')
    xfeat_instance = XFeatWrapper(device='cpu') # Keep CPU for simplicity unless GPU needed


    print(f"Loading Image 1: {PATH_IMAGE1}")
    image1 = cv2.imread(PATH_IMAGE1)
    if image1 is None:
        print(f"Error: Failed to load Image 1 from {PATH_IMAGE1}")
        exit()

    # Define the transformations to be used by xfeat-trasformed / xfeat-star-trasformed
    # This is fixed here, could be made configurable via args if needed
    transformations_list = [
        {
            'type': "rotation",
            'angle': 45,
        },
        {
            'type': "rotation",
            'angle': 90,
        },
        {
            'type': "rotation",
            'angle': 180,
        }
        # Add more transformations like scaling, translation etc. if desired
        # { 'type': "translation", 'pixel_x': 50, 'pixel_y': 20 },
        # { 'type': "scaling", 'scale_factor': 0.8 }, # Requires implementation in get_homography
    ]


    print(f"Comparing {args.matcher_1} against {args.matcher_2}")
    print(f"Iterating through images in: {args.dataset_dir}")

    # Iterate through files in the dataset directory to find image pairs
    processed_count = 0
    for filename in os.listdir(args.dataset_dir):
        # Skip the first image itself and non-image files
        if filename == args.image1 or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        path_image2 = os.path.join(args.dataset_dir, filename)
        print(f"\nProcessing pair: {args.image1} <-> {filename}")

        image2 = cv2.imread(path_image2)
        if image2 is None:
            print(f"Warning: Failed to load Image 2: {path_image2}. Skipping.")
            continue

        print(f"--- Running Matcher 1 ({args.matcher_1}) ---")
        p1, p2 = call_matcher(args.matcher_1, args, xfeat_instance, image1, image2, trasformation=transformations_list)
        num_matches1 = len(p1) if isinstance(p1, np.ndarray) else 0
        print(f"Number of points found by {args.matcher_1}: {num_matches1}")


        print(f"--- Running Matcher 2 ({args.matcher_2}) ---")
        # Pass transformations list even if matcher 2 doesn't use it (call_matcher handles it)
        p1o, p2o = call_matcher(args.matcher_2, args, xfeat_instance, image1, image2, trasformation=transformations_list)
        num_matches2 = len(p1o) if isinstance(p1o, np.ndarray) else 0
        print(f"Number of points found by {args.matcher_2}: {num_matches2}")


        # Visualize only if at least one matcher found points
        if num_matches1 > 0 or num_matches2 > 0:
             print("Visualizing comparison...")
             visualize_comparisons(image1, image2, p1, p2, p1o, p2o)
             # Optionally visualize individual results too
             # if num_matches1 > 0:
             #     visualize_correspondences(image1, image2, p1, p2)
             #     plt.title(f"{args.matcher_1} Matches: {num_matches1}")
             #     plt.show()
             # if num_matches2 > 0:
             #     visualize_correspondences(image1, image2, p1o, p2o)
             #     plt.title(f"{args.matcher_2} Matches: {num_matches2}")
             #     plt.show()

        else:
             print("Skipping visualization as no matches were found by either matcher.")

        processed_count += 1
        # Optional: Break after a few images for quicker testing
        # if processed_count >= 3:
        #     break

    print(f"\nFinished processing {processed_count} image pairs.")

