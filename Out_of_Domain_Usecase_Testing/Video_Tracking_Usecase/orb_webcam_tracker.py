import cv2
import numpy as np
from math import sin, cos, atan2, sqrt, degrees
import sys
import os

# --- Configuration ---
WEBCAM_INDEX = 0          # Index of the webcam (usually 0, 1, etc.)
RESIZE_SCALE_FACTOR = 0.6 # Adjust for performance vs detail
# ORB Parameters
ORB_NFEATURES = 2000 # Number of features to detect (adjust as needed)
# Matching Parameters
MATCHER_RATIO_THRESH = 0.75 # Lowe's ratio test threshold for ORB (good starting point)
RANSAC_THRESHOLD = 5.0  # RANSAC reprojection error threshold (pixels)
MIN_MATCH_COUNT = 10    # Minimum good matches needed for reliable motion estimate
MIN_INLIERS_HIGH_CONF = 25 # Minimum inliers for high confidence path color
PATH_MAX_LENGTH = 200 # Limit path history length
OUTPUT_WINDOW_NAME = 'ORB Enhanced Visual Odometry (Webcam)'

# --- Colors (BGR) ---
CLR_BACKGROUND = (30, 30, 30)    # Dark grey background for panels
CLR_TEXT = (255, 255, 255) # White text
CLR_TEXT_OUTLINE = (0, 0, 0) # Black outline for text
CLR_PATH_HIGH_CONF = (0, 255, 0) # Green path (good tracking)
CLR_PATH_LOW_CONF = (0, 255, 255) # Yellow path (medium tracking)
CLR_PATH_NO_TRACK = (0, 0, 255)  # Red path (tracking lost/few inliers)
CLR_INLIER_MATCH = (0, 255, 0)   # Green lines for good matches
CLR_KEYPOINT_ALL = (255, 150, 0) # Orange-ish for all detected keypoints
CLR_KEYPOINT_INLIER = (255, 0, 0) # Blue for inlier keypoints in main view
CLR_MOTION_ARROW = (0, 255, 255) # Yellow arrow for motion

# --- ORB and Matcher Initialization ---
print("Initializing ORB detector...")
try:
    # Parameters: nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    print("ORB detector initialized.")
except Exception as e:
     print(f"An unexpected error occurred initializing ORB: {e}")
     exit()

print("Initializing BFMatcher (Hamming)...")
# BFMatcher with Hamming distance for ORB descriptors
# crossCheck=False allows using knnMatch with ratio test
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
print("BFMatcher initialized.")
# --- End ORB Initialization ---

# --- Webcam Initialization ---
print(f"Attempting to open webcam index: {WEBCAM_INDEX}")
video = cv2.VideoCapture(WEBCAM_INDEX)

if not video.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}.")
    exit()

# --- Get Frame Dimensions from Webcam ---
print("Reading frame from webcam to get dimensions...")
ret, frame = video.read()
if not ret or frame is None:
    print("Error: Could not read initial frame from webcam.")
    video.release()
    exit()

orig_height, orig_width = frame.shape[:2]
frame_width = int(orig_width * RESIZE_SCALE_FACTOR)
frame_height = int(orig_height * RESIZE_SCALE_FACTOR)
print(f"Webcam native resolution: {orig_width}x{orig_height}")
print(f"Processing at scaled resolution: {frame_width}x{frame_height}")

# --- Initialize Panels, Path, and State ---
panel_width = frame_width
panel_height = frame_height
total_width = panel_width * 2
total_height = panel_height * 2

tracked_path = []
# Store previous frame's keypoints (objects), descriptors (numpy), and coordinates (numpy)
old_frame_data = {'kp_obj': None, 'des': None, 'kp_coords': None}
estimated_center = (frame_width // 2, frame_height // 2)
tracking_confidence = 0

print("Starting webcam processing loop...")
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret or frame is None:
        print("Error: Lost connection or could not read frame from webcam. Exiting.")
        break

    frame_count += 1
    frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
    gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # --- Initialize Panels ---
    panel_main = frame_resized.copy()
    panel_features = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_path = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_stats = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)

    # --- ORB Feature Extraction ---
    kp2_obj = None # KeyPoint objects
    des2 = None    # Descriptors (NumPy Binary)
    kp2_coords = None # Coordinates (NumPy)
    num_kp2 = 0
    try:
        # Detect ORB keypoints and compute descriptors
        kp2_obj, des2 = orb.detectAndCompute(gray_frame, None)

        if kp2_obj is not None and len(kp2_obj) > 0:
            # Extract coordinates into a NumPy array
            kp2_coords = np.array([kp.pt for kp in kp2_obj], dtype=np.float32)
            num_kp2 = len(kp2_obj)
        else:
            kp2_obj = []
            num_kp2 = 0

    except Exception as e:
        print(f"  Warn: ORB detectAndCompute failed on frame {frame_count}: {e}")
        kp2_obj = []
        des2 = None
        kp2_coords = None
        num_kp2 = 0


    # --- Draw All Detected Keypoints (Feature Panel) ---
    if kp2_coords is not None:
        for x, y in kp2_coords:
            if 0 <= x < panel_width and 0 <= y < panel_height:
                 cv2.circle(panel_features, (int(round(x)), int(round(y))), 1, CLR_KEYPOINT_ALL, -1)
    cv2.putText(panel_features, "Detected Keypoints (ORB)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel_features, f"Count: {num_kp2}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)


    # --- Matching and Motion Estimation ---
    num_raw_matches = 0 # Will count matches *after* ratio test
    num_inliers = 0
    mean_angle_deg = 0.0
    mean_magnitude = 0.0
    current_tracking_confidence = 0
    good_matches = [] # Store matches passing the ratio test

    # Ensure we have descriptors from both frames
    # ORB descriptors can be None if no features are found
    if old_frame_data['des'] is not None and des2 is not None and \
       len(old_frame_data['des']) > 0 and len(des2) > 0:

        old_des = old_frame_data['des']
        old_kp_coords = old_frame_data['kp_coords'] # Coordinates from previous frame

        try:
            # Find k=2 best matches for each descriptor
            matches = matcher.knnMatch(old_des, des2, k=2)

            # Apply Lowe's Ratio Test
            good_matches = []
            # Need to handle cases where knnMatch doesn't return 2 neighbors (less likely with crossCheck=False)
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < MATCHER_RATIO_THRESH * n.distance:
                        good_matches.append(m)
                elif len(match_pair) == 1: # Should not happen often with k=2 & crossCheck=False
                    # good_matches.append(match_pair[0]) # Optionally keep single good match
                    pass

            num_raw_matches = len(good_matches) # Count after ratio test

            if num_raw_matches >= MIN_MATCH_COUNT:
                # Extract coordinates of the good matches
                mkpts1_np = np.float32([old_kp_coords[m.queryIdx] for m in good_matches])
                mkpts2_np = np.float32([kp2_coords[m.trainIdx] for m in good_matches])

                # Find Homography with RANSAC
                H, mask = cv2.findHomography(mkpts1_np, mkpts2_np, cv2.RANSAC, RANSAC_THRESHOLD)

                if H is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    num_inliers = np.sum(inlier_mask)

                    if num_inliers >= MIN_MATCH_COUNT:
                        # Filter points using the RANSAC mask
                        inlier_mkpts1 = mkpts1_np[inlier_mask]
                        inlier_mkpts2 = mkpts2_np[inlier_mask]

                        # Calculate motion from inliers
                        delta_vectors = inlier_mkpts2 - inlier_mkpts1
                        avg_delta = np.mean(delta_vectors, axis=0)
                        avg_dx, avg_dy = avg_delta[0], avg_delta[1]

                        mean_angle_rad = atan2(avg_dy, avg_dx)
                        mean_angle_deg = degrees(mean_angle_rad)
                        mean_magnitude = sqrt(avg_dx**2 + avg_dy**2)

                        # Update Path Estimation
                        last_x, last_y = tracked_path[-1] if tracked_path else estimated_center
                        new_center_x = last_x + avg_dx
                        new_center_y = last_y + avg_dy
                        new_center_x = int(round(max(0, min(panel_width - 1, new_center_x))))
                        new_center_y = int(round(max(0, min(panel_height - 1, new_center_y))))
                        estimated_center = (new_center_x, new_center_y)

                        # Determine Confidence
                        if num_inliers >= MIN_INLIERS_HIGH_CONF:
                            current_tracking_confidence = 2 # High
                        else:
                            current_tracking_confidence = 1 # Low

                        # Draw Inlier Matches (Main Panel)
                        for pt1, pt2 in zip(inlier_mkpts1, inlier_mkpts2):
                             pt1_int = (int(round(pt1[0])), int(round(pt1[1])))
                             pt2_int = (int(round(pt2[0])), int(round(pt2[1])))
                             if 0 <= pt1_int[0] < panel_width and 0 <= pt1_int[1] < panel_height and \
                                0 <= pt2_int[0] < panel_width and 0 <= pt2_int[1] < panel_height:
                                 cv2.line(panel_main, pt1_int, pt2_int, CLR_INLIER_MATCH, 1)
                                 cv2.circle(panel_main, pt2_int, 3, CLR_KEYPOINT_INLIER, -1)
                    else:
                         current_tracking_confidence = 0 # Not enough inliers
                else:
                     current_tracking_confidence = 0 # Homography failed
            else:
                 current_tracking_confidence = 0 # Not enough good matches (after ratio test)

        except cv2.error as e:
             print(f"  Warn: OpenCV error during matching/homography on frame {frame_count}: {e}")
             current_tracking_confidence = 0
        except Exception as e:
            print(f"  Warn: Generic error during matching/homography on frame {frame_count}: {e}")
            current_tracking_confidence = 0
    else:
        # Not enough features/descriptors in one or both frames
        current_tracking_confidence = 0

    # --- Update and Draw Path ---
    if not tracked_path:
        if kp2_coords is not None: # Init only if features detected
             tracked_path.append(estimated_center)
             tracking_confidence = current_tracking_confidence
    elif current_tracking_confidence > 0 :
        tracked_path.append(estimated_center)
        tracking_confidence = current_tracking_confidence
    else:
        if tracked_path:
            tracked_path.append(tracked_path[-1])
        # Keep old tracking_confidence for path color

    if len(tracked_path) > PATH_MAX_LENGTH:
        tracked_path.pop(0)

    if tracking_confidence == 2: path_color = CLR_PATH_HIGH_CONF
    elif tracking_confidence == 1: path_color = CLR_PATH_LOW_CONF
    else: path_color = CLR_PATH_NO_TRACK

    if len(tracked_path) > 1:
        path_np = np.array(tracked_path, dtype=np.int32)
        cv2.polylines(panel_path, [path_np], isClosed=False, color=path_color, thickness=2)
        recent_path_segment = path_np[max(0, len(path_np)-20):]
        if len(recent_path_segment) > 1:
             cv2.polylines(panel_main, [recent_path_segment], isClosed=False, color=path_color, thickness=2)

    cv2.putText(panel_path, "Path Overview", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)

    # --- Draw Motion Arrow and Stats (Stats Panel) ---
    cv2.putText(panel_stats, "Motion & Stats (ORB)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    if current_tracking_confidence > 0 and mean_magnitude > 0.1:
        arrow_start_x = panel_width // 2
        arrow_start_y = panel_height // 2
        arrow_len = min(panel_width, panel_height) * 0.15
        arrow_end_x = int(arrow_start_x + cos(mean_angle_rad) * mean_magnitude * 5)
        arrow_end_y = int(arrow_start_y + sin(mean_angle_rad) * mean_magnitude * 5)
        arrow_end_x = max(0, min(panel_width - 1, arrow_end_x))
        arrow_end_y = max(0, min(panel_height - 1, arrow_end_y))
        cv2.arrowedLine(panel_stats, (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y),
                        CLR_MOTION_ARROW, 2, tipLength=0.3)

    conf_text = ["Lost", "Low", "High"][current_tracking_confidence]
    info_text = [
        f"Frame: {frame_count}",
        f"Kpts: {num_kp2}",
        f"Matches (Ratio Test): {num_raw_matches}",
        f"Inliers: {num_inliers}",
        f"Confidence: {conf_text}",
        f"Angle: {mean_angle_deg:.1f} deg",
        f"Disp.: {mean_magnitude:.1f} px"
    ]
    y_offset = 40
    for i, line in enumerate(info_text):
        pos = (10, y_offset + i * 18)
        cv2.putText(panel_stats, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT_OUTLINE, 2, cv2.LINE_AA)
        cv2.putText(panel_stats, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, CLR_TEXT, 1, cv2.LINE_AA)

    # --- Combine Panels ---
    top_row = np.hstack((panel_main, panel_features))
    bottom_row = np.hstack((panel_path, panel_stats))
    combined_output = np.vstack((top_row, bottom_row))

    cv2.imshow(OUTPUT_WINDOW_NAME, combined_output)

    # --- Update Old Frame Data ---
    # ORB can return None for descriptors if no features are found
    if kp2_obj is not None and des2 is not None and kp2_coords is not None:
         old_frame_data['kp_obj'] = kp2_obj
         old_frame_data['des'] = des2
         old_frame_data['kp_coords'] = kp2_coords
    else: # Reset if detection failed or found nothing
         old_frame_data = {'kp_obj': None, 'des': None, 'kp_coords': None}


    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested by user.")
        break

# --- Cleanup ---
print("Releasing webcam and closing windows.")
video.release()
cv2.destroyAllWindows()