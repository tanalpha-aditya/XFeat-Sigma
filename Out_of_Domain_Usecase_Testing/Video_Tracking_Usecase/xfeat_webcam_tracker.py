import cv2
import torch
import numpy as np
from math import sin, cos, atan2, sqrt, degrees
import sys
import os
from modules.xfeat import XFeat

# --- Configuration ---
WEBCAM_INDEX = 0          # Index of the webcam (usually 0, 1, etc.)
RESIZE_SCALE_FACTOR = 0.6 # Adjust for performance vs detail (webcams can be higher res)
XFEAT_TOP_K = 4096*4
XFEAT_MIN_COSSIM = 0.82 # Matching threshold (higher is stricter)
RANSAC_THRESHOLD = 5.0  # RANSAC reprojection error threshold (pixels)
MIN_MATCH_COUNT = 10    # Minimum matches needed for reliable motion estimate
MIN_INLIERS_HIGH_CONF = 25 # Minimum inliers for high confidence path color
PATH_MAX_LENGTH = 200 # Limit path history length
OUTPUT_WINDOW_NAME = 'XFeat Enhanced Visual Odometry (Webcam)'

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

# --- XFeat Initialization ---
print("Loading XFeat model...")
try:
    xfeat = XFeat(top_k=XFEAT_TOP_K)
    print(f"XFeat model loaded. Using device: {xfeat.dev}")
except FileNotFoundError:
    print(f"Error: XFeat weights not found. Check 'weights/xfeat.pt' relative to modules/xfeat.py")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading XFeat: {e}")
    exit()
# --- End XFeat Initialization ---

# --- Webcam Initialization ---
print(f"Attempting to open webcam index: {WEBCAM_INDEX}")
video = cv2.VideoCapture(WEBCAM_INDEX)

if not video.isOpened():
    print(f"Error: Could not open webcam {WEBCAM_INDEX}.")
    print("Check if the webcam is connected and not used by another application.")
    exit()

# --- Get Frame Dimensions from Webcam ---
# Read one frame to determine the dimensions
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
old_frame_data = None
estimated_center = (frame_width // 2, frame_height // 2) # Start path at center
tracking_confidence = 0 # 0: No track, 1: Low Conf, 2: High Conf

print("Starting webcam processing loop...")
frame_count = 0
while True: # Loop indefinitely for webcam feed
    ret, frame = video.read()
    if not ret or frame is None:
        print("Error: Lost connection or could not read frame from webcam. Exiting.")
        break # Exit the loop if reading fails

    frame_count += 1
    frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    # --- Initialize Panels ---
    panel_main = frame_resized.copy()
    panel_features = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_path = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_stats = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)

    # --- XFeat Feature Extraction ---
    kp2_np = None
    num_kp2 = 0
    current_frame_data = None # Reset current data
    try:
        # Make sure frame_resized is valid before passing to XFeat
        if frame_resized is not None and frame_resized.size > 0:
            current_frame_data = xfeat.detectAndCompute(frame_resized)[0]
            kp2 = current_frame_data['keypoints']
            des2 = current_frame_data['descriptors']
            kp2_np = kp2.cpu().numpy()
            num_kp2 = len(kp2)
        else:
             print(f"  Warn: Invalid frame encountered before XFeat on frame {frame_count}.")

    except Exception as e:
        print(f"  Warn: XFeat detectAndCompute failed on frame {frame_count}: {e}")
        # Allow loop to continue, panels will show no new data

    # --- Draw All Detected Keypoints (Feature Panel) ---
    if kp2_np is not None:
        for x, y in kp2_np:
            # Ensure coordinates are valid before drawing
            if 0 <= x < panel_width and 0 <= y < panel_height:
                 cv2.circle(panel_features, (int(round(x)), int(round(y))), 1, CLR_KEYPOINT_ALL, -1)
    cv2.putText(panel_features, "Detected Keypoints", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel_features, f"Count: {num_kp2}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)

    # --- Matching and Motion Estimation ---
    num_raw_matches = 0
    num_inliers = 0
    mean_angle_deg = 0.0
    mean_magnitude = 0.0
    # Reset confidence; it will be set based on this frame's processing
    # Keep previous tracking_confidence value only if processing fails early
    current_tracking_confidence = 0

    # Proceed only if we have current *and* old features
    if old_frame_data is not None and current_frame_data is not None and \
       len(old_frame_data['keypoints']) > 0 and len(current_frame_data['keypoints']) > 0:

        old_kp = old_frame_data['keypoints']
        old_des = old_frame_data['descriptors']
        kp2 = current_frame_data['keypoints'] # Get tensor version for matching
        des2 = current_frame_data['descriptors']

        try:
            idx0, idx1 = xfeat.match(old_des, des2, min_cossim=XFEAT_MIN_COSSIM)
            num_raw_matches = len(idx0)

            if num_raw_matches >= MIN_MATCH_COUNT:
                mkpts1_np = old_kp[idx0].cpu().numpy()
                mkpts2_np = kp2[idx1].cpu().numpy() # Use matched subset

                H, mask = cv2.findHomography(mkpts1_np, mkpts2_np, cv2.RANSAC, RANSAC_THRESHOLD)

                if H is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    num_inliers = np.sum(inlier_mask)

                    if num_inliers >= MIN_MATCH_COUNT:
                        inlier_mkpts1 = mkpts1_np[inlier_mask]
                        inlier_mkpts2 = mkpts2_np[inlier_mask]

                        delta_vectors = inlier_mkpts2 - inlier_mkpts1
                        avg_delta = np.mean(delta_vectors, axis=0)
                        avg_dx, avg_dy = avg_delta[0], avg_delta[1]

                        mean_angle_rad = atan2(avg_dy, avg_dx)
                        mean_angle_deg = degrees(mean_angle_rad)
                        mean_magnitude = sqrt(avg_dx**2 + avg_dy**2)

                        # Update Path
                        last_x, last_y = tracked_path[-1] if tracked_path else estimated_center
                        new_center_x = last_x + avg_dx
                        new_center_y = last_y + avg_dy
                        new_center_x = int(round(max(0, min(panel_width - 1, new_center_x))))
                        new_center_y = int(round(max(0, min(panel_height - 1, new_center_y))))
                        estimated_center = (new_center_x, new_center_y)

                        # Determine Confidence for *this* frame
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

                    else: # Not enough inliers
                         current_tracking_confidence = 0
                else: # Homography failed
                     current_tracking_confidence = 0
            else: # Not enough raw matches
                 current_tracking_confidence = 0

        except Exception as e:
            print(f"  Warn: Matching or Homography failed on frame {frame_count}: {e}")
            current_tracking_confidence = 0
    else:
        # Not enough features in one or both frames, or initialization phase
        current_tracking_confidence = 0


    # --- Update and Draw Path ---
    if not tracked_path: # Initialize path on first valid frame processed
        if current_frame_data is not None: # Only add start point if features were found
             tracked_path.append(estimated_center)
             tracking_confidence = current_tracking_confidence # Use current frame's confidence
    elif current_tracking_confidence > 0 : # Add new point if tracking was successful THIS frame
        tracked_path.append(estimated_center)
        tracking_confidence = current_tracking_confidence # Update global confidence
    else: # Tracking failed this frame, repeat last point and keep OLD confidence for path color
        if tracked_path: # Check if path exists before appending
            tracked_path.append(tracked_path[-1])
        # Do NOT update tracking_confidence here, keep the last known good/bad state for coloring

    # Limit path length
    if len(tracked_path) > PATH_MAX_LENGTH:
        tracked_path.pop(0)

    # Determine path color based on the *last known* tracking state
    if tracking_confidence == 2: path_color = CLR_PATH_HIGH_CONF
    elif tracking_confidence == 1: path_color = CLR_PATH_LOW_CONF
    else: path_color = CLR_PATH_NO_TRACK

    # Draw path on Main Panel (recent segment) and Path Panel (full history)
    if len(tracked_path) > 1:
        path_np = np.array(tracked_path, dtype=np.int32)
        # Draw full path on path panel
        cv2.polylines(panel_path, [path_np], isClosed=False, color=path_color, thickness=2)
        # Draw recent path segment on main panel
        recent_path_segment = path_np[max(0, len(path_np)-20):]
        if len(recent_path_segment) > 1:
             cv2.polylines(panel_main, [recent_path_segment], isClosed=False, color=path_color, thickness=2)

    cv2.putText(panel_path, "Path Overview", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)

    # --- Draw Motion Arrow and Stats (Stats Panel) ---
    cv2.putText(panel_stats, "Motion & Stats", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    # Draw arrow only if motion was calculated THIS frame
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

    # Display Stats Text (use current frame's confidence for the text display)
    conf_text = ["Lost", "Low", "High"][current_tracking_confidence]
    info_text = [
        f"Frame: {frame_count}",
        f"Kpts: {num_kp2}",
        f"Matches: {num_raw_matches}",
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

    # Display the combined output
    cv2.imshow(OUTPUT_WINDOW_NAME, combined_output)

    # Update old frame data *only* if the current frame was processed successfully
    if current_frame_data is not None:
        old_frame_data = current_frame_data

    # Check for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested by user.")
        break

# --- Cleanup ---
print("Releasing webcam and closing windows.")
video.release()
cv2.destroyAllWindows()