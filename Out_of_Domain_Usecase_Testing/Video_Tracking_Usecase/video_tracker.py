import cv2
import torch
import numpy as np
from math import sin, cos, atan2, sqrt, degrees
import sys
import os
# Make sure XFeat module is accessible
try:
    from modules.xfeat import XFeat
except ImportError:
     print("Error: Cannot find the XFeat module.")
     print("Ensure 'modules/xfeat.py' exists and the 'modules' directory is in your Python path or current directory.")
     exit()

# --- Configuration ---
INPUT_SOURCE = 'video' # 'video' or 'webcam'
VIDEO_PATH = 'video_test.mp4' # Used if INPUT_SOURCE is 'video'
WEBCAM_INDEX = 0          # Used if INPUT_SOURCE is 'webcam'

RESIZE_SCALE_FACTOR = 0.4 # <<< REDUCED for performance with 3 methods
# XFeat Params
XFEAT_TOP_K = 2048 # Reduced K for performance
XFEAT_MIN_COSSIM = 0.80 # Slightly relaxed threshold?
# SIFT Params
SIFT_NFEATURES = 1000 # Limited features
SIFT_CONTRAST_THRESH = 0.04
SIFT_EDGE_THRESH = 10
# ORB Params
ORB_NFEATURES = 1500 # Limited features
# Matching Params
MATCHER_RATIO_THRESH_SIFT = 0.75
MATCHER_RATIO_THRESH_ORB = 0.75
RANSAC_THRESHOLD = 5.0
MIN_MATCH_COUNT = 10
MIN_INLIERS_HIGH_CONF = 20 # Adjusted confidence threshold
PATH_MAX_LENGTH = 150
OUTPUT_WINDOW_NAME = 'XFeat vs SIFT vs ORB Visual Odometry'

# --- Colors (BGR) ---
CLR_BACKGROUND = (30, 30, 30)
CLR_TEXT = (255, 255, 255)
CLR_TEXT_OUTLINE = (0, 0, 0)
CLR_PATH_HIGH_CONF = (0, 255, 0)
CLR_PATH_LOW_CONF = (0, 255, 255)
CLR_PATH_NO_TRACK = (0, 0, 255)
CLR_INLIER_MATCH = (50, 200, 50) # Slightly dimmer green
CLR_KEYPOINT_ALL = (255, 150, 0)
CLR_KEYPOINT_INLIER_XF = (255, 0, 0) # Blue
CLR_KEYPOINT_INLIER_SF = (0, 0, 255) # Red
CLR_KEYPOINT_INLIER_OR = (255, 0, 255) # Magenta
CLR_MOTION_ARROW = (0, 255, 255)

# --- Method Initializations ---
print("Initializing Detectors and Matchers...")
# XFeat
try:
    xfeat = XFeat(top_k=XFEAT_TOP_K)
    print(f"XFeat model loaded. Device: {xfeat.dev}")
except FileNotFoundError:
    print("Error: XFeat weights not found.")
    exit()
except Exception as e:
    print(f"Error loading XFeat: {e}")
    exit()

# SIFT
try:
    sift = cv2.SIFT_create(nfeatures=SIFT_NFEATURES,
                           contrastThreshold=SIFT_CONTRAST_THRESH,
                           edgeThreshold=SIFT_EDGE_THRESH)
    print("SIFT detector initialized.")
    sift_available = True
except AttributeError:
     print("\nWarning: Your OpenCV installation might not include SIFT.")
     print("SIFT visualization will be skipped.")
     print("Install 'opencv-contrib-python' for SIFT support.\n")
     sift = None
     sift_available = False
except Exception as e:
     print(f"Error initializing SIFT: {e}")
     sift = None
     sift_available = False

# ORB
try:
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    print("ORB detector initialized.")
    orb_available = True
except Exception as e:
     print(f"Error initializing ORB: {e}")
     orb = None
     orb_available = False

# Matchers
matcher_sift = cv2.BFMatcher() # Default NORM_L2 for SIFT
matcher_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # NORM_HAMMING for ORB

print("Initialization complete.")
# --- End Initializations ---

# --- Input Source Initialization ---
if INPUT_SOURCE == 'video':
    video = cv2.VideoCapture(VIDEO_PATH)
    if not video.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        exit()
    print(f"Processing video: {VIDEO_PATH}")
elif INPUT_SOURCE == 'webcam':
    video = cv2.VideoCapture(WEBCAM_INDEX)
    if not video.isOpened():
        print(f"Error: Could not open webcam {WEBCAM_INDEX}.")
        exit()
    print(f"Processing webcam index: {WEBCAM_INDEX}")
else:
    print(f"Error: Invalid INPUT_SOURCE '{INPUT_SOURCE}'. Choose 'video' or 'webcam'.")
    exit()

# --- Get Frame Dimensions ---
ret, frame = video.read()
if not ret or frame is None:
    print("Error: Could not read initial frame from source.")
    video.release()
    exit()
orig_height, orig_width = frame.shape[:2]
frame_width = int(orig_width * RESIZE_SCALE_FACTOR)
frame_height = int(orig_height * RESIZE_SCALE_FACTOR)
print(f"Source native resolution: {orig_width}x{orig_height}")
print(f"Processing at scaled resolution: {frame_width}x{frame_height}")

if INPUT_SOURCE == 'video':
    video.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video

# --- Initialize Panels, Paths, and State ---
panel_width = frame_width
panel_height = frame_height
# Total window size will be determined by panel layout

# State storage for each method
tracked_path = {'xf': [], 'sf': [], 'orb': []}
old_data = {
    'xf': None,
    'sf': {'kp_obj': None, 'des': None, 'kp_coords': None},
    'orb': {'kp_obj': None, 'des': None, 'kp_coords': None}
}
estimated_center = {
    'xf': (panel_width // 2, panel_height // 2),
    'sf': (panel_width // 2, panel_height // 2),
    'orb': (panel_width // 2, panel_height // 2)
}
tracking_confidence = {'xf': 0, 'sf': 0, 'orb': 0} # Stores the confidence of the *last successful* track

print("Starting processing loop...")
frame_count = 0

while True:
    ret, frame = video.read()
    if not ret or frame is None:
        if INPUT_SOURCE == 'video':
            print("End of video reached or error reading frame.")
        else:
            print("Error: Lost connection or could not read frame from webcam.")
        break

    frame_count += 1
    try:
        frame_resized = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"Error resizing or converting frame {frame_count}: {e}")
        continue # Skip this frame

    # --- Initialize Panels ---
    # 9 panels needed for the 3x3 layout
    panel_main_xf = frame_resized.copy()
    panel_main_sf = frame_resized.copy()
    panel_main_orb = frame_resized.copy()
    panel_feat_xf = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_feat_sf = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_feat_orb = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_ps_xf = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8) # Path + Stats
    panel_ps_sf = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)
    panel_ps_orb = np.full((panel_height, panel_width, 3), CLR_BACKGROUND, dtype=np.uint8)

    # --- Results Storage for Current Frame ---
    results = {
        'xf': {'kp_np': None, 'num_kp': 0, 'des': None, 'num_raw_matches': 0, 'num_inliers': 0, 'angle': 0.0, 'mag': 0.0, 'conf': 0, 'kp_tensor':None},
        'sf': {'kp_obj': [], 'kp_coords': None, 'num_kp': 0, 'des': None, 'num_raw_matches': 0, 'num_inliers': 0, 'angle': 0.0, 'mag': 0.0, 'conf': 0},
        'orb': {'kp_obj': [], 'kp_coords': None, 'num_kp': 0, 'des': None, 'num_raw_matches': 0, 'num_inliers': 0, 'angle': 0.0, 'mag': 0.0, 'conf': 0}
    }
    current_data_xf = None # Store the direct output of xfeat detectAndCompute

    # === Feature Extraction (Run for all methods) ===
    # XFeat
    try:
        current_data_xf = xfeat.detectAndCompute(frame_resized)[0]
        results['xf']['kp_tensor'] = current_data_xf['keypoints']
        results['xf']['des'] = current_data_xf['descriptors']
        results['xf']['kp_np'] = results['xf']['kp_tensor'].cpu().numpy()
        results['xf']['num_kp'] = len(results['xf']['kp_tensor'])
    except Exception as e:
        print(f"  Warn: XFeat detectAndCompute failed: {e}")

    # SIFT
    if sift_available:
        try:
            kp_sf, des_sf = sift.detectAndCompute(gray_frame, None)
            if kp_sf is not None and len(kp_sf) > 0:
                results['sf']['kp_obj'] = kp_sf
                results['sf']['des'] = des_sf
                results['sf']['kp_coords'] = np.array([kp.pt for kp in kp_sf], dtype=np.float32)
                results['sf']['num_kp'] = len(kp_sf)
        except Exception as e:
            print(f"  Warn: SIFT detectAndCompute failed: {e}")

    # ORB
    if orb_available:
        try:
            kp_orb, des_orb = orb.detectAndCompute(gray_frame, None)
            if kp_orb is not None and len(kp_orb) > 0:
                results['orb']['kp_obj'] = kp_orb
                results['orb']['des'] = des_orb
                results['orb']['kp_coords'] = np.array([kp.pt for kp in kp_orb], dtype=np.float32)
                results['orb']['num_kp'] = len(kp_orb)
        except Exception as e:
            print(f"  Warn: ORB detectAndCompute failed: {e}")

    # === Draw All Detected Keypoints (Feature Panels) ===
    if results['xf']['kp_np'] is not None:
        for x, y in results['xf']['kp_np']: cv2.circle(panel_feat_xf, (int(round(x)), int(round(y))), 1, CLR_KEYPOINT_ALL, -1)
    if results['sf']['kp_coords'] is not None:
        for x, y in results['sf']['kp_coords']: cv2.circle(panel_feat_sf, (int(round(x)), int(round(y))), 1, CLR_KEYPOINT_ALL, -1)
    if results['orb']['kp_coords'] is not None:
        for x, y in results['orb']['kp_coords']: cv2.circle(panel_feat_orb, (int(round(x)), int(round(y))), 1, CLR_KEYPOINT_ALL, -1)

    cv2.putText(panel_feat_xf, f"XFeat Kpts: {results['xf']['num_kp']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel_feat_sf, f"SIFT Kpts: {results['sf']['num_kp']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel_feat_orb, f"ORB Kpts: {results['orb']['num_kp']}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)

    # === Matching and Motion Estimation (Run for each method) ===

    # --- XFeat ---
    if old_data['xf'] is not None and results['xf']['des'] is not None and \
       len(old_data['xf']['keypoints']) > 0 and len(results['xf']['kp_tensor']) > 0:
        try:
            idx0, idx1 = xfeat.match(old_data['xf']['descriptors'], results['xf']['des'], min_cossim=XFEAT_MIN_COSSIM)
            results['xf']['num_raw_matches'] = len(idx0)
            if results['xf']['num_raw_matches'] >= MIN_MATCH_COUNT:
                mkpts1_np = old_data['xf']['keypoints'][idx0].cpu().numpy()
                mkpts2_np = results['xf']['kp_tensor'][idx1].cpu().numpy()
                H, mask = cv2.findHomography(mkpts1_np, mkpts2_np, cv2.RANSAC, RANSAC_THRESHOLD)
                if H is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    results['xf']['num_inliers'] = np.sum(inlier_mask)
                    if results['xf']['num_inliers'] >= MIN_MATCH_COUNT:
                        inlier_mkpts1 = mkpts1_np[inlier_mask]
                        inlier_mkpts2 = mkpts2_np[inlier_mask]
                        delta = np.mean(inlier_mkpts2 - inlier_mkpts1, axis=0)
                        results['xf']['angle'] = degrees(atan2(delta[1], delta[0]))
                        results['xf']['mag'] = sqrt(delta[0]**2 + delta[1]**2)
                        results['xf']['conf'] = 2 if results['xf']['num_inliers'] >= MIN_INLIERS_HIGH_CONF else 1
                        # Draw inlier matches
                        for pt1, pt2 in zip(inlier_mkpts1, inlier_mkpts2):
                             p1 = (int(round(pt1[0])), int(round(pt1[1])))
                             p2 = (int(round(pt2[0])), int(round(pt2[1])))
                             cv2.line(panel_main_xf, p1, p2, CLR_INLIER_MATCH, 1)
                             cv2.circle(panel_main_xf, p2, 3, CLR_KEYPOINT_INLIER_XF, -1)
        except Exception as e: print(f"  Warn: XFeat matching/homography failed: {e}")

    # --- SIFT ---
    if sift_available and old_data['sf']['des'] is not None and results['sf']['des'] is not None and \
       len(old_data['sf']['des']) > 0 and len(results['sf']['des']) > 0:
        try:
            matches = matcher_sift.knnMatch(old_data['sf']['des'], results['sf']['des'], k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < MATCHER_RATIO_THRESH_SIFT * n.distance: good_matches.append(m)
            results['sf']['num_raw_matches'] = len(good_matches)
            if results['sf']['num_raw_matches'] >= MIN_MATCH_COUNT:
                mkpts1_np = np.float32([old_data['sf']['kp_coords'][m.queryIdx] for m in good_matches])
                mkpts2_np = np.float32([results['sf']['kp_coords'][m.trainIdx] for m in good_matches])
                H, mask = cv2.findHomography(mkpts1_np, mkpts2_np, cv2.RANSAC, RANSAC_THRESHOLD)
                if H is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    results['sf']['num_inliers'] = np.sum(inlier_mask)
                    if results['sf']['num_inliers'] >= MIN_MATCH_COUNT:
                        inlier_mkpts1 = mkpts1_np[inlier_mask]
                        inlier_mkpts2 = mkpts2_np[inlier_mask]
                        delta = np.mean(inlier_mkpts2 - inlier_mkpts1, axis=0)
                        results['sf']['angle'] = degrees(atan2(delta[1], delta[0]))
                        results['sf']['mag'] = sqrt(delta[0]**2 + delta[1]**2)
                        results['sf']['conf'] = 2 if results['sf']['num_inliers'] >= MIN_INLIERS_HIGH_CONF else 1
                        # Draw inlier matches
                        for pt1, pt2 in zip(inlier_mkpts1, inlier_mkpts2):
                             p1 = (int(round(pt1[0])), int(round(pt1[1])))
                             p2 = (int(round(pt2[0])), int(round(pt2[1])))
                             cv2.line(panel_main_sf, p1, p2, CLR_INLIER_MATCH, 1)
                             cv2.circle(panel_main_sf, p2, 3, CLR_KEYPOINT_INLIER_SF, -1)
        except Exception as e: print(f"  Warn: SIFT matching/homography failed: {e}")


    # --- ORB ---
    if orb_available and old_data['orb']['des'] is not None and results['orb']['des'] is not None and \
       len(old_data['orb']['des']) > 0 and len(results['orb']['des']) > 0:
        try:
            matches = matcher_orb.knnMatch(old_data['orb']['des'], results['orb']['des'], k=2)
            good_matches = []
            for pair in matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < MATCHER_RATIO_THRESH_ORB * n.distance: good_matches.append(m)
            results['orb']['num_raw_matches'] = len(good_matches)
            if results['orb']['num_raw_matches'] >= MIN_MATCH_COUNT:
                mkpts1_np = np.float32([old_data['orb']['kp_coords'][m.queryIdx] for m in good_matches])
                mkpts2_np = np.float32([results['orb']['kp_coords'][m.trainIdx] for m in good_matches])
                H, mask = cv2.findHomography(mkpts1_np, mkpts2_np, cv2.RANSAC, RANSAC_THRESHOLD)
                if H is not None and mask is not None:
                    inlier_mask = mask.ravel() == 1
                    results['orb']['num_inliers'] = np.sum(inlier_mask)
                    if results['orb']['num_inliers'] >= MIN_MATCH_COUNT:
                        inlier_mkpts1 = mkpts1_np[inlier_mask]
                        inlier_mkpts2 = mkpts2_np[inlier_mask]
                        delta = np.mean(inlier_mkpts2 - inlier_mkpts1, axis=0)
                        results['orb']['angle'] = degrees(atan2(delta[1], delta[0]))
                        results['orb']['mag'] = sqrt(delta[0]**2 + delta[1]**2)
                        results['orb']['conf'] = 2 if results['orb']['num_inliers'] >= MIN_INLIERS_HIGH_CONF else 1
                         # Draw inlier matches
                        for pt1, pt2 in zip(inlier_mkpts1, inlier_mkpts2):
                             p1 = (int(round(pt1[0])), int(round(pt1[1])))
                             p2 = (int(round(pt2[0])), int(round(pt2[1])))
                             cv2.line(panel_main_orb, p1, p2, CLR_INLIER_MATCH, 1)
                             cv2.circle(panel_main_orb, p2, 3, CLR_KEYPOINT_INLIER_OR, -1)
        except Exception as e: print(f"  Warn: ORB matching/homography failed: {e}")


    # === Update and Draw Paths (Run for each method) ===
    for method in ['xf', 'sf', 'orb']:
        if not tracked_path[method]: # Initialize
             if results[method]['num_kp'] > 0 : # Only start if features detected
                 tracked_path[method].append(estimated_center[method])
                 tracking_confidence[method] = results[method]['conf'] # Use current confidence
        elif results[method]['conf'] > 0: # Tracking successful this frame
            # Update estimated center based on this frame's motion
            last_x, last_y = tracked_path[method][-1]
            mag = results[method]['mag']
            rad = np.radians(results[method]['angle'])
            new_x = last_x + cos(rad) * mag
            new_y = last_y + sin(rad) * mag
            estimated_center[method] = (
                int(round(max(0, min(panel_width - 1, new_x)))),
                int(round(max(0, min(panel_height - 1, new_y))))
            )
            tracked_path[method].append(estimated_center[method])
            tracking_confidence[method] = results[method]['conf'] # Update overall confidence
        else: # Tracking failed, repeat last point
            if tracked_path[method]:
                tracked_path[method].append(tracked_path[method][-1])
            # Keep the old tracking_confidence for path color

        # Limit path length
        if len(tracked_path[method]) > PATH_MAX_LENGTH:
            tracked_path[method].pop(0)

        # Determine path color based on last known confidence
        if tracking_confidence[method] == 2: path_color = CLR_PATH_HIGH_CONF
        elif tracking_confidence[method] == 1: path_color = CLR_PATH_LOW_CONF
        else: path_color = CLR_PATH_NO_TRACK

        # Get correct panels for drawing
        panel_main = {'xf': panel_main_xf, 'sf': panel_main_sf, 'orb': panel_main_orb}[method]
        panel_ps = {'xf': panel_ps_xf, 'sf': panel_ps_sf, 'orb': panel_ps_orb}[method]

        # Draw paths
        if len(tracked_path[method]) > 1:
            path_np = np.array(tracked_path[method], dtype=np.int32)
            # Full path on path/stats panel
            cv2.polylines(panel_ps, [path_np], isClosed=False, color=path_color, thickness=2)
            # Recent path on main panel
            recent = path_np[max(0, len(path_np)-20):]
            if len(recent) > 1: cv2.polylines(panel_main, [recent], isClosed=False, color=path_color, thickness=2)

        # === Draw Stats and Motion Arrow (Path/Stats Panels) ===
        method_name = {'xf': "XFeat", 'sf': "SIFT", 'orb': "ORB"}[method]
        cv2.putText(panel_ps, f"Path & Stats ({method_name})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_TEXT, 1, cv2.LINE_AA)

        # Motion Arrow (only if tracking OK this frame)
        if results[method]['conf'] > 0 and results[method]['mag'] > 0.1:
            mag = results[method]['mag']
            rad = np.radians(results[method]['angle'])
            arrow_start = (panel_width // 2, panel_height * 3 // 4) # Arrow lower down
            arrow_end = (
                int(arrow_start[0] + cos(rad) * mag * 4), # Scale visually
                int(arrow_start[1] + sin(rad) * mag * 4)
            )
            arrow_end = (max(0, min(panel_width - 1, arrow_end[0])), max(0, min(panel_height - 1, arrow_end[1])))
            cv2.arrowedLine(panel_ps, arrow_start, arrow_end, CLR_MOTION_ARROW, 2, tipLength=0.3)

        # Stats Text
        conf_text = ["Lost", "Low", "High"][results[method]['conf']]
        info = [
            f"Kpts: {results[method]['num_kp']}",
            f"Match: {results[method]['num_raw_matches']}",
            f"Inlier: {results[method]['num_inliers']}",
            f"Conf: {conf_text}",
            f"Ang: {results[method]['angle']:.1f}",
            f"Disp: {results[method]['mag']:.1f}"
        ]
        y_off = 40
        for i, line in enumerate(info):
            pos = (10, y_off + i * 16) # Smaller spacing
            cv2.putText(panel_ps, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_TEXT_OUTLINE, 2, cv2.LINE_AA)
            cv2.putText(panel_ps, line, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLR_TEXT, 1, cv2.LINE_AA)


    # === Combine Panels ===
    row1 = np.hstack((panel_main_xf, panel_main_sf, panel_main_orb))
    row2 = np.hstack((panel_feat_xf, panel_feat_sf, panel_feat_orb))
    row3 = np.hstack((panel_ps_xf, panel_ps_sf, panel_ps_orb))
    combined_output = np.vstack((row1, row2, row3))

    # Display
    cv2.imshow(OUTPUT_WINDOW_NAME, combined_output)

    # === Update Old Data ===
    if current_data_xf is not None: old_data['xf'] = current_data_xf
    if sift_available and results['sf']['kp_obj'] is not None and results['sf']['des'] is not None:
        old_data['sf']['kp_obj'] = results['sf']['kp_obj']
        old_data['sf']['des'] = results['sf']['des']
        old_data['sf']['kp_coords'] = results['sf']['kp_coords']
    else: old_data['sf'] = {'kp_obj': None, 'des': None, 'kp_coords': None} # Reset if failed
    if orb_available and results['orb']['kp_obj'] is not None and results['orb']['des'] is not None:
        old_data['orb']['kp_obj'] = results['orb']['kp_obj']
        old_data['orb']['des'] = results['orb']['des']
        old_data['orb']['kp_coords'] = results['orb']['kp_coords']
    else: old_data['orb'] = {'kp_obj': None, 'des': None, 'kp_coords': None} # Reset if failed


    # --- Quit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exit requested by user.")
        break

# --- Cleanup ---
print("Releasing video source and closing windows.")
video.release()
cv2.destroyAllWindows()