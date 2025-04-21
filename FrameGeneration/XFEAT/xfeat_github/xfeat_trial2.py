import cv2
import numpy as np
import torch
from tqdm import tqdm
from modules.xfeat import XFeat
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Video Frame Doubling with XFeat")
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--max_kpts', type=int, default=2048, help='Max number of keypoints')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    return parser.parse_args()



class VideoFrameDoubler:
    def __init__(self, args):
        self.args = args
        # self.xfeat = XFeat(top_k=args.max_kpts, device=args.device)
        # self.matcher = XFeat(device=args.device)

        self.xfeat = XFeat(top_k=args.max_kpts).to(args.device).eval()
        self.matcher = XFeat().to(args.device).eval()
        
        # Video capture
        self.cap = cv2.VideoCapture(args.input)
        if not self.cap.isOpened():
            raise ValueError("Could not open input video")
            
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            args.output,
            fourcc,
            self.fps * 2,  # Double the frame rate
            (self.width, self.height)
        )
        
        self.prev_frame = None
        self.prev_features = None

    
    def process_frame_pair(self, frame1, frame2):
        # Convert to grayscale and add channel dimension
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)[:, :, None]
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)[:, :, None]
        
        with torch.no_grad():
            # Convert to proper tensor format
            t1 = torch.tensor(gray1, device=self.args.device).float()
            t1 = t1.permute(2, 0, 1).unsqueeze(0) / 255.0
            
            t2 = torch.tensor(gray2, device=self.args.device).float()
            t2 = t2.permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Detect features and descriptors
            res1 = self.xfeat.detectAndCompute(t1)
            kpts1  = res1['keypoints']
            desc1 = res1['descriptors']
            res2 = self.xfeat.detectAndCompute(t2)
            kpts2, desc2 = res2['keypoints'], res2['descriptors']
            
            # Match features
            idx0, idx1 = self.matcher.match(desc1, desc2, 0.8)
            
            if len(idx0) < 20:
                return None
            
            # Convert points to numpy arrays
            pts1 = kpts1[idx0].cpu().numpy()
            pts2 = kpts2[idx1].cpu().numpy()
            
            # Estimate homography
            H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
            return H

    # Rest of the class remains the same

    def generate_interpolated_frame(self, frame1, frame2, H):
        if H is None:
            # Fallback to simple blending if no homography
            return cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)
        
        # Create intermediate transformation
        H_half = np.eye(3)
        H_half[:2] = 0.5 * (np.eye(3)[:2] + H[:2])
        
        # Warp first frame
        warped = cv2.warpPerspective(frame1, H_half, (self.width, self.height))
        
        # Blend with second frame
        return cv2.addWeighted(warped, 0.5, frame2, 0.5, 0)

    def process_video(self):
        progress = tqdm(total=self.total_frames, desc="Processing video")
        
        ret, prev_frame = self.cap.read()
        if not ret:
            return
            
        self.out.write(prev_frame)  # Write first frame
        
        while True:
            ret, curr_frame = self.cap.read()
            if not ret:
                break
                
            # Process frame pair
            H = self.process_frame_pair(prev_frame, curr_frame)
            inter_frame = self.generate_interpolated_frame(prev_frame, curr_frame, H)
            
            # Write both original and interpolated frame
            self.out.write(prev_frame)  # Original frame
            self.out.write(inter_frame)  # Interpolated frame
            
            prev_frame = curr_frame
            progress.update(1)
            
        progress.close()
        self.cap.release()
        self.out.release()

if __name__ == "__main__":
    args = parse_args()
    doubler = VideoFrameDoubler(args)
    doubler.process_video()
    print(f"Successfully processed video. Output saved to {args.output}")