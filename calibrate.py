"""Calibrate a fisheye camera using images of a checkerboard pattern"""

import argparse
import cv2
assert cv2.__version__[0] >= '3', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import os
import glob

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save frame as image
        frame_filename = os.path.join(output_folder, f"{video_path}_frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Done: {frame_count} frames saved in '{output_folder}'")

def calibrate(path='calibration'):
    # Inner corner of the checkerboard. A normal 8x8 checkerboard has 7x7 inner corners
    CHECKERBOARD = (7,7)

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(os.path.join(path, '*.jpg'))

    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."
            
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)

        if ret == False:
            print("No corners found in image " + fname)
            # delete the image
            os.remove(fname)
            print("Image " + fname + " deleted.")
            # continue

        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )

    [[fx,  a, cx],
    [ a, fy, cy],
    [ a,  a,  a]] = K
    [[k1], [k2], [p1], [p2]] = D

    print("Found " + str(N_OK) + " valid images for calibration in " + path)
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print("fx = " + str(fx))
    print("fy = " + str(fy))
    print("cx = " + str(cx))
    print("cy = " + str(cy))
    print("k1 = " + str(k1))
    print("k2 = " + str(k2))
    print("p1 = " + str(p1))
    print("p2 = " + str(p2))
    print("k3 = 0.0")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a fisheye camera using images of a checkerboard pattern.")
    parser.add_argument('--path', type=str, default='calibration', help='Path to the folder containing checkerboard images.')
    parser.add_argument('--video', type=str, default=None, help='Path to a video file to extract frames from for calibration.')
    args = parser.parse_args()

    if args.video:
        extract_frames(args.video, args.path)
    else:
        calibrate(args.path)