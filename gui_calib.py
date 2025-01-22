"""Code from https://github.com/jerinka/camcalib_gui with minor modifications"""

from __future__ import print_function
from __future__ import division
import cv2
import numpy as np
import argparse
np.set_printoptions(suppress=True)

class Fisheye:
    def __init__(self, img):
        self.WINDOW_NAME = 'cam calib'
        self.img = img

    def get_track_vals(self):
        NAME = self.WINDOW_NAME
        fx = cv2.getTrackbarPos("fx",NAME)-1000
        fy = cv2.getTrackbarPos("fy",NAME)-1000
        cx = cv2.getTrackbarPos("cx",NAME)-1000
        cy = cv2.getTrackbarPos("cy",NAME)-1000
        k1 = (cv2.getTrackbarPos("k1",NAME)-1000)/1000
        k2 = (cv2.getTrackbarPos("k2",NAME)-1000)/1000
        p1 = (cv2.getTrackbarPos("p1",NAME)-1000)/1000
        p2 = (cv2.getTrackbarPos("p2",NAME)-1000)/100000
        k3 = (cv2.getTrackbarPos("k3",NAME)-1000)/10000
        mtx = np.array(
                        [[fx   ,  0.,  cx],
                         [  0. ,  fy,  cy],
                         [  0. ,  0.,  1.]])   
        dist = np.array([[k1, k2, p1, p2, k3]])
        return mtx, dist   

    def on_trackbar(self,val):
        mtx, dist = self.get_track_vals()
        self.dst = cv2.undistort(self.img, mtx, dist, None, None)
        cv2.imshow(self.WINDOW_NAME, self.dst)
        

    def fisheye_gui(self, save_path=None):
        """GUI for fishe eye correction
        Save parameters if save_path is provided and returns camera matrix and distortion coeffs
        """

        mtx = np.array( [[897.,  0. , 653.],
                         [  0., 973., 333.],
                         [  0.,  0. ,  1. ]])
        dist = np.array([[-0.43, -0.097,  -0.05,  0.00001,  0.0]])
        
        [[fx,  a, cx],
         [ a, fy, cy],
         [ a,  a,  a]] = mtx
        
        [[k1, k2, p1, p2, k3]] = dist
        
        NAME = self.WINDOW_NAME
        cv2.namedWindow(NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(NAME, 1000, 1000)
        cb = self.on_trackbar
        cv2.createTrackbar("fx",NAME,int(1000+fx)       , 3000, cb)
        cv2.createTrackbar("fy",NAME,int(1000+fy)       , 3000, cb)
        cv2.createTrackbar("cx",NAME,int(1000+cx)       , 3000, cb)
        cv2.createTrackbar("cy",NAME,int(1000+cy)       , 3000, cb)
        cv2.createTrackbar("k1",NAME,int(1000+k1*1000)  , 2000, cb)
        cv2.createTrackbar("k2",NAME,int(1000+k2*1000)  , 2000, cb)
        cv2.createTrackbar("p1",NAME,int(1000+p1*1000)  , 2000, cb)
        cv2.createTrackbar("p2",NAME,int(1000+p2*100000), 2000, cb)
        cv2.createTrackbar("k3",NAME,int(1000+k3 )      , 2000, cb)

        # Show some stuff
        self.on_trackbar(0)
        # Wait until user press some key
        print('Press a key to continue')
        cv2.waitKey()
        
        
        mtx, dist = self.get_track_vals()
        print('\nmtx:\n',mtx)
        print('\ndist:\n',dist,'\n\n')
        # np.save('mtx.npy',mtx)
        # np.save('dist.npy',dist,'\n')
        # cv2.imwrite('corrected.png', self.dst)

        # Save the camera matrix and distortion coefficients to a file
        if save_path:
            [[fx,  a, cx],
            [ a, fy, cy],
            [ a,  a,  a]] = mtx
            [[k1, k2, p1, p2, k3]] = dist
            
            parameters = {'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy, 'k1':k1, 'k2':k2, 'p1':p1, 'p2':p2, 'k3':k3}
            with open(save_path, 'w') as file:
                for key, value in parameters.items():
                    file.write(f"{key} = {value}\n")  # Write key-value pairs in 'key = value' format
    
        cv2.destroyAllWindows()
        return mtx, dist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='cam1_images/0.jpg', help='input image path')
    args = parser.parse_args()
    print(args)
    
    img = cv2.imread(args.path,1)
    fisheye = Fisheye(img)
    fisheye.fisheye_gui()