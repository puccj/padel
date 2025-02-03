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
        self.size = img.shape[:2][::-1]

    def get_track_vals(self):
        NAME = self.WINDOW_NAME
        fx = cv2.getTrackbarPos("fx",NAME)-1000
        fy = cv2.getTrackbarPos("fy",NAME)-1000
        cx = cv2.getTrackbarPos("cx",NAME)-1000
        cy = cv2.getTrackbarPos("cy",NAME)-1000
        k1 = (cv2.getTrackbarPos("k1",NAME)-1000)/1000
        k2 = (cv2.getTrackbarPos("k2",NAME)-1000)/1000
        p1 = (cv2.getTrackbarPos("p1",NAME)-1000)/1000
        p2 = (cv2.getTrackbarPos("p2",NAME)-1000)/1000
        
        mtx = np.array(
                        [[fx   ,  0.,  cx],
                         [  0. ,  fy,  cy],
                         [  0. ,  0.,  1.]])   
        dist = np.array([[k1, k2, p1, p2]])
        return mtx, dist   

    def on_trackbar(self,val):
        mtx, dist = self.get_track_vals()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, self.size, cv2.CV_16SC2)
        self.dst = cv2.remap(self.img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imshow(self.WINDOW_NAME, self.dst)
        

    def fisheye_gui(self, save_path=None):
        """GUI for fishe eye correction
        Save parameters if save_path is provided and returns camera matrix and distortion coeffs
        """

        mtx = np.array( [[482.33371165083673,         0.       , 640.5263855643664 ],
                         [         0.       , 476.5974871231719, 365.35509460062445],
                         [         0.       ,         0.       ,  1.               ]])
        dist = np.array([[-0.05232771734313145, 0.16777043123166238,  -0.24699992577326232,  0.11441728580428522]])
        
        [[fx,  a, cx],
         [ a, fy, cy],
         [ a,  a,  a]] = mtx
        
        [[k1, k2, p1, p2]] = dist
        
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
        cv2.createTrackbar("p2",NAME,int(1000+p2*1000), 2000, cb)

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
            [[k1, k2, p1, p2]] = dist
            
            parameters = {'fx':fx, 'fy':fy, 'cx':cx, 'cy':cy, 'k1':k1, 'k2':k2, 'p1':p1, 'p2':p2}
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