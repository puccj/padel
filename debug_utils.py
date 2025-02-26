"""Debugging and testing utilities, that won't be used during normal execution."""


import cv2

point = None

def click_event(event, x, y, flags, param):
    global point
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x, y)
        print(f"Selected point: {x}, {y}")

def select_point(image_path):
    """
    Open image and select a point with mouse click.
    
    Parameters
    ----------
    image_path : str
        Path to the image file.
        
    Returns
    -------
    point : tuple
        Coordinates of the selected point.

    Raises
    ------
    FileNotFoundError
        If the image is not found.
    """
    global point
    point = None    # reset point

    img = cv2.imread(image_path)

    if img is None:
        print("Images not found")
        return None


    cv2.imshow("Click a point and press any key", img)
    cv2.setMouseCallback("Click a point and press any key", click_event)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if point is None:
        print("No point selected!")
        return None

    return point

if __name__ == "__main__":
    point1 = select_point("input_videos/primo test pallina/cam1.png")
    point2 = select_point("input_videos/primo test pallina/cam2.png")
    print("Punto 1:", point1)
    print("Punto 2:", point2)

    