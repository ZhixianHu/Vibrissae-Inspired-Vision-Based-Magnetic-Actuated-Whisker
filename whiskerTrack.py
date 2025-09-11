"""
whiskerTrack.py

This script provides the main functions to track whisker displacements 
from images. By comparing the positions of red markers on whisker roots 
between a reference (initial) image and an input image, the pixel-level 
movements of each whisker can be extracted.

Workflow:
1. Load and preprocess the image (mask + undistort).
2. Detect red markers and calculate their centroids.
3. Reorder and rearrange detected points to maintain whisker index consistency.
4. Compare marker positions between reference and input states.
5. Output pixel displacements (dx, dy) for all whiskers.

Usage example:
    python whiskerTrack.py

Dependencies:
    numpy, opencv-python, pandas
"""

import numpy as np
import cv2
import pandas as pd

polygonMask = cv2.imread("img/polygonMask.png", cv2.IMREAD_GRAYSCALE)
polygonMask = cv2.merge([polygonMask, polygonMask, polygonMask])


def listReOrder(ptList):
    """
    Reorder detected centroid points in-place based on their x-coordinates.
    Ensures left-right consistency for paired whisker markers.
    
    Args:
        ptList (list of tuples): List of centroid points [(x0, y0), ..., (x7, y7)]
    """
    if ptList[0][0] < ptList[1][0]:
        mid = ptList[0]
        ptList[0] = ptList[1]
        ptList[1] = mid
    if ptList[2][0] < ptList[3][0]:
        mid = ptList[2]
        ptList[2] = ptList[3]
        ptList[3] = mid
    if ptList[4][0] < ptList[5][0]:
        mid = ptList[4]
        ptList[4] = ptList[5]
        ptList[5] = mid
    if ptList[6][0] < ptList[7][0]:
        mid = ptList[6]
        ptList[6] = ptList[7]
        ptList[7] = mid

def rearrangeList(ptList):
    """
    Rearrange centroid points into a fixed order that matches whisker IDs.
    The order is determined by experiment convention to keep whisker indexing consistent.
    
    Args:
        ptList (list of tuples): List of centroid points.
    Returns:
        list of tuples: Reordered list with consistent whisker indexing.
    """
    newList = [None] * 8
    newList[0] = ptList[0]
    newList[1] = ptList[1]
    newList[2] = ptList[3]
    newList[3] = ptList[5]
    newList[4] = ptList[7]
    newList[5] = ptList[6]
    newList[6] = ptList[4]
    newList[7] = ptList[2]
    return newList


def imgUndistort(frame, calibParaPath = "img/calibration_data.npz"):
    """
    Correct lens distortion using pre-calibrated camera parameters.
    
    Args:
        frame (numpy.ndarray): Input image frame.
        calibParaPath (str): Path to calibration data file (npz).
    
    Returns:
        numpy.ndarray: Undistorted and cropped image.
    """
    calibPara = np.load(calibParaPath)
    mtx = calibPara['mtx']
    dist = calibPara['dist']
    # rvecs = calibPara['rvecs']
    # tvecs = calibPara['tvecs']
    height, width = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    frame = frame[y:y+h, x:x+w]
    return frame

def imgPreMaskProcess(frame, mask):
    """
    Apply polygon mask to restrict processing area.
    
    Args:
        frame (numpy.ndarray): Input image.
        mask (numpy.ndarray): Mask image.
    
    Returns:
        numpy.ndarray: Masked image.
    """  
    frame = cv2.bitwise_and(frame, mask)
    return frame

def imgPreProcess(frame):
    """
    Preprocess image: apply polygon mask and undistort.
    
    Args:
        frame (numpy.ndarray): Input raw image.
    Returns:
        numpy.ndarray: Processed image ready for whisker tracking.
    """
    frame = imgPreMaskProcess(frame, polygonMask)
    frame = imgUndistort(frame)
    return frame

def imgRedFindCentroid(frame, minArea = 10, mskID = 0):
    """
    Detect red markers (whisker roots) in HSV color space, 
    filter by contour area, and calculate centroids.
    
    Args:
        frame (numpy.ndarray): Input image frame.
        minArea (int): Minimum area threshold to filter small contours.
        mskID (int): Internal flag to try different morphology pipelines.
    
    Returns:
        mask (numpy.ndarray): Binary mask of detected red regions.
        centerPtList (list of tuples): Centroid coordinates of 8 markers.
        successFlag (bool): Whether exactly 8 centroids were found.
    """
    hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mskID == 0:
        mskRedThr1_low = np.array([0, 80, 60])
        mskRedThr1_high = np.array([10, 255, 255])
        mskRedThr2_low = np.array([170, 80, 60])
        mskRedThr2_high = np.array([180, 255, 255])
        
        msk1 = cv2.inRange(hsvImg, mskRedThr1_low, mskRedThr1_high)
        msk2 = cv2.inRange(hsvImg, mskRedThr2_low, mskRedThr2_high)
        mask = msk1 + msk2
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 1)
        mask = dilation
    elif mskID == 1:
        mskRedThr1_low = np.array([0, 80, 60])
        mskRedThr1_high = np.array([10, 255, 255])
        mskRedThr2_low = np.array([170, 80, 60])
        mskRedThr2_high = np.array([180, 255, 255])
        
        msk1 = cv2.inRange(hsvImg, mskRedThr1_low, mskRedThr1_high)
        msk2 = cv2.inRange(hsvImg, mskRedThr2_low, mskRedThr2_high)
        mask = msk1 + msk2
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask, kernel, iterations = 1)
        dilation = cv2.dilate(erosion, kernel, iterations = 2)
        erosion = cv2.erode(dilation, kernel, iterations = 1)   
        mask = erosion

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filteredContours = []
    centerPtList = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= minArea:
            filteredContours.append(cnt)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centerPtList.append((cX, cY))

    if len(filteredContours) != 8:
        centerPtList = []
        # cv2.imshow("Img Red", mask)
        # cv2.imshow('Original Frame', frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        if mskID == 1:
            print("-----!!! Contour detection failed.--------")
            print("contour: ", len(contours), "mskID: ", mskID)
            return mask, centerPtList, False
        return imgRedFindCentroid(frame, minArea, mskID+1)
    listReOrder(centerPtList)

    # Visualize the result
    # trkImg = frame.copy()
    # redMask = np.zeros_like(mask, dtype=np.uint8)
    # cv2.drawContours(redMask, filteredContours, -1, 255, thickness=cv2.FILLED)
    # result = cv2.bitwise_and(frame, frame, mask=redMask)

    # for i in range(8):
    #     cv2.circle(trkImg, orgCenterPtList[i], 3, (255, 255, 255), -1)
    #     cv2.circle(redMask, orgCenterPtList[i], 3, (0, 255, 0), -1)
    # cv2.imshow("Track", trkImg)
    # cv2.imshow("Img Red", mask)
    # cv2.imshow('Original Frame', frame)
    # cv2.imshow('Result', result)
    # cv2.waitKey(0)
    return mask, centerPtList, True

orgImg = cv2.imread("img/org.jpg")
orgImg = imgPreProcess(orgImg)

orgRedMask, orgCenterPtList, orgRedFindFlag = imgRedFindCentroid(orgImg)
orgCenterPtList = rearrangeList(orgCenterPtList)

curveFitCenterPtList = [
    (282, 297),  
    (190, 287), 
    (345, 233),
    (130, 220), 
    (353,  142), 
    (135, 127), 
    (295, 74),
    (202, 66), 
]
curveFitCenterPtList = rearrangeList(curveFitCenterPtList)

curve_fit_para = [
    [  0.79557368, 282.3508395, 297.03295553,  -0.42777786],
    [  0.89075305, 190.45936981, 286.96051867,  -0.35218606],
    [  0.91560196, 345.01432358, 232.57820699,  -3.61919832],
    [  0.85412267, 130.00728004, 219.92369247,  -0.24930196],
    [  0.870223958, 353.435857,  142.422444, -0.334622223],
    [  0.87530027, 134.67438286, 127.0337014,   -0.17724465],
    [  0.830036046, 295.187737, 73.6946555, -0.218070369],
    [  0.821017368, 202.121623, 66.0259798, 0.068201556]
]
curve_fit_para = rearrangeList(curve_fit_para)

def imgRedCentroidTrack(frame, orgCenterPtL, imgPath, minArea = 10):
    """
    Track whisker displacements between original (reference) and 
    input image frames by comparing red centroid positions.
    
    Args:
        frame (numpy.ndarray): Input image frame.
        orgCenterPtL (list of tuples): Centroid positions in the reference image.
        imgPath (str): Path to current image (for debug outputs).
        minArea (int): Minimum contour area for detection.
    
    Returns:
        trkImg (numpy.ndarray): Visualization with tracking arrows drawn.
        redMask (numpy.ndarray): Binary mask of detected red markers.
        org (numpy.ndarray): Original copy of input image.
        centerPtList (list of tuples): Detected centroid coordinates.
        dx (numpy.ndarray): x-displacements of whiskers.
        dy (numpy.ndarray): y-displacements of whiskers.
    """
    org = frame.copy()
    trkImg = frame.copy()
    redMask, centerPtList, redFindFlag = imgRedFindCentroid(frame, minArea)
    centerPtList = rearrangeList(centerPtList)
    int_orgCenterPtL = [(int(pt[0]), int(pt[1])) for pt in orgCenterPtL]
    if not redFindFlag:
        print(imgPath)
        centerPtList = []
        cv2.imshow("RedCentroid Track", trkImg)
        cv2.imshow("RedCentroid Img Red", redMask)
        cv2.imshow('RedCentroid Original Frame', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return frame, redMask, org, centerPtList, [], []
    
    mask = np.zeros_like(redMask, dtype=np.uint8)
    dx = []
    dy = []
    for i in range(8):
        cv2.circle(trkImg, int_orgCenterPtL[i], 3, (255, 255, 255), -1)
        cv2.circle(trkImg, centerPtList[i], 3, (255, 255, 0), -1)
        cv2.circle(mask, centerPtList[i], 3, (0, 255, 0), -1)
        cv2.arrowedLine(trkImg, pt1=int_orgCenterPtL[i], pt2 = centerPtList[i], color=(0,255,255),\
                        thickness = 2, line_type = cv2.LINE_8, shift=0, tipLength=0.5)
        dx.append(centerPtList[i][0] - orgCenterPtL[i][0])
        dy.append(centerPtList[i][1] - orgCenterPtL[i][1])
    dx = np.array(dx)
    dy = np.array(dy)
    # return frame, redMask, org, centerPtList, dx, dy
    return trkImg, redMask, org, centerPtList, dx, dy

if __name__ == "__main__":
    img = cv2.imread("img/actuated.jpg")
    img = imgPreProcess(img)
    trkImg, redMask, org, centerPtList, dx, dy = imgRedCentroidTrack(img, orgCenterPtList, "img/actuated.png")
    cv2.imshow("Track", trkImg)
    cv2.imshow("Original", orgImg)
    cv2.waitKey(0)