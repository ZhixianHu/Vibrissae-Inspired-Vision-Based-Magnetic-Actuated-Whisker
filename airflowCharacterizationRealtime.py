"""
airflowCharacterizationRealtime.py

Real-time airflow/interaction state classification on Raspberry Pi using the PiCamera.  
Tracks 8 whisker tip markers, builds short time windows of displacements, and uses FFT + peak/valley features to label each whisker as **Origin / Dynamic / Contact**.

Workflow
1. Initialize PiCamera (preview config, ~15 FPS via `FrameDurationLimits`) and output folder.
2. Load reference image `img/org.jpg`, preprocess (mask + undistort), detect 8 red centroids, and fix whisker indexing.
3. Capture frames → preprocess → track red centroids → compute per-whisker `dx, dy`.
4. Maintain sliding windows (`N` samples) of `dx, dy, dx+ i·dy`; run `fftProcess` + peak/valley checks in `flowProcess`.
5. Threshold and smooth to classify each whisker: **Origin / Dynamic / Contact** and overlay labels on the live view.

Usage example:
    python airflowCharacterizationRealtime.py

Dependencies:
    opencv-python, numpy, scipy, picamera2, plus local `whiskerTrack`
"""

import cv2
import numpy as np
from scipy.fftpack import fft, ifft
import copy
import scipy.signal as signal
import time
import os
from whiskerTrack import imgPreProcess, imgRedFindCentroid, imgRedCentroidTrack, rearrangeList

def fftProcess(list, T, N = 30):
    yf = fft(list)
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    frequencies = np.fft.fftfreq(N, T)
    magnitude_spectrum = np.abs(yf)[:N//2]
    subFrequncies = frequencies[1:]
    dominant_freq = subFrequncies[np.argmax(magnitude_spectrum)]
    return xf, 2.0/N * np.abs(yf[0:N//2]), dominant_freq, magnitude_spectrum

def flowProcess(dxL, dyL, dComplexL, dxPeakerFormer, dxValleyFormer, dyPeakerFormer, \
                dyValleyFormer, T = 1/15, N=30, magThre = 4, numWhisker = 8):
    dxArray = np.array(dxL).T
    dyArray = np.array(dyL).T
    dComplexArray = np.array(dComplexL).T

    staticFlag = [False]*numWhisker
    contactFlag = [False]*numWhisker

    def findPeakValley(array):
        peakIndex, _ = signal.find_peaks(np.abs(array))
        valleyIndex, _ = signal.find_peaks(-np.abs(array))
        peak = array[peakIndex]
        valley = array[valleyIndex]
        if len(peak) == 0:
            peakMax = array[0]
        else:
            peakMax = np.max(peak)
        if len(valley) == 0:
            valleyMin = array[0]
        else:
            valleyMin = np.min(valley)
        return peakMax, valleyMin

    dxPeakList = [0] * numWhisker
    dxValleyList = [0] * numWhisker
    dyPeakList = [0] * numWhisker
    dyValleyList = [0] * numWhisker
    for j in range(numWhisker):
        xf, yf, dominant_freq, magnitude_spectrum = fftProcess(dComplexArray[j], T, N)
        magnitude_spectrum = magnitude_spectrum[1:]
        # print(np.max(magnitude_spectrum))
        if np.max(magnitude_spectrum) < 5:
            staticFlag[j] = True
        dxPeak, dxValley = findPeakValley(dxArray[j])
        dyPeak, dyValley = findPeakValley(dyArray[j])
    
        
        dxPeakList[j] = dxPeak
        dxValleyList[j] = dxValley
        dyPeakList[j] = dyPeak
        dyValleyList[j] = dyValley

        if abs(dxPeak - dxPeakerFormer[j]) > magThre:
            contactFlag[j] = True
        if abs(dxValley - dxValleyFormer[j]) > magThre:
            contactFlag[j] = True
        if abs(dyPeak - dyPeakerFormer[j]) > magThre:
            contactFlag[j] = True
        if abs(dyValley - dyValleyFormer[j]) > magThre:
            contactFlag[j] = True
    return staticFlag, contactFlag, dxPeakList, dxValleyList, dyPeakList, dyValleyList

def dataSmooth(array, arrayFormer, thre):
    for i in range(len(array)):
        if abs(array[i] - arrayFormer[i]) <= thre:
            array[i] = arrayFormer[i]
    return array

# Experiment parameters
expID = 0
expName ='airflow'
N = 15
threshold = 1


date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
outputFolder = 'data/' + expName + '_' + str(expID) + '_' + date + '/'
os.makedirs(outputFolder, exist_ok=True)

dxFlowDataList = []
dyFlowDataList = []
dComplexFlowDataList = []

def main():

    from picamera2 import Picamera2
    camera = Picamera2()
    mode = camera.sensor_modes[0]
    config = camera.create_preview_configuration(sensor={'output_size': mode['size'], \
                                                        'bit_depth': mode['bit_depth']})
    #config = camera.create_preview_configuration(main={"size":(640, 480)})
    config["controls"]["FrameDurationLimits"] = (15000, 16677)
    camera.configure(config)
    camera.start()
    
    
    orgImg = cv2.imread("img/org.png")
    orgImg = imgPreProcess(orgImg)
    orgRedMask, orgCenterPtList, orgRedFindFlag = imgRedFindCentroid(orgImg)
    orgCenterPtList = rearrangeList(orgCenterPtList)
    int_orgCenterPtList = [(int(pt[0]), int(pt[1])) for pt in orgCenterPtList]

    k = 0
    dxFormer = [0]*8
    dyFormer = [0]*8
    while k < N:
        frame = camera.capture_array()
        print("Captured")
        r,g,b,dumb = cv2.split(frame)
        img = cv2.merge([b,g,r])
        img = imgPreProcess(img)

        trkImg, org, centerPtList, dx, dy, int_centerPtL = imgRedCentroidTrack(img, orgCenterPtList, int_orgCenterPtList)
        if dx == []:
            continue
        dx = dataSmooth(dx, dxFormer, threshold)
        dy = dataSmooth(dy, dyFormer, threshold)
        dxFormer = copy.deepcopy(dx)
        dyFormer = copy.deepcopy(dy)

        dxFlowDataList.append(dx)
        dyFlowDataList.append(dy)
        dComplexFlowDataList.append([dx[i] + 1j*dy[i] for i in range(8)])
        k += 1
        cv2.imshow("Track", trkImg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    dxData = dxFlowDataList[-N:]
    dyData = dyFlowDataList[-N:]
    dComplexData = dComplexFlowDataList[-N:]
    staticFlag, contactFlag, dxPeak, dxValley, dyPeak, dyValley = flowProcess(dxData, \
        dyData, dComplexData, [0]*8, [0]*8, [0]*8, [0]*8)

    count = 0
    currTime = time.time()
    while True:
        frame = camera.capture_array()
#         print("Captured")
        r,g,b,dumb = cv2.split(frame)
        img = cv2.merge([b,g,r])
#         timestamp = time.time()
#         image_filename = os.path.join(outputFolder, f"img_{count+30}_{timestamp:.5f}.jpg")
#         cv2.imwrite(image_filename, img)
        
        img = imgPreProcess(img)
        trkImg, org, centerPtList, dx, dy, int_centerPtL = imgRedCentroidTrack(img, orgCenterPtList, int_orgCenterPtList)
        dxFlowDataList.append(dx)
        dyFlowDataList.append(dy)
        dComplexFlowDataList.append([dx[i] + 1j*dy[i] for i in range(8)])
        
        dxData = dxFlowDataList[-N:]
        dyData = dyFlowDataList[-N:]
        dComplexData = dComplexFlowDataList[-N:]
        
        dxSubData = dxFlowDataList[-2]
        dySubData = dyFlowDataList[-2]

        staticFlag, contactFlag, dxPeak, dxValley, dyPeak, dyValley = flowProcess(dxData, \
            dyData, dComplexData, dxPeak, dxValley, dyPeak, dyValley)
        count += 1

        for i in range(8):
            if staticFlag[i] or contactFlag[i]:
                if ((centerPtList[i][0] - orgCenterPtList[i][0])**2+ \
                    (centerPtList[i][1] - orgCenterPtList[i][1])**2) <= 13:
                    cv2.putText(trkImg, "Origin", int_centerPtL[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(trkImg, "Contact", int_centerPtL[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(trkImg, "Dynamic", int_centerPtL[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Track", trkImg)
        currTime = time.time()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()    


if __name__ == "__main__":
    main()
