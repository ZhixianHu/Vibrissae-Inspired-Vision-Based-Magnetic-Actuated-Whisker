# Vibrissae-Inspired-Vision-Based-Magnetic-Actuated-Whisker
This is the code for the paper "[Vibrissae-Inspired Vision-Based Magnetic-Actuated Whisker](https://doi.org/10.1038/s41467-025-67672-x)".

## Requirements

The code has been tested with **Python 3.8–3.10**.  
Install the required packages with:

```bash
pip install -r requirements.txt
```
## `whiskerTrack.py` — Whisker Tracking from Images

This script extracts pixel-level displacements of whiskers by comparing red markers on whisker roots between a reference image and input images. It supports the sensing experiments described in the paper.

**Key Components**

* `imgPreProcess(frame)` – Combine masking and undistortion preprocessing steps.
* `imgRedFindCentroid(frame, minArea, mskID)` – Detect red markers and compute their centroids.
* `imgRedCentroidTrack(frame, orgCenterPtL, imgPath)` – Track whisker displacements by comparing centroids between reference and input frames.

**Usage Examples**

*Run directly from command line:*

```bash
# Run whisker tracking on example images
python whiskerTrack.py
```

*Import as a module in Python:*

```python
import cv2
from whiskerTrack import imgPreProcess, imgRedCentroidTrack, orgCenterPtList

# Load and preprocess the origin image
orgRedMask, orgCenterPtList, orgRedFindFlag = imgRedFindCentroid(orgImg)
orgCenterPtList = rearrangeList(orgCenterPtList)

# Load and preprocess an image
img = cv2.imread("img/input.jpg")
img = imgPreProcess(img)

# Track displacements relative to the reference whisker positions
trkImg, redMask, org, centerPtList, dx, dy = imgRedCentroidTrack(
    img, orgCenterPtList, "img/input.jpg"
)

print("Whisker displacements (dx, dy):")
print(dx, dy)
```

**Inputs**

* `img/org.jpg` – Reference image (whiskers at rest).
* `img/input.jpg` – Input image (whiskers deflected).
* `img/polygonMask.png` – Mask defining the region of interest.
* `img/calibration_data.npz` – Camera calibration parameters.

**Outputs**

* Visualization with whisker tip tracking arrows.
* Pixel displacements `dx`, `dy` for all whiskers.




## `objectClassification.ipynb` — Object Classification (10 Classes)

This script trains and tests a simple MLP (128-32-10) for classifying 10 objects (`cherry, eraser, lemon, lemonPiece, lightbulb, pear, pinecone, strawberry, tapeDispenser, yuanbao`) from feature vectors.

**Key Components**

* `objClsf` — Custom PyTorch `Dataset` loading 128-dimensional feature vectors from CSVs.
* `MLP` — Two-layer neural network with ReLU + Softmax.
* `Train` / `Test` — Training and evaluation functions (report accuracy).

**Data Format**

* CSV files in `data/objCls/{object}.csv`, 180 samples per class.
* Each line: Python-style list of 128 floats, e.g. `[0.12, 1.03, ..., 2.17]`.

**Usage**
*Run as script:*

```bash
python objectClassification.py
```


## airflowCharacterizationRealtime.py

Real-time airflow/interaction state classification on Raspberry Pi using the PiCamera.  
Tracks 8 whisker tip markers, builds short time windows of displacements, and uses FFT + peak/valley features to label each whisker as **Origin / Dynamic / Contact**.


**Usage**
Run on Raspberry Pi:
```bash
python airflowCharacterizationRealtime.py
````

* Press `q` to quit the live window.

Import selected functions (e.g., for offline replay):

```python
from airflowCharacterizationRealtime import fftProcess, flowProcess, dataSmooth
```

**Key Parameters**

* `N=15` — window length (frames) for FFT/features
* `T=1/15` — sampling period (default 15 FPS)
* `magThre=4` — contact magnitude threshold
* `threshold=1` — temporal smoothing for `dx, dy`

**Inputs**

* `img/org.jpg` – Reference image (whiskers at rest).
* `img/polygonMask.png` – Mask defining the region of interest.
* `img/calibration_data.npz` – Camera calibration parameters.
* **Runtime**: Raspberry Pi + PiCamera


**Output**

* Live OpenCV window with per-whisker labels (**Origin / Dynamic / Contact**) and tracking overlay
* Displacement buffers: `dxFlowDataList`, `dyFlowDataList`, `dComplexFlowDataList` (in memory)

**Notes**

* If you change camera FPS, update `T` accordingly for correct dominant frequency estimation.

## airflowCharacterizationOffline.py

Offline airflow/interaction state classification.  
Loads a CSV (`Time, dx, dy`), builds sliding windows of displacements,
and reuses `flowProcess/fftProcess/dataSmooth` (from the realtime module) to
label each whisker state as **Origin / Dynamic / Contact**. Also plots the
pixel displacement trace with state-colored background.

**Usage**

*Whisker displacement CSV* (`Time, dx, dy` or `Time, dx_0..., dy_0...`):

```bash
python airflowCharacterizationOffline.py --csv data/airflow/airflowDemoData.csv
```

*No plot (just classification)*:

```bash
python airflowCharacterizationOffline.py --csv data/airflow/airflowDemoData.csv --no_plot
```

**Inputs**

`data/airflow/airflowDemoData.csv` with columns:
* `Time`: frame index or timestamp
* `dx`, `dy`: per-frame displacements (pixels)

**Key Parameters**

* `T = 1/15` (sampling period), `N = 30` (window size)
* `numWhisker = 1` (this offline demo uses a single whisker trace)


**Output**

* Matplotlib figure: pixel displacement vs. time with Origin/Dynamic/Contact shading.
* In-memory lists of flags for quick verification.

**Notes**

* This script reuses `flowProcess`, `fftProcess`, and `dataSmooth` from
  `airflowCharacterizationRealtime.py`, ensuring consistency with the live pipeline.
* The demo uses `numWhisker = 1`. To extend to multiple whiskers, provide
  per-whisker columns or stacked CSVs and adapt loading accordingly.
* The raw image sequences have been deposited in the Zenodo database via the link: [https://doi.org/10.5281/zenodo.17554484](https://doi.org/10.5281/zenodo.17554484).

## Cite this article

Hu, Z., Cheng, Y., Wachs, J. et al. Vibrissae-inspired vision-based magnetic-actuated whisker. Nat Commun (2025). https://doi.org/10.1038/s41467-025-67672-x
```bibtex
@article{hu2025vibrissae,
  title={Vibrissae-inspired vision-based magnetic-actuated whisker},
  author={Hu, Zhixian and Cheng, Yi and Wachs, Juan and She, Yu},
  journal={Nature Communications},
  year={2025},
  doi={10.1038/s41467-025-67672-x},
  url={https://doi.org/10.1038/s41467-025-67672-x},
  issn={2041-1723},
  publisher={Nature Publishing Group UK London}
}


