"""
airflowCharacterizationOffline.py

Offline airflow/interaction state classification.  
Loads a CSV (`Time, dx, dy`), builds sliding windows of displacements,
and reuses `flowProcess/fftProcess/dataSmooth` (from the realtime module) to
label each whisker state as **Origin / Dynamic / Contact**. Also plots the
pixel displacement trace with state-colored background.

Workflow
1. Read `data/airflow/airflowDemoData.csv` with columns `Time, dx, dy`.
2. Maintain sliding windows of the last `N` frames and call `flowProcess`.
3. Produce per-frame state flags: `Origin (0)`, `Contact (1)`, `Dynamic (2)`.
4. Plot time series with shaded spans for the inferred state.

Usage examples:
python airflowCharacterizationOffline.py --csv data/airflow/airflowDemoData.csv

python airflowCharacterizationOffline.py --csv data/airflow/airflowDemoData.csv --no_plot

Dependencies:
    opencv-python, numpy, scipy, pandas, matplotlib, plus local `airflowCharacterizationRealtime`
"""

import argparse
import numpy as np
import pandas as pd
from airflowCharacterizationRealtime import flowProcess, fftProcess, dataSmooth
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import copy


def load_csv(csv_path: str, timecol: str = "Time", dxcol: str = "dx",
    dycol: str = "dy", dx_prefix: str = "dx_", dy_prefix: str = "dy_"):
    """
    Load CSV in either single- or multi-whisker layout.

    Single-whisker:
      Time, dx, dy

    Multi-whisker:
      Time, dx_0, dx_1, ..., dx_{W-1}, dy_0, dy_1, ..., dy_{W-1}

    Returns:
      time (np.ndarray) shape [T]
      dx   (np.ndarray) shape [T, W]
      dy   (np.ndarray) shape [T, W]
    """
    df = pd.read_csv(csv_path)

    if timecol not in df.columns:
        raise ValueError(f"Time column '{timecol}' not found in CSV.")

    # Try multi-whisker first (columns starting with prefixes)
    dx_cols = [c for c in df.columns if c.startswith(dx_prefix)]
    dy_cols = [c for c in df.columns if c.startswith(dy_prefix)]

    if dx_cols and dy_cols:
        dx_cols_sorted = sorted(dx_cols, key=lambda c: int(c.split("_")[-1]))
        dy_cols_sorted = sorted(dy_cols, key=lambda c: int(c.split("_")[-1]))
        if len(dx_cols_sorted) != len(dy_cols_sorted):
            raise ValueError("Mismatched number of dx_* and dy_* columns.")
        time = df[timecol].to_numpy()
        dx = df[dx_cols_sorted].to_numpy()  # [T, W]
        dy = df[dy_cols_sorted].to_numpy()  # [T, W]
        return time, dx, dy

    # Fallback to single-whisker columns
    if dxcol not in df.columns or dycol not in df.columns:
        raise ValueError(
            "CSV must have either multi-whisker dx_*/dy_* columns or "
            f"single-whisker '{dxcol}'/'{dycol}' columns."
        )

    time = df[timecol].to_numpy()
    dx = df[dxcol].to_numpy().reshape(-1, 1)  # [T, 1]
    dy = df[dycol].to_numpy().reshape(-1, 1)  # [T, 1]
    return time, dx, dy

def classify_offline(timeList: np.ndarray, dx: np.ndarray, dy: np.ndarray,
    T: float = 1 / 15, N: int = 30, smooth_threshold: float = 1.0):
    """
    Run offline classification using the same logic as the realtime pipeline.

    Args:
      timeList: [T]
      dx, dy:  [T, W]
      T:       sampling period
      N:       window size (frames)
      smooth_threshold: threshold used by dataSmooth

    Returns:
      statusFlagList: per-frame list of lists (len T) with W ints in {0,1,2}
      pixarray:       [W, T] magnitude for plotting
    """
    Tlen, W = dx.shape
    dxFormer = [0] * W
    dyFormer = [0] * W

    dxFlowDataList, dyFlowDataList, dComplexFlowDataList, pixList = [], [], [], []
    dxPeak = [0] * W
    dxValley = [0] * W
    dyPeak = [0] * W
    dyValley = [0] * W

    staticFlagList = []
    contactFlagList = []
    statusFlagList = []

    for j in range(Tlen):
        # Current frame (vector over W whiskers)
        dxSub = dx[j, :].tolist()
        dySub = dy[j, :].tolist()

        # Smoothing vs previous frame
        dxSub = dataSmooth(dxSub, dxFormer, smooth_threshold)
        dySub = dataSmooth(dySub, dyFormer, smooth_threshold)
        dxFormer = copy.deepcopy(dxSub)
        dyFormer = copy.deepcopy(dySub)

        # Append buffers
        dxFlowDataList.append(dxSub)
        dyFlowDataList.append(dySub)
        dComplexFlowDataList.append([dxSub[i] + 1j * dySub[i] for i in range(W)])
        pixList.append([np.hypot(dxSub[i], dySub[i]) for i in range(W)])

        # Classification
        statusFlag = [0] * W
        if j < N:
            # First N frames: assume static (Origin)
            staticFlag = [1] * W
            contactFlag = [0] * W
            staticFlagList.append(staticFlag)
            contactFlagList.append(contactFlag)
            statusFlag = [0] * W  # Origin
        else:
            # Sliding window over last N frames (lists of lists)
            dxData = dxFlowDataList[-N:]
            dyData = dyFlowDataList[-N:]
            dComplexData = dComplexFlowDataList[-N:]

            staticFlag, contactFlag, dxPeak, dxValley, dyPeak, dyValley = flowProcess(
                dxData,
                dyData,
                dComplexData,
                dxPeak,
                dxValley,
                dyPeak,
                dyValley,
                T=T,
                N=N,
                numWhisker=W,
            )
            staticFlagList.append(staticFlag)
            contactFlagList.append(contactFlag)

            # Gate to Origin/Contact/Dynamic
            for k in range(W):
                if staticFlag[k] or contactFlag[k]:
                    # Origin if displacement small; else Contact
                    if pixList[-1][k] <= np.sqrt(13):
                        statusFlag[k] = 0  # Origin
                    else:
                        statusFlag[k] = 1  # Contact
                else:
                    statusFlag[k] = 2  # Dynamic
        statusFlagList.append(statusFlag)

    pixarray = np.array(pixList).T  # [W, T]
    return statusFlagList, pixarray

def plot_with_states(timeList, pixarray, statusFlagList):
    """
    Plot per-whisker displacement with shaded state regions.
    """
    W, Tlen = pixarray.shape
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
    color_map = {0: "#FFFFFF", 1: sm.to_rgba(0.1), 2: sm.to_rgba(0.95)}  # Origin/Contact/Dynamic

    for i in range(W):
        plt.figure()
        plt.plot(
            timeList,
            pixarray[i],
            label=f"Pixel displacement {i}",
            color=sm.to_rgba(0),
        )
        for j in range(Tlen - 1):
            plt.axvspan(
                timeList[j],
                timeList[j + 1],
                facecolor=color_map[statusFlagList[j][i]],
                alpha=0.4,
            )
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (Pixel)")
        origin_patch = mpatches.Patch(color=color_map[0], label="Origin State")
        contact_patch = mpatches.Patch(color=color_map[1], label="Contact State")
        dynamic_patch = mpatches.Patch(color=color_map[2], label="Dynamic State")
        plt.legend(
            handles=[origin_patch, contact_patch, dynamic_patch],
            loc="upper right",
            bbox_to_anchor=(0.93, 0.92),
            facecolor="#FFFFFF",
        )
        plt.tight_layout()
        plt.show()

def parse_args():
    ap = argparse.ArgumentParser(
        description="Offline airflow/contact state classification from CSV."
    )
    ap.add_argument(
        "--csv",
        type=str,
        default="data/airflow/airflowDemoData.csv",
        help="Path to input CSV.",
    )

    ap.add_argument(
        "--dx_prefix",
        type=str,
        default="dx_",
        help="Prefix for multi-whisker dx columns (e.g., dx_0, dx_1, ...).",
    )
    ap.add_argument(
        "--dy_prefix",
        type=str,
        default="dy_",
        help="Prefix for multi-whisker dy columns (e.g., dy_0, dy_1, ...).",
    )
    ap.add_argument("--T", type=float, default=1 / 15, help="Sampling period (s).")
    ap.add_argument("--N", type=int, default=30, help="Window size (frames).")
    ap.add_argument(
        "--smooth_threshold",
        type=float,
        default=1.0,
        help="Smoothing threshold for dataSmooth.",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, do not show plots (classification only).",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    # Load CSV in flexible formats
    timeList, dx, dy = load_csv(
        csv_path=args.csv,
        dx_prefix=args.dx_prefix,
        dy_prefix=args.dy_prefix,
    )

    # Classify
    statusFlagList, pixarray = classify_offline(
        timeList=timeList,
        dx=dx,
        dy=dy,
        T=args.T,
        N=args.N,
        smooth_threshold=args.smooth_threshold,
    )

    # Print a quick summary for sanity check
    last_states = statusFlagList[-1]
    counts = {s: last_states.count(s) for s in set(last_states)}
    print(f"[OK] Classified {dx.shape[0]} frames over {dx.shape[1]} whisker(s).")
    print(f"Last-frame state counts: {counts} (0=Origin, 1=Contact, 2=Dynamic)")

    if not args.no_plot:
        plot_with_states(timeList, pixarray, statusFlagList)


if __name__ == "__main__":
    main()