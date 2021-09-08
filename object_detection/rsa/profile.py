# -*- coding: utf-8 -*-
"""Extracts camera information used for rolling shutter attack modeling.

This script analyzes the videos taken in the dark room and extracts
the number of rows that the rolling shutter attack illuminated by 
analyzing individual camera frames.

Example:
    Print the camera-extracted information about number of illuminated
    rows to console for a given video::

        $ python profile.py -f /home/data/profiling/Logitech/30.1Hz/Exposure\ 1/freq_30.1_exp_1_dc_10.mkv
    
    Perform the analysis for an entire folder `/home/data/profiling/<camera_id>`
    and save a .csv with results in `/home/data/results/profiling/<camera_id>`::

        $ python profile.py --camera_id Logitech 

"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import scipy.stats
import skvideo.io
import gc

from joblib import Parallel, delayed


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m, m - h, m + h, h


def get_mask(
    diff: np.array, thresh_r: float, thresh_g: float, thresh_b: float
) -> np.array:
    """Extracts a binary mask from an image using a cutoff threshold.

    Args:
        diff: numpy array containing the image in shape [h,w,3].
        thresh_r: cutoff threshold on red channel.
        thresh_g: cutoff threshold on green channel.
        thresh_b: cutoff threshold on blue channel.

    Returns:
        A numpy array containing the extracted binary mask.

    """
    h, w, _ = diff.shape
    r_flat, g_flat, b_flat = diff.reshape(-1, 3).T.copy()

    b_flat[b_flat >= thresh_b] = 255
    b_flat[b_flat < thresh_b] = 0
    g_flat[g_flat >= thresh_g] = 255
    g_flat[g_flat < thresh_g] = 0
    r_flat[r_flat >= thresh_r] = 255
    r_flat[r_flat < thresh_r] = 0

    b = b_flat.reshape((h, w, 1))
    g = g_flat.reshape((h, w, 1))
    r = r_flat.reshape((h, w, 1))
    diff_mask = np.concatenate((b, g, r), axis=-1)  # (h, w, 3)

    diff_mask = np.sum(diff_mask, axis=-1)  # (h, w)
    diff_mask = np.clip(diff_mask, 0, 1)  # (h, w)

    diff_mask = diff_mask[..., np.newaxis].astype(np.uint8)  # (h, w) -> (h, w, 1)
    diff_mask = np.tile(diff_mask, (1, 1, 3))  # (h, w, 1) -> (h, w, 3)
    return diff_mask


def get_mask_row_indexes(masks: np.array) -> list:
    """Extracts indexes of illuminated (True) rows in masks.

    Args:
        masks: array containing a set of binary masks, shape [n,h,w,3]

    Returns:
        A n-long list containing indexes of activated rows in the n masks

    """
    # flatten last two dimensions, what remains is [n,h]
    sm = masks.sum(axis=-1).sum(axis=-1)
    indxs = []
    for p in sm:
        indxs.append(np.argwhere(p != 0).flatten())
    return indxs


def get_masks_every_n(
    video_fpath: str,
    n: int,
    thresh: tuple = (10, 10, 10),
    dead_lines: bool = False,
) -> tuple:
    """Analyze frames from video and extract binary masks of illuminated pixels.

    Args:
        video_fpath: video file absolute path.
        n: analyze every n-th frame only.
        thresh: cutoff threshold for binarization.
        dead_lines: whether the camera has dead lines

    Returns:
        A tuple with extracted difference frames, binary masks, and row indexes

    """
    # read file
    vidcap_gen = skvideo.io.vreader(video_fpath)

    # set a reference frame under no-illumination to be all zeros
    reference = np.zeros_like(next(vidcap_gen))

    diffs, diffs_masks, indexes = [], [], []

    for i, ith_frame in enumerate(vidcap_gen):
        if i % n == 0:
            diff = np.abs(ith_frame.astype(float) - reference.astype(float)).astype(
                np.uint8
            )
            diff_mask = get_mask(diff, thresh[0], thresh[1], thresh[2])  # (h, w, 3)
            h, w, _ = ith_frame.shape
            if 0 == diff_mask.sum():
                # if there's no illuminated pixel in the mask, then skip this frame
                continue
            if 0 < diff_mask.sum() < (h * 0.5 / 100.0) * w:
                # only consider illumination when it's at least 0.5% of height
                print(
                    f"Skipping mask of {diff_mask.sum():d} pixels, lower than 0.5% height ({(h * 0.5/100.0) * w:.1f})"
                )
                continue
            if dead_lines:
                # make sure we ignore frames where the perturbation is partly in the dead areas
                bin_mask = diff_mask[..., 0].astype(bool)  # (h, w)
                aff_row_mask = np.any(bin_mask, axis=1)  # (h,)
                aff_row_indexes = np.where(aff_row_mask)[0]
                first_aff_row_i = aff_row_indexes.min()
                last_aff_row_i = aff_row_indexes.max()
                if first_aff_row_i != 0 and last_aff_row_i != diff_mask.shape[0] - 1:
                    # guarantees that perturbation is completely visible and not in the dead lines areas
                    diffs.append(diff)
                    diffs_masks.append(diff_mask)
                    indexes.append(i)
                else:
                    print(
                        f"Skipping frame first_row={first_aff_row_i:d}, last_row={last_aff_row_i:d}"
                    )
            else:
                diffs.append(diff)
                diffs_masks.append(diff_mask)
                indexes.append(i)

    diffs_np = np.array(diffs)
    diffs_mask_np = np.array(diffs_masks)

    # these can be large depending on camera resolution, so explicitly delete them
    del diffs
    del diffs_masks

    gc.collect()

    return diffs_np, diffs_mask_np, indexes


def process_one_file(
    filepath: str, dead_lines: bool = False, pen: int = 5
) -> pd.DataFrame:
    """Extracts row-information from a video file

    Args:
        filepath: video file absolute path.
        pen: analyze every n-th frame only.
        dead_lines: whether the camera has dead lines

    Returns:
        A pd.DataFrame containing the extracted row-information.

    """
    file_ext = filepath.split(".")[-1]

    if file_ext.lower() not in _ALLOWED_VIDEO_FMTS:
        # only analyze files with formats allowed in _ALLOWED_VIDEO_FMTS
        return pd.DataFrame()

    params = params_from_fullpath(filepath)
    sys.stdout.write("* --- %s\n" % filepath[6:])

    try:
        # get binary masks with illuminated pixels
        _, diffs_masks, _ = get_masks_every_n(filepath, n=pen, dead_lines=dead_lines)
        # get exact masks illuminated pixels row-indices
        aff_row_indexes = get_mask_row_indexes(diffs_masks)
    except (ValueError, TypeError) as e:
        print(f"ERROR - skip {filepath}", str(e))
        return None
    # diffs_masks is (n, h, w, 3)
    no_aff_rows = np.array([len(x) for x in aff_row_indexes]).astype(int)

    # construct result dictionary
    params = {k: [params[k]] for k in params.keys()}
    params["affected_rows_mean"] = [no_aff_rows.mean()]
    params["affected_rows_std"] = [no_aff_rows.std()]
    params["affected_rows_99ci"] = [
        mean_confidence_interval(no_aff_rows, confidence=0.99)[-1]
    ]
    params["affected_rows_95ci"] = [
        mean_confidence_interval(no_aff_rows, confidence=0.95)[-1]
    ]

    df = pd.DataFrame.from_dict(params)

    return df


def params_from_fullpath(fullpath: str) -> dict:
    """Infer video file parameters based on its absolute path.
    The parameters include:
     * the id of the camera
     * the frequency of the laser at the time of recording
     * the exposure of the camera at the time of recording
     * the duty cycle of the laser at the time of recording

    Args:
        fullpath: video file absolute path.

    Returns:
        A dict with the video parameters

    """
    # A file correctly placed would match this template filepath
    # /home/data/profiling/CAM_ID/30.1Hz/Exposure X/freq_30.1_exp_30_dc_5.mkv
    assert os.path.isfile(fullpath)
    params = os.path.abspath(fullpath).split(os.path.sep)[4:]

    try:
        cam_id, freq, exposure, fname = params
        _, _, _, _, _, duty_cycle = fname[:-4].split("_")

        params = {
            "frequency": freq,
            "exposure": exposure,
            "duty_cycle": duty_cycle,
            "camera": cam_id,
        }
    except Exception as e:
        print(
            f"Could not infer parameters for video file {fullpath}. Returning defaults"
        )
        params = {
            "frequency": "000",
            "exposure": "000",
            "duty_cycle": "000",
            "camera": "None",
        }
    return params


# Holds camera-specific information
_CCONF = {
    "Logitech": {"dead_lines": False, "pattern_every_n": 5},
    "Axis": {"dead_lines": True, "pattern_every_n": 2},
}

# Only analyse video files in these formats
_ALLOWED_VIDEO_FMTS = ["mkv"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse video files to compute the number of rows affected by \
            the rolling shutter attack illumination; construct a .csv file with \
            the results (or print to console)."
    )
    parser.add_argument(
        "-c",
        "--camera_id",
        type=str,
        choices=["Axis", "Logitech"],
        default="Logitech",
        help="Name of camera in _CCONF. If running for folder name of camera folder \
            in /home/data/profiling/.",
    )
    parser.add_argument(
        "-n",
        "--n_jobs",
        type=int,
        default=1,
        help="Number of processes to use in parallel.",
    )
    parser.add_argument(
        "-f",
        "--filepath",
        type=str,
        help="Run for single file.",
    )
    args = parser.parse_args()
    assert args.camera_id in _CCONF

    conf = _CCONF[args.camera_id]
    profiling_dataframe = None
    columns = ["freq", "exposure", "duty_cycle"]

    if args.filepath:
        print(f"Running for single file, using {args.camera_id} configuration.")
        assert os.path.isfile(args.filepath)
        profiling_dataframe = process_one_file(
            args.filepath, conf["dead_lines"], conf["pattern_every_n"]
        )
        prof_dct = profiling_dataframe.to_dict()
        print("### Profiling Info ###")
        for k, v in prof_dct.items():
            print(k, ":", v[0])
    else:
        video_folder = f"/home/data/profiling/{args.camera_id}"
        assert os.path.isdir(video_folder)

        output_folder = f"/home/data/results/profiling/"
        os.makedirs(output_folder, exist_ok=True)

        dir_walk = list(os.walk(video_folder))

        for i, (folder, subfolders, filenames) in enumerate(dir_walk):

            if len(filenames) == 0:
                continue

            files_to_analyse = [os.path.join(folder, _f) for _f in filenames]

            dfs = Parallel(n_jobs=args.n_jobs)(
                delayed(process_one_file)(
                    filepath, conf["dead_lines"], conf["pattern_every_n"]
                )
                for filepath in files_to_analyse
            )

            profiling_dataframe = pd.concat(
                [profiling_dataframe] + dfs, ignore_index=True
            )

            sys.stdout.write("\n")
            sys.stdout.flush()
            profiling_dataframe.to_csv(f"{output_folder}/{args.camera_id}.csv")

        profiling_dataframe.to_csv(f"{output_folder}/{args.camera_id}.csv")
        print(f"Output written to {output_folder}/{args.camera_id}.csv")
