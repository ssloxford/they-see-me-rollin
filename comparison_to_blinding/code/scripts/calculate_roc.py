import sys
import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import argparse

exposures = [32, 200]
duty_cycles = np.array([1.0, 20, 40])

def run_calculation(input_file):

    df = pd.read_csv(input_file)

    df["ssim"] = 1 - df["ssim"]
    df["ms_ssim"] = 1 - df["ms_ssim"]
    df["uqi"] = 1 - df["uqi"]
    metrics = ["ssim", "ms_ssim", "uqi"]

    for exposure_index, exposure in enumerate(exposures):
        for metric_index, metric in enumerate(metrics):

            # blinding setup for roc auc
            _df = df[(df.frequency == "blinding")]
            blind_pts = _df[metric].values
            blind_pts = blind_pts[blind_pts != 0.0]

            # plot legitimate stuff
            _df = df[(df.frequency == "0")]
            legit_pts = _df[metric].values
            legit_pts = legit_pts[legit_pts != 0.0]

            y_true = np.concatenate((np.ones(shape=blind_pts.shape[0]), np.zeros(shape=legit_pts.shape[0])))
            y_pred = np.concatenate((blind_pts, legit_pts), axis=0)
            s1 = roc_auc_score(y_true, y_pred)

            bfpr, btpr, bthresholds = roc_curve(y_true, y_pred)
            beer = brentq(lambda x: 1. - x - interp1d(bfpr, btpr)(x), 0., 1.)
            blind_thresh = interp1d(bfpr, bthresholds)(beer)

            #print(exposure, metric, "blinding", s1, "%.3f" % eer)        

            for duty_cycle_index, duty_cycle in enumerate(duty_cycles):
                _df = df[(df.frequency != "0") & (df.frequency != "blinding") & (df.frequency == "750") & (df.duty_cycle == duty_cycle) & (df.exposure == exposure)]
                if len(_df) < 1:
                    continue

                pts = _df[metric].values
                pts = pts[pts != 0.0]

                y_true = np.concatenate((np.ones(shape=pts.shape[0]), np.zeros(shape=legit_pts.shape[0])))
                y_pred = np.concatenate((pts, legit_pts), axis=0)
                s2 = roc_auc_score(y_true, y_pred)

                fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
                thresh = interp1d(fpr, thresholds)(eer)


                # print("fpr, tpr", "%.3f" % fpr[iof], "%.3f" % tpr[iof])

                iof = np.argwhere(fpr>0).flatten()[0]-1
                # biof = np.argwhere(bthresholds<=thresholds[iof]).flatten()[0]
                print(exposure, metric, "our_attack", "\t", duty_cycle, "\t", "%.3f" % s2, "fpr %.3f" % (fpr[iof]*100), "tpr %.3f" % (tpr[iof]*100))
                # print("bfpr", bfpr[biof]*100, "btpr", btpr[biof]*100)


                #print("bfpr", bfpr[biof])
                #print("btpr", btpr[biof]*100)

                #print("fpr", fpr[iof])
                #print("tpr", (tpr)[iof]*100)
            
if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description='Calculate the ROC-AUC.')
    parser.add_argument('--input_file', '-i', type=str, help='Path to the unaggregated CSV.', required=True)
    args = parser.parse_args()
    
    run_calculation(args.input_file)
