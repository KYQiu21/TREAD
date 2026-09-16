import matplotlib.pyplot as plt
import numpy as np
import itertools
from itertools import groupby

def plot_profile(score_profile, motif_ranges, save=False, save_path=None):
    
    plt.figure(figsize=(5, 5), dpi=300)
    plt.plot(score_profile)
    for start, end in motif_ranges:
        plt.axvspan(start, end, alpha=0.3)
    plt.xlabel("Residue")
    plt.ylabel("Residue score")
    plt.title("Residue-wise repeat score profile")
    if save:
        plt.savefig(save_path)
    plt.show()

    
def get_ranges(preds, cutoff1=0.5, min_len=15, cutoff2=0.5, frac2=0.5):

    preds = preds.flatten()
    above_threshold = preds > cutoff1
    peaks = []
    for k, g in groupby(enumerate(above_threshold), key=lambda x: x[1]):
        if k:
            g = list(g)
            if len(g) >= min_len:
                beg = g[0][0]
                end = g[0][0]+len(g)
                if len(np.where(preds[beg:end] > cutoff2)[0]) / len(g) >= frac2:
                    peaks.append((g[0][0], g[0][0]+len(g)))
    if len(peaks) > 0:
        return peaks
    else:
        return []
    

def analyze_profile(preds, 
                    cutoff1=0.5, min_len=15, cutoff2=0.5, frac2=0.5, 
                    plot=True, save_plot=False, save_path=None):
    
    motif_ranges = get_ranges(preds, cutoff1=cutoff1, min_len=min_len, cutoff2=cutoff2, frac2=frac2)
    print("Predicted motifs are located at: ", motif_ranges)
    
    if plot:
        plot_profile(preds, motif_ranges, save=save_plot, save_path=save_path)
        print("Plot done!")
        
    