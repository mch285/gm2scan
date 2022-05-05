import pathlib
import os
import matplotlib.pyplot as plt
from gm2fr.Histogram1D import Histogram1D
from gm2mt.Kicker import Kicker
import gm2mt.auxiliary as aux
from gm2mt.Ring import Ring
import gm2mt.Distributions as dist
from gm2mt.StateGenerator import StateGenerator
import numpy as np
import gm2fr.constants as const
# from gm2scan.Scanner import Scanner

fig, ax = plt.subplots()
output = "p_mean_none1e7_same"
subruns = 11
source_dir = "alg_source_1e7_same"
scan_dir = "p_mean_1E7_sames"
source = Histogram1D.load(f"results/{output}/analyses/src|{source_dir}/transform.root", "transform_f")
src_heights = source.heights
src_centers = source.centers
histograms = [Histogram1D.load(f"results/{output}/analyses/scan|{scan_dir}_{subrun}_{subruns}/transform.root", "transform_f") for subrun in range(1, subruns + 1)]

# Mask out the unphysical frequencies.
# source = source.normalize()
mask = const.physical(source.centers)
# hist_src = source.heights[mask]

scale = np.sum(source.heights * (source.width))
print(scale)

ax.plot(src_centers[mask], src_heights[mask], label = "Source")
for idx, hist in enumerate(histograms):
    ax.errorbar(hist.centers[mask], hist.heights[mask], yerr = hist.errors[mask], label = idx)

plt.savefig("fig.png")
    

# for idx, cov in enumerate(covs):
#     covs[idx] = cov[mask][:, mask]
# for idx, histogram in enumerate(histograms):
#     histograms[idx] = histogram[mask]