from gm2fr.analysis.Analyzer import Analyzer
from gm2fr.Histogram1D import Histogram1D
import matplotlib.pyplot as plt
import os

total = len(next(os.walk('/home/myhwang1999/gm2/gm2mt/results/m2-alg_scan'))[1])

# Create the analyzer object.
# Specify the input fast rotation signal file:
#   for ROOT, this is a tuple (filename, histogram).
#   for NumPy, this is just a filename.
# Also specify the name of the desired analysis output folder.
#   This folder will be placed at gm2fr/analysis/results/{yourFolderName}).
total = 7
analyzer = Analyzer(
#   files = "gm2fr/simulation/data/test/data.npz",
files = [f"../gm2fr/simulation/data/alg_scan_{i}_{total}/simulation.root" for i in range(1, total +1)],
tags = [f"subrun_{i}_{total}" for i in range(1, total +1)],
units = "us", # analyzer works in units of microseconds, but simulation was in microseconds
group = "alg_scan_narrow",
truth = "same"
)

# Run the analysis.
analyzer.analyze(
fit = None, # wiggle fit model: None / "two" / "five" / "nine"
t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
start = 4, # start time for cosine transform (us)
end = 200, # end time for cosine transform (us)
iterate = False,
model = "sinc", # background fit model: "parabola" / "sinc" / "error"
)



analyzer = Analyzer(
#   files = "gm2fr/simulation/data/test/data.npz",
  files = "../gm2fr/simulation/data/alg_source/simulation.root",
  tags = "alg_source",
  units = "us", # analyzer works in units of microseconds, but simulation was in microseconds
  truth = "same"
)

# Run the analysis.
analyzer.analyze(
  fit = None, # wiggle fit model: None / "two" / "five" / "nine"
  t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
  start = 4, # start time for cosine transform (us)
  end = 200, # end time for cosine transform (us)
  iterate = False,
  model = "sinc", # background fit model: "parabola" / "sinc" / "error"
)

lst = []
source_hist = Histogram1D.load("../gm2fr/analysis/results/alg_source/transform.root", "transform_f")

for subrun in range(1, total + 1):
  scan_hist = Histogram1D.load(f"../gm2fr/analysis/results/alg_scan_narrow/subrun_{subrun}_{total}/transform.root", "transform_f")
  chisq = sum(((source_hist.heights- scan_hist.heights)**2 / scan_hist.errors**2)) / len(source_hist.centers)
  lst.append(chisq)
print(lst)

plt.close('all')
fig, ax = plt.subplots()
# ax.plot([1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], lst)
ax.plot([2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4], lst)
ax.set_xlabel("alpha_width [mrad]")
ax.set_ylabel("chi2/ndf")
plt.show()