from gm2fr.simulation.simulator import Simulator
import gm2fr.style as style
import ROOT as root
import os 
style.setStyle()


# inFile = root.TFile.Open("../gm2mt/results/s2-alg_source/momentum_time.root", "READ")
# joint = inFile.Get("joint")
# simulation = Simulator(
#   f"alg_source",
#   overwrite = True,
#   jointDistribution = joint,
#   kinematicsUnits = "frequency",
#   timeUnits = "nanoseconds"
# )

# simulation.simulate(muons = 1E7, end = 200)
# simulation.save()
# simulation.plot()

# dir = "alg_scan_tnorm_narrow"
# dir = "alg_scan_amean_wide"
dir = "alg_scan_pwide"

total = len(next(os.walk(f'/home/myhwang1999/gm2/gm2mt/results/m2-{dir}'))[1])

for i in range(total):
  print(f"Simulating subrun {i+1} of {total}")
  inFile = root.TFile.Open(f"/home/myhwang1999/gm2/gm2mt/results/m2-{dir}/subrun_{i+1}_{total}/momentum_time.root", "READ")
  joint = inFile.Get("joint")
  simulation = Simulator(
    f"{dir}_{i+1}_{total}",
    overwrite = True,
    jointDistribution = joint,
    kinematicsUnits = "frequency",
    timeUnits = "nanoseconds"
  )

  # Run the simulation, using the specified number of muons.
  simulation.simulate(muons = 1E7, end = 200)

  # Save and plot the results.
  simulation.save()
  simulation.plot()

# from gm2fr.analysis.Analyzer import Analyzer

# # Create the analyzer object.
# # Specify the input fast rotation signal file:
# #   for ROOT, this is a tuple (filename, histogram).
# #   for NumPy, this is just a filename.
# # Also specify the name of the desired analysis output folder.
# #   This folder will be placed at gm2fr/analysis/results/{yourFolderName}).
# analyzer = Analyzer(
#   # "../../simulation/testing/numpy/signal.npz",
#   ("../../simulation/data/testing/simulation.root", "signal"),
#   "testing",
#   units = "ns" # analyzer works in units of microseconds, but simulation was in nanoseconds -- I'll fix this soon
# )

# # Run the analysis.
# analyzer.analyze(
#   fit = None, # wiggle fit model: None / "two" / "five" / "nine"
#   t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
#   start = 4, # start time for cosine transform (us)
#   end = 200, # end time for cosine transform (us)
#   model = "sinc", # background fit model: "parabola" / "sinc" / "error"
# )