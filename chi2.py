# from gm2scan.Scanner import Scanner
import ROOT as root
from gm2fr.Simulator import Simulator
from gm2fr.Analyzer import Analyzer

# dir = "init_offset_width"
# dir = "p_mean_nonestr"
# dir = "p_mean"
# scanner = Scanner(output = "p_mean_nonestr", source_dir = "alg_source", scan_dir = dir, verbose = False)
# scanner.scan(muons = 1E7, fr_plot = False, model = None, fit = "both", show = False)
# scanner.compare(fit = "simple", show = False)

inFile = root.TFile.Open("../gm2mt/results/m2-alg_scan_pwide/subrun_2_5/momentum_time.root", "READ")
joint = inFile.Get("joint")

for i in range(5):

    simulation = Simulator(
        f"test{i}",
        overwrite = True,
        jointDistribution = joint,
        kinematicsUnits = "frequency",
        timeUnits = "nanoseconds"
    )
    simulation.simulate(muons = 1E7, end = 200)
    simulation.save()
    simulation.plot()

# perform chisq fits:

analyzer = Analyzer(
    files = [f"..gm2fr/simulation/data/test{i}/simulation.root" for i in range(5)],
    tags = [f"test{i}" for i in range(5)],
    units = "us", # analyzer works in units of microseconds, but simulation was in microseconds
    group = "chi2_test",
    truth = "same")

analyzer.analyze(
    fit = None, # wiggle fit model: None / "two" / "five" / "nine"
    # t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
    t0 = 0, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
    # start = 4, # start time for cosine transform (us)
    start = 0, # start time for cosine transform (us)
    end = 200, # end time for cosine transform (us)
    iterate = False,
    model = None)