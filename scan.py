from gm2scan.Scanner import Scanner
import numpy as np

pmeans = np.linspace(99.5, 100.5, 5)
tnorms = np.linspace(24805, 24850, 5)

# dir = "init_offset_width"
# dir = "p_mean_nonestr"
# dir = "p_mean"
# scanner = Scanner(output = "p_mean_none1e7_same_t0on", source_dir = "alg_source_1e7_same", scan_dir = "p_mean_1E7_sames", verbose = False)
# scanner = Scanner(output = "p_mean", source_dir = "alg_source", scan_dir = "p_mean", verbose = False)
# scanner = Scanner(output = "p_mean_percent_sinc_5E7", source_dir = "alg_source_1e7", scan_dir = "p_mean_percent", verbose = False)
# scanner = Scanner(output = "p_mean_none1e7_again", source_dir = "alg_source_1e7", scan_dir = "p_mean_1E7", verbose = False)
# scanner = Scanner(output = "p_std_5E8", source_dir = "alg_source_1e7_100k", scan_dir = "p_std", verbose = False)
# scanner = Scanner(output = "tnorm", source_dir = "alg_source_1e7", scan_dir = "tnorm", verbose = False)
# scanner = Scanner(output = "alphamean", source_dir = 'alg_source_1e7_100k', scan_dir = "alphamean", verbose = False)
# scanner = Scanner(output = "alphamean_wide", source_dir = 'alg_source_1e7_100k', scan_dir = "alphamean_wide", verbose = False)
scanner = Scanner(output = "xmean", source_dir = 'alg_source_1e7_100k', scan_dir = "xmean", verbose = False)
# scanner = Scanner(output = "alphastd", source_dir = 'alg_source_1e7_100k', scan_dir = "alphastd", verbose = False)
# scanner.scan(muons = 1E8, fr_plot = False, model = "sinc", fit = "cov", show = False)
scanner.fit(np.linspace(99.5, 100.5, 5), np.linspace(24805, 24850, 5))
# scanner.compare(fit = "cov", show = False)

