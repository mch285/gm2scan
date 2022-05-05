from gm2scan.Scanner import Scanner, FullScanner
from gm2fr.Histogram1D import Histogram1D
import matplotlib.pyplot as plt
import numpy as np

scanner = FullScanner(output = "fullscan", source_dir = 'alg_source_5e7_100k', scan_dir = "fullscan", verbose = False)
# scanner.scan()
# scanner.compare()
# scanner.fit(np.linspace(99.5, 100.5, 5), np.linspace(24805, 24850, 5))
scanner.fit()


# lst = np.zeros(shape = (5, 3))
# lst[1] = [1, 2, 3]
# print(lst)