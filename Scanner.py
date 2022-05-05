from inspect import Attribute
from gm2fr.simulation.simulator import Simulator
from gm2fr.analysis.Analyzer import Analyzer
from gm2fr.Histogram1D import Histogram1D
import gm2fr.style as style
import gm2fr.constants as const
from gm2scan.Errors import FitError

from gm2mt.Ring import Ring
from gm2mt.StateGenerator import StateGenerator
from gm2mt.Plotter import Plotter
import gm2mt.auxiliary as aux

import matplotlib.pyplot as plt
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import numpy.linalg as la
import numpy as np
import ROOT as root
import pathlib
import contextlib
import os
import shutil
import time
import logging

def _verbose(func):
    """Decorator to suppress terminal printing and warnings (counting them).
    Level 1 ('v'): suppress both unneeded prints and warnings.
    Level 2 ('vv', 'print only'): Show prints but not warnings.
    Level 3 ('vvv'): Show prints and warnings.
    """
    def deco(*args, **kwargs):
        if args[0].verbose in [3, "vvv", "all", True]:
            value = func(*args, **kwargs)
        elif args[0].verbose in [2, "vv", "print only"]:
            with warnings.catch_warnings(record = True) as w:
                value = func(*args, **kwargs)
                args[0].warnings = w
        elif args[0].verbose in ["warnings only"]:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                value = func(*args, **kwargs)
        elif args[0].verbose in [1, "v", False]:
            with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
                with warnings.catch_warnings(record = True) as w:
                    value = func(*args, **kwargs)
                    args[0].warnings = w
        return value
    return deco

class Scanner:
    def __init__(self, output, source_dir, scan_dir, verbose = False):
        self.output = output
        print(f"Working in directory /{self.output}.")
        self.gm2path = str(pathlib.Path(os.path.abspath(__file__)).parents[1])
        self.source_dir = source_dir
        self.scan_dir = scan_dir
        try:
            self.subruns = len(next(os.walk(f"{self.gm2path}/gm2mt/results/m2-{self.scan_dir}"))[1])
        except StopIteration:
            pass
        if verbose in [1, 2, 3, "v", "vv", "vvv", "print only", "warnings only", "all", True, False]:
            self.verbose = verbose
        else:
            raise ValueError(f"Your verbose level '{verbose}' is not recognized.")
        self.warnings = []

        style.setStyle()

    def scan(self, muons = 1E7, fr_plot = False, model = "sinc", fit = "both", show = False):
        begin = time.perf_counter()
        self.simulate(muons, fr_plot)
        self.analyze(model)
        self.compare(fit, show)
        print(f"Full scan completed in {time.perf_counter() - begin} s.")
    
    def simulate(self, muons = 1E7, fr_plot = False):
        self._prep()
        begin = time.perf_counter()
        print(f"Simulating FR signal for source distribution '/s2-{self.source_dir}'...  ", end = "")
        inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/s2-{self.source_dir}/momentum_time.root", "READ")
        joint = inFile.Get("joint")
        self._simulate(f"src|{self.source_dir}", joint, muons, fr_plot)
        print("\rSource distribution FR signal simulated.                                  ")

        print(f"Currently simulating FR signals for scan distributions '/m2-{self.scan_dir}'...  ", end = "")
        for subrun in range(1, self.subruns + 1):
            print(f"\rCurrently simulating FR signals for scan distributions '/m2-{self.scan_dir}'... {subrun}/{self.subruns}.", end = "")
            # inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/m2-{self.scan_dir}/subrun_{subrun}_{self.subruns}/momentum_time.root", "READ")
            inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/m2-{self.scan_dir}/subrun_{subrun}_{self.subruns}/momentum_time.root", "READ")
            joint = inFile.Get("joint")
            self._simulate(f"scan|{self.scan_dir}_{subrun}_{self.subruns}", joint, muons, fr_plot)
        print(f"\r{self.subruns}/{self.subruns} scan distribution FR signals simulated.  Finished in {time.perf_counter() - begin} s.")

        self._move(f"{self.gm2path}/gm2fr/simulation/data/src|{self.source_dir}", f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")
        self._move([f"{self.gm2path}/gm2fr/simulation/data/scan|{self.scan_dir}_{subrun}_{self.subruns}" for subrun in range(1, self.subruns + 1)], f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")
        self._untemp()
    
    @_verbose
    def _simulate(self, dir, joint_dist, muons, fr_plot):
        simulation = Simulator(
            dir,
            overwrite = True,
            jointDistribution = joint_dist,
            kinematicsUnits = "frequency",
            timeUnits = "nanoseconds"
        )
        simulation.simulate(muons = muons, end = 200)
        simulation.save()
        if fr_plot:
            simulation.plot()

    def _move(self, origin, destination):
        origin = [origin] if not isinstance(origin, (list, np.ndarray)) else origin
        for dir in origin:
            shutil.move(dir, destination)

    def analyze(self, model = "sinc"):
        print("Analyzing source FR signal...  ", end = "")
        begin = time.perf_counter()
        self._analyze(files = f"{self.gm2path}/gm2scan/results/{self.output}/simulations/src|{self.source_dir}/simulation.root",
            tags = f"src|{self.source_dir}",
            group = None,
            model = model)
        print("Done.")
        print("Analyzing scan FR signals...  ", end = "")
        self._analyze(files = [f"{self.gm2path}/gm2scan/results/{self.output}/simulations/scan|{self.scan_dir}_{subrun}_{self.subruns}/simulation.root" for subrun in range(1, self.subruns +1)],
            tags = [f"scan|{self.scan_dir}_{subrun}_{self.subruns}" for subrun in range(1, self.subruns +1)],
            group = f"{self.scan_dir}|analyses",
            model = model)
        print(f"Done. Analyses completed in {time.perf_counter() - begin} s.")
        
        self._move(f"{self.gm2path}/gm2fr/analysis/results/{self.scan_dir}|analyses", f"{self.gm2path}/gm2scan/results/{self.output}")
        os.rename(f"{self.gm2path}/gm2scan/results/{self.output}/{self.scan_dir}|analyses", f"{self.gm2path}/gm2scan/results/{self.output}/analyses")
        self._move(f"{self.gm2path}/gm2fr/analysis/results/src|{self.source_dir}", f"{self.gm2path}/gm2scan/results/{self.output}/analyses")
       
    @_verbose
    def _analyze(self, files, tags, group, model):
        analyzer = Analyzer(
            files = files,
            tags = tags,
            units = "us", # analyzer works in units of microseconds, but simulation was in microseconds
            group = group,
            truth = "same")

        if model is None:
            start = 0
        elif model == "sinc":
            start = 4
        else:
            raise SyntaxError("model is not a sinc or None")
        
        analyzer.analyze(
            fit = None, # wiggle fit model: None / "two" / "five" / "nine"
            # t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
            t0 = 0, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
            # start = 4, # start time for cosine transform (us)
            start = start, # start time for cosine transform (us)
            end = 200, # end time for cosine transform (us)
            iterate = False,
            model = model)

    def compare(self, fit = "simple", custom = None, show = False):
        self.var, self.var_values = self.__extract_var()
        if not custom:
            self.truth, self.var, self.var_values = Scanner.__translate(self.source_dir, self.var, self.var_values)
        else:
            self.var, self.var_values = Scanner.__custom(custom, self.var_values)

        print(f"truth is {self.truth} and type is {type(self.truth)}")
        print(self.var)
        print(self.var_values)

        if fit == "simple":
            chi2 = self.__simple()
        elif fit == "cov" or fit == "full":
            try:
                chi2 = self.__cov()
            except FitError as e:
                warnings.warn(f"Warning: encountered {repr(e)}; switching to simple chi-square comparison.", category = UserWarning, stacklevel = 2)
                fit = "simple"
                chi2 = self.__simple()
        print(f"xvals: {self.var_values}")
        print(f"chi2: {chi2}")
        if self.var_values[0] > self.var_values[-1]:
            self.var_values.reverse()
            chi2.reverse()
        
        print(f"new xvals: {self.var_values}")
        print(f"new chi2: {chi2}")

        self.guess, self.min_chi2, self.uncertainty = self.__parfit(self.var_values, chi2)
        print(f"ndf: {self.ndf}")
        self.plot(chi2, fit, show)
        
        # elif fit == "both":
        #     chi2 = self.__simple()
        #     self.plot(chi2, "simple", show)
        #     chi2 = self.__cov()
        #     self.plot(chi2, "cov", show)

        self.save(chi2)
          
    def __simple(self):
        chi2 = []
        hist_src, histograms, errors = self.__load_transforms_simple()
        
        for hist, error in zip(histograms, errors):
            # chi2.append(sum(((hist_src - hist) / error) ** 2) / len(hist))
            chi2.append(sum(((hist_src - hist) / error) ** 2))
            self.ndf = len(hist)
        return chi2

    def __load_transforms_simple(self):
        source = Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/src|{self.source_dir}/transform.root", "transform_f")
        histograms = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.root", "transform_f").heights for subrun in range(1, self.subruns + 1)]
        errors = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.root", "transform_f").errors for subrun in range(1, self.subruns + 1)]

        mask = const.physical(source.centers)
        hist_src = source.heights[mask]
        for idx, error in enumerate(errors):
            errors[idx] = error[mask]
        for idx, histogram in enumerate(histograms):
            histograms[idx] = histogram[mask]

        return hist_src, histograms, errors
    
    def __cov(self):
        chi2 = []
        hist_src, histograms, covs = self.__load_transforms_cov()

        for hist, cov in zip(histograms, covs):
            residuals = hist - hist_src
            # chi2.append((residuals.transpose() @ la.inv(cov) @ residuals) / len(hist))
            chi2.append((residuals.transpose() @ la.inv(cov) @ residuals))
            self.ndf = len(hist)
        return chi2

    def __load_transforms_cov(self):
        source = Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/src|{self.source_dir}/transform.root", "transform_f").normalize()
        histograms = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.npz").normalize().heights for subrun in range(1, self.subruns + 1)]
        # covs = [np.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.npz")["cov"] for subrun in range(1, self.subruns + 1)]
        # covs = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.root", "transform_f").normalize().cov for subrun in range(1, self.subruns + 1)]
        covs = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.npz").normalize().cov for subrun in range(1, self.subruns + 1)]
        
        src_states = np.load(f"{self.gm2path}/gm2mt/results/s2-{self.source_dir}/final_states.npz")
        f = aux.p_to_f(aux.state_to_mom_cyl(r = src_states["r"], vr = src_states["vr"], vphi = src_states["vphi"]), n = 0.108)
        f_s = Plotter._mask_lost(src_states["lost"], f)
        h, edges = np.histogram(f_s, bins = source.edges)
        # print(len(h))
        width = 2
        errors = np.sqrt(h) / (len(f_s) * width)
        assert sum(h) == len(f_s)
        # print(errors)
        src_new = np.diag(np.square(errors))
        # print(src_new)
        # print(len(f_s))
        # print(source.edges)

        scan_states = [np.load(f"{self.gm2path}/gm2mt/results/m2-{self.scan_dir}/subrun_{subrun}_{self.subruns}/final_states.npz") for subrun in range(1, self.subruns + 1)]
        scan_news = []
        for states in scan_states:
            f = aux.p_to_f(aux.state_to_mom_cyl(r = states["r"], vr = states["vr"], vphi = states["vphi"]), n = 0.108)
            f_s = Plotter._mask_lost(states["lost"], f)
            h, edges = np.histogram(f_s, bins = source.edges)
            errors = np.sqrt(h) / (len(f_s) * width)
            scan_news.append(np.diag(np.square(errors)))
        
        #Ensure that the covariance matrix is square.
        for cov in covs:
            if len(cov.shape) != 2:
                raise FitError(f"Covariance matrix is not 2D; has shape {cov.shape}")
            if cov.shape[0] != cov.shape[1]:
                raise FitError(f"Covariance matrix is not square; has shape {cov.shape}")
            for histogram in histograms:
                if cov.shape[0] != len(histogram):
                    raise FitError(f"Covariance matrix length {len(histogram)} do not match unmasked histograms, shape {cov.shape}")
        
        # Mask out the unphysical frequencies.
        mask = const.physical(source.centers)
        hist_src = source.heights[mask]
        for idx, cov in enumerate(covs):
            covs[idx] = cov[mask][:, mask]
        for idx, histogram in enumerate(histograms):
            histograms[idx] = histogram[mask]
        
        ref_cov = Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/src|{self.source_dir}/transform.npz").normalize().cov
        combined_covs = [cov + src_new[mask][:, mask] + ref_cov[mask][:, mask] + scan_new[mask][:, mask] for cov, scan_new in zip(covs, scan_news)]
        
        # combined_covs = [cov + ref_cov[mask][:, mask] for cov, scan_new in zip(covs, scan_news)]
        # return hist_src, histograms, covs
        return hist_src, histograms, combined_covs

    def __parfit(self, xvals, chi2):
        idx = np.argmin(chi2)
        print(f"min indx: {idx}")
        # print(xvals)
        print(f"chi2s: {chi2}")
        a, b, c = np.polyfit(xvals[idx-1:idx+2], chi2[idx-1:idx+2], deg = 2)
        guess, min_chi2, uncertainty = -b / (2 * a), c - (b**2 / 4 / a), 1 / np.sqrt(a)
        return guess, min_chi2, uncertainty
    
    def plot(self, chi2, fit, show):
        plt.close('all')
        fig, ax = plt.subplots()

        ax.plot(self.var_values, chi2)
        ax.set_xlabel(self.var)
        
        # ax.plot(self.var_values, chi2)
        # ax.set_xlabel(self.var)
        ax.set_title(r"$\chi^2$ comparison $(n_{df} = $" + f"{self.ndf}$)$")
        ax.set_ylim(bottom = 0)
        ax.text(0.5, 0.7, min(chi2), ha = 'center', va = 'center', transform = ax.transAxes)
        ax.text(0.5, 0.9, f"guess: ${self.guess:3f} \pm {self.uncertainty:.3f}$, fit chi2: {self.min_chi2:.2f}", ha = 'center', va = 'center', transform = ax.transAxes)

        pdf = PdfPages(f"{self.gm2path}/gm2scan/results/{self.output}/chi2_{fit}.pdf")
        pdf.savefig(fig)
        pdf.close()

        if show:
            plt.show()

    def _prep(self):
        prefix = "&TMP|"
        self.tmp = f"{prefix}{self.output}"

        if pathlib.Path(f"{self.gm2path}/gm2scan/results/{self.tmp}").is_dir():
            print("The temporary directory already exists. ", end = "")
            while True:
                clear = input("Clear? [y/n] ")
                if clear in ["y", "Y", "N", "n"]:
                    break
                else:
                    print("Invalid clear option. ", end = "")
            if clear in ["y", "Y"]:
                print("Clearing existing temporary directory... ", end = "")
                shutil.rmtree(f"{self.gm2path}/gm2scan/results/{self.tmp}")
                print("Cleared.")
            else:
                raise KeyboardInterrupt("Temporary directory not cleared, simulation ending.")
        os.mkdir(f"{self.gm2path}/gm2scan/results/{self.tmp}")
        os.mkdir(f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")
    
    def _untemp(self):
        if pathlib.Path(f"{self.gm2path}/gm2scan/results/{self.output}").is_dir():
            print(f"Overwriting /{self.output}.")
            shutil.rmtree(f"{self.gm2path}/gm2scan/results/{self.output}")
        else:
            print(f"Creating /{self.output}.")
        os.rename(f"{self.gm2path}/gm2scan/results/{self.tmp}", f"{self.gm2path}/gm2scan/results/{self.output}")
    
    def __extract_var(self):
        rg = Ring.load(dir = f"m2-{self.scan_dir}")
        sg = StateGenerator.load(dir = f"m2-{self.scan_dir}")
        if sg.plex == "multiplex":
            subruns, labels = sg.search()
        elif rg.mode == "multiplex":
            subruns, labels = rg.search()
        else:
            raise RuntimeError("Neither the ring nor the state generator are listed as multiplex mode...")
        
        # ylabel = labels[0].split("=")[0].split(": ")[4:]
        ylabel = labels[0].split("=")[0]
        yvals = [float(label.split("=")[1]) for label in labels]

        return ylabel, yvals

    def save(self, chi2):
        np.savetxt(fname = f"{self.gm2path}/gm2scan/results/{self.output}/chi2.txt", X = chi2)

    @staticmethod
    def __translate(source_dir, ylabel, yvals):
        if ": " in ylabel:
            # The changing variable is in the state generator.
            variable, parameter = ylabel.split(": ")
            if parameter == "mean":
                prefix = "\mu_"
            elif parameter == "std":
                prefix = "\sigma_"
            elif parameter == "value":
                prefix = ""

            if variable == "x":
                label = rf"${prefix}x$ [mm]"
            elif variable == "p":
                label = rf"${prefix}p$ [%]"
                yvals /= (aux.p_magic / 100)
            elif variable == "f":
                label = rf"${prefix}f$ [kHz]"
            elif variable == "r_co":
                label = rf"${prefix}" + r"{x_e}$ [mm]"
            elif variable == "alpha":
                label = rf"${prefix}$ [mrad]"
                yvals /= 1000
            elif variable == "phi_0":
                label = rf"${prefix}" + r"{\phi_0}$ [mrad]"
            elif variable == "t":
                if parameter == "zero":
                    label = r"$t_a$ [ns]"
                elif parameter == "dir":
                    label = "Filename"
            
            sg = StateGenerator.load(f"s2-{source_dir}")
            truth = aux.delist(getattr(getattr(sg, variable), parameter))

        else: # Ring or kicker multiplex
            object = "rg"
            variable = ylabel
            if variable == "b_norm":
                label = r"$B_{max}$ [G]"
            elif variable == "t_norm":
                label = r"$t_{norm}$ [ns]"
            elif variable == "kick_max":
                label = "Number of kicks"
            elif variable == "kick_num":
                label = "Number of kicker segments"
            elif variable == "quad_num":
                label = "Number of ESQ segments"
            elif variable == "n":
                label = r"Field index $n$"
            elif variable == "collimators":
                label = "Collimation type"
            elif variable == "b_nom":
                label = "Nominal magnetic field strength [T]"
            
            rg = Ring.load(f"s2-{source_dir}")
            try:
                truth = aux.delist(getattr(rg, variable))
            except AttributeError:
                truth = aux.delist(getattr(rg.b_k, variable))
            
 
        return truth, label, yvals

    @staticmethod
    def __custom(custom, yvalues):
        ylabel, func = custom
        post = [func(yval) for yval in yvalues]

        return ylabel, post

    @staticmethod
    def load(dir):
        return np.loadtxt(fname = f"{pathlib.Path(os.path.abspath(__file__)).parents[1]}/gm2scan/results/{dir}/chi2.txt")

    @classmethod
    def remake(dir):
        pass



class FullScanner(Scanner):
    def __init__(self, output, source_dir, scan_dir, verbose = False):
        super().__init__(output, source_dir, scan_dir, verbose)
        self._calc_indices(scan_dir)

    def _calc_indices(self, dir):
        subdir_list = next(os.walk(f"{self.gm2path}/gm2mt/results/{dir}"))[1]
        if subdir_list[0].startswith(f"s2-{dir}_"):
            self.suffixes = [subdir[len(dir)+4:] for subdir in subdir_list] # removes s2 prefix, the dir, and the remaining _
            self.indices = [tuple([int(num) for num in subdir.split("_")]) for subdir in self.suffixes]
            length = len(self.indices[0])
            self.maxima = []
            for i in range(length):
                self.maxima.append(max([idx[i] for idx in self.indices]))
            self.maxima = tuple(self.maxima)
        else:
            raise ValueError("Subdirectories couldn't be parsed.")

    def scan(self, muons = 1E8, fr_plot = False, model = "sinc"):
        begin = time.perf_counter()
        self.simulate(muons, fr_plot)
        self.analyze(model)
        self.compare()
        self.fit()
        print(f"Full scan completed in {time.perf_counter() - begin} s.")

    def simulate(self, muons = 1E8, fr_plot = False):
        self._prep()
        begin = time.perf_counter()
        print(f"Simulating FR signal for source distribution '/s2-{self.source_dir}'...  ", end = "")
        inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/s2-{self.source_dir}/momentum_time.root", "READ")
        joint = inFile.Get("joint")
        self._simulate(f"src|{self.source_dir}", joint, muons, fr_plot)
        print("\rSource distribution FR signal simulated.                                  ")

        print(f"Currently simulating FR signals for scan distributions '/{self.scan_dir}'...  ", end = "")
        for current_count, suffix in enumerate(self.suffixes):
            print(f"\rCurrently simulating FR signals for scan distributions '/{self.scan_dir}'... {current_count + 1}/{len(self.indices)}.", end = "")
            # inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/m2-{self.scan_dir}/subrun_{subrun}_{self.subruns}/momentum_time.root", "READ")
            inFile = root.TFile.Open(f"{self.gm2path}/gm2mt/results/{self.scan_dir}/s2-{self.scan_dir}_{suffix}/momentum_time.root", "READ")
            joint = inFile.Get("joint")
            self._simulate(f"scan|{self.scan_dir}_{suffix}", joint, muons, fr_plot)
        print(f"\r{len(self.indices)}/{len(self.indices)} scan distribution FR signals simulated.  Finished in {time.perf_counter() - begin} s.")

        self._move(f"{self.gm2path}/gm2fr/simulation/data/src|{self.source_dir}", f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")
        self._move([f"{self.gm2path}/gm2fr/simulation/data/scan|{self.scan_dir}_{suffix}" for suffix in self.suffixes], f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")
        self._untemp()

    # def _move(self, origin, destination):
    #     origin = [origin] if not isinstance(origin, (list, np.ndarray)) else origin
    #     for dir in origin:
    #         shutil.move(dir, destination)

    def analyze(self, model = "sinc"):
        print("Analyzing source FR signal...  ", end = "")
        begin = time.perf_counter()
        self._analyze(files = f"{self.gm2path}/gm2scan/results/{self.output}/simulations/src|{self.source_dir}/simulation.root",
            tags = f"src|{self.source_dir}",
            group = None,
            model = model)
        print("Done.")
        print("Analyzing scan FR signals...  ", end = "")
        self._analyze(files = [f"{self.gm2path}/gm2scan/results/{self.output}/simulations/scan|{self.scan_dir}_{suffix}/simulation.root" for suffix in self.suffixes],
            tags = [f"scan|{self.scan_dir}_{suffix}" for suffix in self.suffixes],
            group = f"{self.scan_dir}|analyses",
            model = model)
        print(f"Done. Analyses completed in {time.perf_counter() - begin} s.")
        
        self._move(f"{self.gm2path}/gm2fr/analysis/results/{self.scan_dir}|analyses", f"{self.gm2path}/gm2scan/results/{self.output}")
        os.rename(f"{self.gm2path}/gm2scan/results/{self.output}/{self.scan_dir}|analyses", f"{self.gm2path}/gm2scan/results/{self.output}/analyses")
        self._move(f"{self.gm2path}/gm2fr/analysis/results/src|{self.source_dir}", f"{self.gm2path}/gm2scan/results/{self.output}/analyses")
        
    # @_verbose
    # def _analyze(self, files, tags, group, model):
    #     analyzer = Analyzer(
    #         files = files,
    #         tags = tags,
    #         units = "us", # analyzer works in units of microseconds, but simulation was in microseconds
    #         group = group,
    #         truth = "same")

    #     if model is None:
    #         start = 0
    #     elif model == "sinc":
    #         start = 4
    #     else:
    #         raise SyntaxError("model is not a sinc or None")
        
    #     analyzer.analyze(
    #         fit = None, # wiggle fit model: None / "two" / "five" / "nine"
    #         # t0 = 0.110, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
    #         t0 = 0, # supply an initial t0 guess (in us), which will be optimized within +/- 15 ns (default from simulation is 0.74*T_magic ~ 110 ns)
    #         # start = 4, # start time for cosine transform (us)
    #         start = start, # start time for cosine transform (us)
    #         end = 200, # end time for cosine transform (us)
    #         iterate = False,
    #         model = model)

    def compare(self):
        # self.var, self.var_values = self.__extract_var()
        # if not custom:
        #     self.var, self.var_values = Scanner.__translate(self.source_dir, self.var, self.var_values)
        # else:
        #     self.var, self.var_values = Scanner.__custom(custom, self.var_values)

        # print(f"truth is {self.truth} and type is {type(self.truth)}")
        # print(self.var)
        # print(self.var_values)

        chi2 = self._cov()
        print(f"chi2s: {chi2}")
        print(f"minimum chi2: {chi2.min()}")
        print(f"location of minimum chi2: {np.unravel_index(chi2.argmin(), chi2.shape)}")

        np.save(f"{self.gm2path}/gm2scan/results/{self.output}/chi2.npy", chi2)
        # if self.var_values[0] > self.var_values[-1]:
        #     self.var_values.reverse()
        #     chi2.reverse()
        
        # print(f"new xvals: {self.var_values}")
        # print(f"new chi2: {chi2}")

        # self.guess, self.min_chi2, self.uncertainty = self.__parfit(self.var_values, chi2)
        # print(f"ndf: {self.ndf}")
        # self.plot(chi2, fit, show)
        
        # elif fit == "both":
        #     chi2 = self.__simple()
        #     self.plot(chi2, "simple", show)
        #     chi2 = self.__cov()
        #     self.plot(chi2, "cov", show)

        # self.save(chi2)

    def _cov(self):
        chi2 = np.zeros(shape = self.maxima)
        residuals, covs = self._load_transforms_cov()

        for idx in self.indices:
            arr_idx = tuple([num - 1 for num in idx])
            residual = residuals[idx]
            cov = covs[idx]
            chi2[arr_idx] = residual.transpose() @ la.inv(cov) @ residual

        return chi2

    def _load_transforms_cov(self):
        source = Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/src|{self.source_dir}/transform.npz").normalize()
        # source = Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/src|{self.source_dir}/transform.root", "transform_f").normalize()
        # histograms = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.npz").normalize().heights for subrun in range(1, self.subruns + 1)]
        # covs = [np.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.npz")["cov"] for subrun in range(1, self.subruns + 1)]
        # covs = [Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_{subrun}_{self.subruns}/transform.root", "transform_f").normalize().cov for subrun in range(1, self.subruns + 1)]

        scan_hists = {idx: Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_" + "_".join([str(num) for num in idx]) + "/transform.npz").normalize().heights for idx in self.indices}
        scan_fr_covs = {idx: Histogram1D.load(f"{self.gm2path}/gm2scan/results/{self.output}/analyses/scan|{self.scan_dir}_" + "_".join([str(num) for num in idx]) + "/transform.npz").normalize().cov for idx in self.indices}
        scan_mt_covs = {idx: np.load(f"{self.gm2path}/gm2mt/results/{self.output}/s2-{self.output}_" + "_".join([str(num) for num in idx]) + "/f_dist.npz")['cov'] for idx in self.indices}
        
        src_hist = source.heights
        src_fr_cov = source.cov
        try:
            src_mt_cov = np.load(f"{self.gm2path}/gm2mt/results/s2-{self.source_dir}/f_dist.npz")['cov']
            print("Extracted source MT covariance from 'f_dist.npz' file.")
        except FileNotFoundError:
            print("Source MT covariance matrix not found; setting to zero.")
            src_mt_cov = np.zeros(shape = (len(source.centers), len(source.centers)))

        
        # Mask out the unphysical frequencies.
        mask = const.physical(source.centers)
        scan_hists, scan_fr_covs, scan_mt_covs, src_hist, src_fr_cov, src_mt_cov = self._mask(mask, scan_hists, scan_fr_covs, scan_mt_covs, src_hist, src_fr_cov, src_mt_cov)
        self.ndf = len(src_hist)

        for cov in scan_fr_covs.values():
            assert cov.shape == (43, 43)
        for cov in scan_mt_covs.values():
            assert cov.shape == (43, 43)
        assert src_fr_cov.shape == (43, 43)
        assert src_mt_cov.shape == (43, 43)

        # combined_covs = {idx: scan_fr_covs[idx] + scan_mt_covs[idx] + src_fr_cov + src_mt_cov for idx in self.indices}
        combined_covs = {idx: scan_fr_covs[idx] + src_fr_cov for idx in self.indices}
        residuals = {idx: scan_hists[idx] - src_hist for idx in self.indices}
        
        return residuals, combined_covs

    def _mask(self, mask, *objects):
        objects_list = list(objects)
        for index, obj in enumerate(objects_list):
            if isinstance(obj, dict):
                for key, array in obj.items():
                    obj[key] = self.__mask(mask, array)
            elif isinstance(obj, np.ndarray):
                objects_list[index] = self.__mask(mask, obj)
            else:
                raise ValueError("oops")
        return objects_list

    def __mask(self, mask, array):
        # print(array)
        if len(array.shape) == 2:
            return array[mask][:, mask]
        elif len(array.shape) == 1:
            return array[mask]
        else:
            raise SyntaxError(f"Something's gone wrong with the masking; array shape is {array.shape}")

    def fit(self):
        chi2 = np.load(f"{self.gm2path}/gm2scan/results/{self.output}/chi2.npy")
        params = np.load(f"{self.gm2path}/gm2mt/results/{self.scan_dir}/params.npz")

        with open(f"{self.gm2path}/gm2mt/results/{self.scan_dir}/directory.txt", 'r') as f:
            lines = f.read().splitlines()
        axis_labels = [0] * len(chi2.shape)

        for line in lines:
            axis_num, variable = line.split(" ")
            axis_labels[int(axis_num)] = params[variable]

        self._fit(chi2, axis_labels)

    def _fit(self, chi2, axis_labels):
        assert len(chi2.shape) == len(axis_labels)
        location = np.unravel_index(chi2.argmin(), chi2.shape)
        min_chi2 = chi2.min()
        
        fit_data = np.zeros(shape = (len(axis_labels), 3))

        print(f"min: {chi2.argmin()}, located at: {location}")
        for idx in range(len(location)):
            labels = axis_labels[idx]
            loc = list(location)
            mid_label = labels[loc[idx]]

            loc[idx] -= 1
            left_label = labels[loc[idx]]
            left_idx = tuple(loc)
            
            loc[idx] += 2
            right_label = labels[loc[idx]]
            right_idx = tuple(loc)
            
            left_chi2 = chi2[left_idx]
            right_chi2 = chi2[right_idx]
            chi2_slice = [left_chi2, min_chi2, right_chi2]
            label_slice = [left_label, mid_label, right_label]

            guess, min_chi2, uncertainty = self._parfit(chi2_slice, label_slice)
            print(f"chi2slice: {chi2_slice}, labelslice: {label_slice}")
            print(f"axis {idx}: guess {guess}, min_chi2 {min_chi2}, unc {uncertainty}")
            fit_data[idx] = [guess, min_chi2, uncertainty]
        
        np.save(f"{self.gm2path}/gm2scan/results/{self.output}/fit.npy", fit_data)
      
    def _parfit(self, chi2_slice, label_slice):
        a, b, c = np.polyfit(label_slice, chi2_slice, deg = 2)
        guess, min_chi2, uncertainty = -b / (2 * a), c - (b**2 / 4 / a), 1 / np.sqrt(a)
        return guess, min_chi2, uncertainty

    def plot(self, chi2, fit, show):
        plt.close('all')
        fig, ax = plt.subplots()

        ax.plot(self.var_values, chi2)
        ax.set_xlabel(self.var)
        
        # ax.plot(self.var_values, chi2)
        # ax.set_xlabel(self.var)
        ax.set_title(r"$\chi^2$ comparison $(n_{df} = $" + f"{self.ndf}$)$")
        ax.set_ylim(bottom = 0)
        ax.text(0.5, 0.7, min(chi2), ha = 'center', va = 'center', transform = ax.transAxes)
        ax.text(0.5, 0.9, f"guess: ${self.guess:3f} \pm {self.uncertainty:.3f}$, fit chi2: {self.min_chi2:.2f}", ha = 'center', va = 'center', transform = ax.transAxes)

        pdf = PdfPages(f"{self.gm2path}/gm2scan/results/{self.output}/chi2_{fit}.pdf")
        pdf.savefig(fig)
        pdf.close()

        if show:
            plt.show()

    def _prep(self):
        prefix = "&TMP|"
        self.tmp = f"{prefix}{self.output}"

        if pathlib.Path(f"{self.gm2path}/gm2scan/results/{self.tmp}").is_dir():
            print("The temporary directory already exists. ", end = "")
            while True:
                clear = input("Clear? [y/n] ")
                if clear in ["y", "Y", "N", "n"]:
                    break
                else:
                    print("Invalid clear option. ", end = "")
            if clear in ["y", "Y"]:
                print("Clearing existing temporary directory... ", end = "")
                shutil.rmtree(f"{self.gm2path}/gm2scan/results/{self.tmp}")
                print("Cleared.")
            else:
                raise KeyboardInterrupt("Temporary directory not cleared, simulation ending.")
        os.mkdir(f"{self.gm2path}/gm2scan/results/{self.tmp}")
        os.mkdir(f"{self.gm2path}/gm2scan/results/{self.tmp}/simulations")

    # def _untemp(self):
    #     if pathlib.Path(f"{self.gm2path}/gm2scan/results/{self.output}").is_dir():
    #         print(f"Overwriting /{self.output}.")
    #         shutil.rmtree(f"{self.gm2path}/gm2scan/results/{self.output}")
    #     else:
    #         print(f"Creating /{self.output}.")
    #     os.rename(f"{self.gm2path}/gm2scan/results/{self.tmp}", f"{self.gm2path}/gm2scan/results/{self.output}")

    def __extract_var(self):
        rg = Ring.load(dir = f"m2-{self.scan_dir}")
        sg = StateGenerator.load(dir = f"m2-{self.scan_dir}")
        if sg.plex == "multiplex":
            subruns, labels = sg.search()
        elif rg.mode == "multiplex":
            subruns, labels = rg.search()
        else:
            raise RuntimeError("Neither the ring nor the state generator are listed as multiplex mode...")
        
        # ylabel = labels[0].split("=")[0].split(": ")[4:]
        ylabel = labels[0].split("=")[0]
        yvals = [float(label.split("=")[1]) for label in labels]

        return ylabel, yvals

    def save(self, chi2):
        np.savetxt(fname = f"{self.gm2path}/gm2scan/results/{self.output}/chi2.txt", X = chi2)

    @staticmethod
    def __translate(source_dir, ylabel, yvals):
        if ": " in ylabel:
            # The changing variable is in the state generator.
            variable, parameter = ylabel.split(": ")
            if parameter == "mean":
                prefix = "\mu_"
            elif parameter == "std":
                prefix = "\sigma_"
            elif parameter == "value":
                prefix = ""

            if variable == "x":
                label = rf"${prefix}x$ [mm]"
            elif variable == "p":
                label = rf"${prefix}p$ [%]"
                yvals /= (aux.p_magic / 100)
            elif variable == "f":
                label = rf"${prefix}f$ [kHz]"
            elif variable == "r_co":
                label = rf"${prefix}" + r"{x_e}$ [mm]"
            elif variable == "alpha":
                label = rf"${prefix}$ [mrad]"
                yvals /= 1000
            elif variable == "phi_0":
                label = rf"${prefix}" + r"{\phi_0}$ [mrad]"
            elif variable == "t":
                if parameter == "zero":
                    label = r"$t_a$ [ns]"
                elif parameter == "dir":
                    label = "Filename"
            
            sg = StateGenerator.load(f"s2-{source_dir}")
            truth = aux.delist(getattr(getattr(sg, variable), parameter))

        else: # Ring or kicker multiplex
            object = "rg"
            variable = ylabel
            if variable == "b_norm":
                label = r"$B_{max}$ [G]"
            elif variable == "t_norm":
                label = r"$t_{norm}$ [ns]"
            elif variable == "kick_max":
                label = "Number of kicks"
            elif variable == "kick_num":
                label = "Number of kicker segments"
            elif variable == "quad_num":
                label = "Number of ESQ segments"
            elif variable == "n":
                label = r"Field index $n$"
            elif variable == "collimators":
                label = "Collimation type"
            elif variable == "b_nom":
                label = "Nominal magnetic field strength [T]"
            
            rg = Ring.load(f"s2-{source_dir}")
            try:
                truth = aux.delist(getattr(rg, variable))
            except AttributeError:
                truth = aux.delist(getattr(rg.b_k, variable))
            

        return truth, label, yvals

    @staticmethod
    def __custom(custom, yvalues):
        ylabel, func = custom
        post = [func(yval) for yval in yvalues]

        return ylabel, post

    @staticmethod
    def load(dir):
        return np.loadtxt(fname = f"{pathlib.Path(os.path.abspath(__file__)).parents[1]}/gm2scan/results/{dir}/chi2.txt")

    @classmethod
    def remake(dir):
        pass