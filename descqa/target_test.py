from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt
import math
import matplotlib.ticker as ticker
import pandas as pd

__all__ = ['Targettest']

class Targettest(BaseValidationTest):
    figx_p = 10
    figy_p = 10
    fsize = 16 #fontsize
    msize = 6  #markersize 
    lsize = 10 #legendsize
    possible_observations = {
        'DESI_TG': {
            'filename_template': 'N_z/DESI_TS/desi_{}.txt',
            'usecols': (0, 1),
            'colnames': ('z','dndAdz'),
            'label': 'desi_targaet selection/ collected by chto',
        },
    }
    datacolor="r"
 
    def __init__(self, observation="", **kwargs):
        self.kwargs = kwargs
        
        possible_mag_fields = ('mag_{}_des', 
                               'mag_{}_sdss', 
                               'mag_{}_lsst',
                               'mag_true_{}_wise'
                              )
        self.bands = ['g','r','z','W1','W2']
        self.zbins = np.linspace(*kwargs.get('z_bins', (0,2,40)))
        self.possible_mag_fields = possible_mag_fields
        self._other_kwargs = kwargs
        self.dz = self.zbins[1:] - self.zbins[:-1]
        self.z_center = (self.zbins[1:] + self.zbins[:-1])*0.5
        self.galaxytypes=['BGS','LRG','ELG','QSO']
        self.ncols = 2
        self.n_gtypes = len(self.galaxytypes)
        self.validation_data = None
        self.observation = observation
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        elif observation not in self.possible_observations:
            raise ValueError('Observation {} not available'.format(observation))
        else:
            self.validation_data = self.get_validation_data(observation)
    def get_validation_data(self, observation): 
        data_args = self.possible_observations[observation]
        data_paths=[]
        for galtype in self.galaxytypes:
            data_path = os.path.join(self.data_dir, data_args['filename_template'].format(galtype))
            if not os.path.exists(data_path):
               raise ValueError("{}-band data file {} not found".format(band, data_path))
            if not os.path.getsize(data_path):
               raise ValueError("{}-band data file {} is empty".format(band, data_path))
            data_paths.append(data_path)
        datas=[]
        for data_path in data_paths:
            tmp = pd.read_csv(data_path, delimiter = r"\s+", header=None,comment='#')
            datas.append({"z": tmp[data_args['usecols'][0]].as_matrix(),"dndAdz": tmp[data_args['usecols'][1]].as_matrix()})
         
        return datas 
    def prepare_galaxy_catalog(self, gc):
        quantities_needed = []
        try:
            for band in self.bands:
                possible_mag_field=[f.format(band) for f in self.possible_mag_fields]
                absolute_magnitude_field = gc.first_available(*possible_mag_field)
                quantities_needed.append(absolute_magnitude_field)
        except ValueError:
            return
        quantities_needed.extend(['redshift_true'])
        if not gc.has_quantities(quantities_needed):
            return

        return quantities_needed
        
    def selec_target_type(self, galtype, quantities_needed, data):
        g = data[quantities_needed[self.bands.index('g')]]
        r = data[quantities_needed[self.bands.index('r')]]
        z = data[quantities_needed[self.bands.index('z')]]
        w1 = data[quantities_needed[self.bands.index('W1')]]
        w2 = data[quantities_needed[self.bands.index('W2')]]
        if galtype == "BGS":
            mask = r < 19.45
        if galtype == "LRG":
            mask = ((z-w1)-0.7*(r-z) > -0.6) & ((z-w1)-0.7*(r-z) < 1.0) & (z<20.4)&(z>18)&(r-z>0.8)&(r-z<2.5)& (z-2*(r-z-1.2)>17.4) &(z-2*(r-z-1.2)<19.45)&(((r-z)>1.2)|((g-r)>1.7))
        if galtype == "ELG":
            mask = (r<23.4)&(r-z>0.3)&(r-z<1.6)&((1.15*(r-z)-0.15)>(g-r))&(1.6-1.2*(r-z)>(g-r))
        if galtype == "QSO":
            gflux= 10**((22.5-g)/2.5)
            rflux= 10**((22.5-r)/2.5)
            zflux=10**((22.5-z)/2.5)
            w1flux=10**((22.5-w1)/2.5)
            w2flux=10**((22.5-w2)/2.5)
            wflux = 0.75*w1+0.25*w2
            grzflux =  (gflux + 0.8*rflux + 0.5*zflux) / 2.3
            grzmag = 22.5-2.5*np.log10(grzflux)
            wmag = 22.5-2.5*np.log10(wflux)
            mask = (r<22.7)&(grzmag>17.0)&((g-r) < 1.3)&((r-z) > -0.3)&((r-z) < 1.1)&((grzmag-wmag)>(g-z-1.0))&(w1-w2>-0.4)
        return mask
            
    def run_on_single_catalog(self,catalog_instance, catalog_name, output_dir):
        prepared = self.prepare_galaxy_catalog(catalog_instance)
        if prepared is None:
            TestResult(skipped=True)

        quantities_needed = prepared
        colnames = [*quantities_needed]
        data = catalog_instance.get_quantities(quantities_needed)
        normalizedhist=[]
        area = catalog_instance.get_catalog_info()['sky_area']
        z = data['redshift_true']
        for galtype in self.galaxytypes:
            mask = self.selec_target_type(galtype,quantities_needed,data)
            hist = np.histogram(z[mask],bins=self.zbins)
            normalizedhist.append(hist[0]/self.dz/area)
        self.make_plot(normalizedhist,catalog_name,os.path.join(output_dir, 'target_test.png'))
        return TestResult(passed=True, score=0)
        
    def make_plot(self, normalizedhsit, name, save_to):
        nrows=math.ceil(float(self.n_gtypes)/self.ncols)
        fig, ax = plt.subplots(nrows, self.ncols, sharex=True, figsize=(self.figx_p,self.figy_p), dpi=100)
        for index in range(self.n_gtypes):
            ax_this = ax.flat[index]
            upper = np.max(normalizedhsit[index])
            ax_this.step(self.z_center, normalizedhsit[index],"-", where="mid", color="k",label="sim"+" "+self.galaxytypes[index])
            if self.validation_data:
               data = self.validation_data[index]
               upper = np.max([np.max(data['dndAdz']), upper])
               ax_this.step(data['z'], data['dndAdz'],"-", where="mid", color=self.datacolor, label=self.observation+" "+self.galaxytypes[index])

            ax_this.set_ylim([0, upper])
            ax_this.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax_this.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=self.lsize, numpoints=1)
        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(False)
        ax.set_ylabel(r'$dN/dzdA (1/deg^2)$',size=self.fsize,labelpad=25)
        ax.set_xlabel(r'z',size=self.fsize)
        ax.set_title(name)
        
        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)
