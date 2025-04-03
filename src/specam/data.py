import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Union
import abc
from dataclasses import dataclass


class Data(abc.ABC):
    def __init__(self):
        self.datastores = {}

    def add_datastore(self, name : str, ds: dict):
        self.datastores[name] = ds

    def check_batch_sizes(self):
        sizes = np.array([
            len(data) 
                for ds in self.datastores.values() 
                for data in ds.values()
        ])
        assert len(sizes) == 0 or all(sizes[1:] == sizes[0])

    @property
    def batch_size(self):
        ds = next(iter(self.datastores.values()))
        data = next(iter(ds.values()))
        return len(data)

    def __getitem__(self, key):
        for ds in self.datastores.values():
            try:
                return ds[key]
            except KeyError:
                continue
        else:
            raise KeyError(key)

    def get(self, key, value=None):
        try:
            return self[key]
        except KeyError:
            return value
        
    def keys(self):
        return sum((list(ds.keys()) for ds in self.datastores.values()), [])


class SpectralData(Data):
    """Spectral data are arrays of shape [batch_size, spectral_dimension]"""
    def __init__(
        self, 
        lam: np.ndarray, 
        spectral_data: dict = None,
        intensity_label: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.spectral_data = spectral_data if spectral_data else {}
        # self.spectral_data.update({
        #     'lam': lam,
        # })
        self.add_datastore('spectral', self.spectral_data)
        self.lam = lam
        self.intensity_label = intensity_label

        self.check_batch_sizes()

    @property
    def intensity(self):
        return self.spectral_data[self.intensity_label]

    def get_intensity(self, i):
        return self.intensity[i]


class ScalarData(Data):
    """Scalar data are arrays of shape [batch_size, ...]"""
    def __init__(self, scalar_data: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.scalar_data = scalar_data if scalar_data else {}
        self.add_datastore('scalar', self.scalar_data)

        self.check_batch_sizes()
    

class SpectralDataGenerated(SpectralData, ScalarData):
    def __init__(
        self, 
        intensity_func: callable, 
        intensity_func_log: callable, 
        intensity_params: dict, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.intensity_func = intensity_func
        self.intensity_func_log = intensity_func_log
        self.intensity_params = intensity_params

    # T: np.ndarray
    # # epsilon: np.ndarray
    # noise_sigma: np.ndarray
    # intensity_func: callable
    # intensity_func_log: callable
    # intensity_params: dict

    @classmethod
    def create(
        cls,
        lam: np.ndarray,
        T: np.ndarray,
        intensity_func: callable,
        intensity_func_log: callable,
        intensity_params: dict,
        noise_sigma: Optional[float] = None,
        signal_noise_ratio: Optional[float] = None,
    ):
        grid = np.meshgrid(T, lam, indexing="ij")
        I = intensity_func(grid[1], grid[0], **intensity_params)

        if signal_noise_ratio is None and noise_sigma is None:
            raise ValueError(
                '`signal_noise_ratio` or `noise_sigma` must be specified.'
            )
        if noise_sigma is None:
            noise_sigma = I.max(axis=1) / 0.9 / signal_noise_ratio
        else:
            noise_sigma = np.full(len(T), noise_sigma)

        for i, spectrum in enumerate(I):
            spectrum += np.random.normal(0, noise_sigma[i], len(lam))
        I[I <= 0.] = 1e-8

        # epsilon = I[:, 0] / planck_eqn(lam_test, T_0)

        spectral_data = {
            'intensity': I,
        }
        scalar_data = {
            'T': T,
            'noise_sigma': noise_sigma
        }

        return cls(
            intensity_func,
            intensity_func_log,
            intensity_params,
            lam=lam, 
            spectral_data=spectral_data,
            intensity_label='intensity',
            scalar_data=scalar_data,
        )


@dataclass
class SpectralDataMeasured(SpectralData, ScalarData):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def load_far_data(
        cls, 
        logpath: Path, 
        datapath: Optional[Path] = None,
        start_idx: int = 0,
    ):
        if datapath is None:
            datapath = logpath.with_suffix('')

        log_data = pd.read_csv(
            logpath, 
            skiprows=13, 
            delimiter=r"\s+", 
            usecols=range(7)
        )
        log_ignore_cols = ('sequence', )
        scalar_data = {
            c: log_data[c].to_numpy() 
                for c in log_data.columns 
                if c.lower() not in log_ignore_cols
        }

        data_pattern = "Temp*.DAT"
        data_col_names = (
            'wavelength', 
            'raw intensity', 
            'background', 
            'corrected intensity', 
            'spectral emissivity',
        )
        data_ignore_cols = ('wavelength', )
        
        wc_start = data_pattern.find('*')
        wc_end = len(data_pattern) - wc_start - 1

        data_idxes = []
        spectral_data = []
        for data_file in sorted(
            datapath.glob(data_pattern), 
            key=lambda s:int(s.name[wc_start:-wc_end])
        ):
            data_idxes.append(int(data_file.name[wc_start:-wc_end]))
            spectral_data.append(
                np.loadtxt(data_file, usecols=range(len(data_col_names)))
            )
        data_idxes = np.array(data_idxes)
        spectral_data = np.array(spectral_data)

        if not np.all(log_data['Sequence'].to_numpy() == data_idxes):
            raise ValueError('Inconsistent data between log and output files.')
        
        lam_all = spectral_data[:, :, 0]
        if not np.all([np.array_equal(lam_all[0], lam) for lam in lam_all[1:]]):
            raise ValueError('Inconsistent wavelengths for some timesteps.')
        
        spectral_data_dict = {
            c: spectral_data[:, start_idx:, i] 
                for i, c in enumerate(data_col_names)
                if c.lower() not in data_ignore_cols
        }

        return cls(
            lam=lam_all[0, start_idx:] * 1e-9, 
            spectral_data=spectral_data_dict,
            intensity_label='corrected intensity',
            scalar_data=scalar_data,
        )


class SpectralDataFitted(ScalarData):
    def __init__(
        self,
        fitted_data: type[SpectralData],
        intensity_func: callable,
        intensity_func_log: callable,
        intensity_params: tuple,
        intensity_props: dict,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fitted_data = fitted_data
        # below not necessarily there, look after
        self.intensity_func = intensity_func
        self.intensity_func_log = intensity_func_log
        self.intensity_params = intensity_params
        self.intensity_props = intensity_props

    @property
    def lam(self):
        return self.fitted_data.lam
    
    def get_intensity(self, i):
        params = (self[k][i] for k in self.intensity_params)
        return self.intensity_func(
            self.lam,
            self["T"][i],
            *params,
            **self.intensity_props,
        )
