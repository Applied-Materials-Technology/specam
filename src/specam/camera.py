import numpy as np

from specam.constants import c1, c2, abs0
from specam.data import SpectralDataGenerated
from specam.models import fit_data
from specam.plotting import SpectrumPlot


def planck_eqn(lam, T):
    return c1 / (lam**5 * (np.exp(c2 / (lam * T)) - 1))
    # return c1 / (lam**5 * np.exp(c2 / (lam * T)))


class Camera(object):
    # props:
    # lam_vals, lam_0, lam_inf, lam_N, signal_noise_ratio
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.props = kwargs
        self.results = {}
        self.test_data = None

    @classmethod
    def create(cls, name: str, kind: str = None, **kwargs):
        constructor_lookup = {
            'vals': cls.from_vals,
            'linspace': cls.from_linspace,
            None: cls.from_linspace,
        }
        constructor = constructor_lookup.get(kind, None)
        if constructor is None:
            raise ValueError(f'Unknown camera constructor `{kind}`')
        
        return constructor(name, **kwargs)

    @classmethod
    def from_linspace(
        cls, 
        name: str, 
        start: float, 
        stop: float, 
        num: int, 
        **kwargs
    ):
        kwargs.update({
            'lam_vals': np.linspace(start, stop, num),
            'lam_0': start, 
            'lam_inf': stop,
            'lam_N': num,
        })
        return cls(name, **kwargs)
    
    @classmethod
    def from_vals(cls, name: str, vals: np.ndarray, **kwargs):
        vals = np.copy(vals)
        kwargs.update({
            'lam_vals': vals,
            'lam_0': vals.min(), 
            'lam_inf': vals.max(),
            'lam_N': len(vals),
        })
        return cls(name, **kwargs)

    def create_test_data(
        self,
        T_0: float,
        T_inf: float,
        T_N: int,
        intensity_func: callable,
        intensity_func_log: callable,
        intensity_params: dict,
        noise_sigma: float = None,
    ) -> None:
        T = np.linspace(T_0, T_inf, T_N)
        lam = self.props["lam_vals"]
        intensity_params.update({
            "lam_0": lam[0],
            "lam_inf": lam[-1],
        })
        signal_noise_ratio = self.props.get("signal_noise_ratio", 1000)
        
        self.test_data = SpectralDataGenerated.create(
            lam,
            T,
            intensity_func,
            intensity_func_log,
            intensity_params,
            noise_sigma=noise_sigma,
            signal_noise_ratio=signal_noise_ratio,
        )

    def fit_data(self, model_name, spectral_data, **kwargs):
        return fit_data(
            model_name,
            spectral_data,
            lam_0=self.props["lam_0"],
            lam_inf=self.props["lam_inf"],
            **kwargs,
        )

    def fit_test_data(self, model_name, save_name=None, **kwargs):
        self.results[save_name or model_name] = self.fit_data(
            model_name, self.test_data, **kwargs
        )

    def plot_spectrum(self, result_names=None, **kwargs):
        if isinstance(result_names, str):
            result_names = [result_names]
        result_names = result_names or []

        plot = SpectrumPlot(**kwargs)
        plot.add_data(kind='true', plot_data=self.test_data, label='measured')
        for result_name in result_names:
            plot.add_data(
                plot_data=self.results[result_name], label=result_name
            )

        return plot
