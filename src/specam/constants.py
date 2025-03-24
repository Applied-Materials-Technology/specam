from scipy import constants

__all__ = ['c1', 'c2', 'abs0']

c1 = constants.physical_constants['first radiation constant for spectral radiance'][0]
c2 = constants.physical_constants['second radiation constant'][0]
abs0 = constants.zero_Celsius