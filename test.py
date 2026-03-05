from src.afm_io import load_afm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

a, b, z = load_afm('data/pvp8k/2-5-dmfa-pvp-temp.032', fmt='spm')
print(z.shape, z.dtype, np.nanmin(z), np.nanmax(z))
# plt.imshow(z, cmap='afmhot')
print(a, b)
assert z.ndim == 2
assert isinstance(z, np.ndarray)

info = pd.DataFrame(z.flatten(), columns=['height_nm'])
print(info.describe())
