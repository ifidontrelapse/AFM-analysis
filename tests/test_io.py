import numpy as np


def test_load_spm():
    from src.afm_io import load_afm

    try:
        z = load_afm("data/5.011", fmt="spm")
    except ImportError:
        print("pyfmreader not installed, skipping .spm test.")
        return