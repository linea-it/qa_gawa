import matplotlib.pyplot as plt
import numpy as np
import warnings, json, astropy, os
import astropy.io.fits as fits
from astropy.io.fits import getdata
import astropy.units as u
from astropy.coordinates import SkyCoord
warnings.filterwarnings("ignore")
from qa_gawa import plot_pure, plot_comp

# Main settings:
confg = "qa_gawa.json"

# read config file
with open(confg) as fstream:
    param = json.load(fstream)

globals().update(param)

os.system('jupyter nbconvert --execute --to html --EmbedImagesPreprocessor.embed_images=True qa_gawa.ipynb')
os.system('mkdir -p ' + copy_dir)
os.system('cp qa_gawa.html ' + copy_dir + '/index.html')
