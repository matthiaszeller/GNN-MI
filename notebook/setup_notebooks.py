import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('matplotlib', type=bool, default=True, nargs='?')
args = parser.parse_args()

# Add the project directory in the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.joinpath('src')))
sys.path.insert(0, str(Path(__file__).parent.parent))

if args.matplotlib:
    # Override matplotlib defaults for nicer plots
    sns.set(style='whitegrid')
    # Make plots more latex-like looking
    matplotlib.rcParams.update({
        'backend': 'ps',
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{gensymb}',
    })
    # Increase font size
    SMALL_SIZE, MEDIUM_SIZE, BIGGER_SIZE = 14, 16, 18
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def process_figure(filename):
    """Prettify figure (e.g. despine and make it latex-like) and save it to file"""
    sns.despine()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

