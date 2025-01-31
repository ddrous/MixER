print("\n############# Contextual Self-Modulation #############\n")

## System-level configuration
import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.25'
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = 'true'
# os.environ["EQX_ON_ERROR"] = 'breakpoint'
import sys

## Numpy configs
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=sys.maxsize)

## Plotting configs
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
# plt.style.use("dark_background")
mpl.rcParams['savefig.facecolor'] = 'w'

# Set the following parameters for scientfic plots
sns.set_theme(context='notebook', style='ticks',
        font='sans-serif', font_scale=1, color_codes=True, rc={"lines.linewidth": 2})
## plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'

## JAX configuration
import jax
import equinox as eqx
import diffrax
import einops

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)

print("Jax version:", jax.__version__)
print("Available devices:", jax.devices())
