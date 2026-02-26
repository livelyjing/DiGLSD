import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Create figure
fig = plt.figure(figsize=(8, 5))
ax = fig.add_axes([0.1, 0.45, 0.8, 0.1])

# Normalize values from 0 to 360
norm = mpl.colors.Normalize(vmin=0, vmax=360)

# Create colorbar
cbar = mpl.colorbar.ColorbarBase(
    ax,
    cmap=plt.cm.cool,
    norm=norm,
    orientation='horizontal'
)
cbar.set_ticks(np.arange(0, 361, 30))
cbar.set_label("Angle (degrees)")

plt.show()