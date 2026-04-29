import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import hsv_to_rgb
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import numpy as np
import xarray as xr


def plot_xarray_2d(X: xr.DataArray, need_log=False, show_ax_names=True,
                   clim=(None, None)):
    coord_x = X.dims[1]
    coord_y = X.dims[0]
    cx = X.coords[coord_x].values
    cy = X.coords[coord_y].values
    ext = [cx[0], cx[-1], cy[-1], cy[0]]
    X_ = X.values
    if need_log:
        X_ = np.log(X_)
    plt.imshow(X_, aspect='auto', extent=ext, origin='upper',
               vmin=clim[0], vmax=clim[1])
    plt.xlim(cx[0], cx[-1])
    plt.ylim(cy[-1], cy[0])
    if show_ax_names:
        plt.xlabel(coord_x)
        plt.ylabel(coord_y)

def plot_xarray_2d_irreg(X: xr.DataArray, need_log=False, cmap=None):
    coord_x = X.dims[1]
    coord_y = X.dims[0]
    cx = X.coords[coord_x].values
    cy = X.coords[coord_y].values
    Cx, Cy = np.meshgrid(cx, cy)
    ax = plt.gca()
    X_ = X.values
    if need_log:
        X_ = np.log(X_)
    if cmap is not None:
        cmap = plt.get_cmap(cmap)
    m = ax.pcolormesh(Cx, Cy, X_, shading='auto', cmap=cmap)
    #ax.set_xscale('log')
    ax.invert_yaxis()
    plt.xlabel(coord_x)
    plt.ylabel(coord_y)
    plt.colorbar(m)
    plt.title(X.name)
    
def polar_to_rgb(X, s_mult=2, v=0.9):
    X_hsv = np.full((*X.shape, 3), np.nan)
    phi = np.angle(X)
    phi = phi % (2 * np.pi)
    X_hsv[:, :, 0] = phi / (2 * np.pi)           # hue = angle
    X_hsv[:, :, 1] = np.abs(X) * s_mult          # saturation = absolute value
    X_hsv[:, :, 2] = v                           # value = const
    X_hsv = np.minimum(X_hsv, 1)
    return hsv_to_rgb(X_hsv)

def hsv_colorbar(shift=0.5):
    def _shifted_cmap(cmap, shift_, name='shifted_cmap'):
        N = cmap.N
        colors_array = cmap(np.linspace(0, 1, N))
        colors_array = np.roll(colors_array, int(N * shift_), axis=0)
        return mcolors.LinearSegmentedColormap.from_list(name, colors_array)
    shifted_hsv = _shifted_cmap(plt.get_cmap('hsv'), shift_=shift)
    cbar_ax = plt.gca().inset_axes([1.05, 0.1, 0.03, 0.8])
    norm = mcolors.Normalize(vmin=-np.pi, vmax=np.pi)
    hsv_cbar = mcolorbar.ColorbarBase(cbar_ax, cmap=shifted_hsv, norm=norm, orientation='vertical')
    hsv_cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    hsv_cbar.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])


def plot_circle_matrix(C, R, clim=None, rmax=None, cmap='twilight', fig_num=None):

    if not np.all(C.shape == R.shape):
        raise ValueError('C and R shhould have the same size')
    ny, nx = C.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    
# =============================================================================
#     if fig_num is None:
#         fig, ax = plt.subplots()
#     else:
#         fig, ax = plt.subplots(num=fig_num)
#         plt.clf()
# =============================================================================

    fig = plt.gcf()
    ax = plt.gca()
    
    if rmax is None:
        rmax = R.max()
    R = R / rmax / 2
    
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    p = PatchCollection(circles, array=C.flatten(), cmap=cmap)
    if clim is not None:
        p.set_clim(clim)
    ax.add_collection(p)
    
    ax.set_xticks(np.arange(nx + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(ny + 1) - 0.5, minor=True)
    ax.grid(which='minor')
    
    fig.colorbar(p)
    plt.show()
