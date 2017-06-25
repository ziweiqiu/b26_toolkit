
# This file is part of b26_toolkit, a PyLabControl add-on for experiments in Harvard LISE B26.
# Copyright (C) <2016>  Arthur Safira, Jan Gieseler, Aaron Kabcenell
#
# Foobar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

def update_fluorescence(image_data, axes_image, min_counts = -1, max_counts = -1):
    """
    updates a the data in a fluorescence  plot. This is more efficient than replotting from scratch
    Args:
        image_data: 2D - array
        axes_image: axes object on which to plot
        implot: reference to image plot
    Returns:

    """

    if max_counts >= 0:
        image_data = np.clip(image_data, None, max_counts)
        colorbar_max = max_counts
    else:
        colorbar_max = np.max(image_data)

    if min_counts >=0:
        image_data = np.clip(image_data, min_counts, None)
        colorbar_min = min_counts
    else:
        colorbar_min = np.min(image_data)

    implot = axes_image.images[0]
    colorbar = implot.colorbar

    implot.set_data(image_data)

    if colorbar is not None:
        colorbar_labels = [np.floor(x) for x in np.linspace(colorbar_min, colorbar_max, 5, endpoint=True)]
        colorbar.set_ticks(colorbar_labels)
        colorbar.set_clim(colorbar_min, colorbar_max)
        colorbar.update_normal(implot)

def plot_fluorescence_new(image_data, extent, varlbls, varinialpos, axes_image, min_counts = -1, max_counts = -1, colorbar = None):
    """
    plots fluorescence data in a 2D plot
    Args:
        image_data: 2D - array
        extent: vector of length 4, i.e. [x_min, x_max, y_max, y_min]
        axes_image: axes object on which to plot
        max_counts: cap colorbar at this value if negative autoscale

    Returns:

    """
    if max_counts >= 0:
        image_data = np.clip(image_data, None, max_counts)
        colorbar_max = max_counts
    else:
        colorbar_max = np.max(image_data)

    if min_counts >=0:
        image_data = np.clip(image_data, min_counts, None)
        colorbar_min = min_counts
    else:
        colorbar_min = np.min(image_data)

    extra_x_extent = (extent[1]-extent[0])/float(2*(len(image_data[0])-1))
    extra_y_extent = (extent[2]-extent[3])/float(2*(len(image_data)-1))
    extent = [extent[0] - extra_x_extent, extent[1] + extra_x_extent, extent[2] + extra_y_extent, extent[3] - extra_y_extent]
    # extent[0] = extent[0]*varcalib[0]
    # extent[1] = extent[1]*varcalib[0]
    # extent[2] = extent[2]*varcalib[1]
    # extent[3] = extent[3]*varcalib[1]

    fig = axes_image.get_figure()

    implot = axes_image.imshow(image_data, cmap='pink', interpolation="nearest", extent=extent)
    # axes_image.scatter(x=[varinialpos[0]*varcalib[0]], y=[varinialpos[1]*varcalib[1]], c='r', s=40, marker='x')
    axes_image.scatter(x=[varinialpos[0]], y=[varinialpos[1]], c='r', s=40, marker='x')
    axes_image.set_xlabel(varlbls[0])
    axes_image.set_ylabel(varlbls[1])
    axes_image.set_title('Confocal Image')

    axes_image.set_xticklabels(axes_image.get_xticks(), rotation=90)

    colorbar_labels = [np.floor(x) for x in np.linspace(colorbar_min, colorbar_max, 5, endpoint=True)]

    if colorbar is None:
        colorbar = fig.colorbar(implot, label='kcounts/sec')
    else:
        colorbar = fig.colorbar(implot, cax=colorbar.ax, label='kcounts/sec')

    colorbar.set_ticks(colorbar_labels)
    colorbar.set_clim(colorbar_min, colorbar_max)