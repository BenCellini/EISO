# written by Floris van Breugel, with some help from Andrew Straw and Will Dickson
# dependencies for LaTex rendering: texlive, ghostscript, dvipng, texlive-latex-extra

# general imports
import matplotlib
##print matplotlib.__version__
##print 'recommended version: 1.1.1 or greater'

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# used for colorline
from matplotlib.collections import LineCollection

# used in histogram
from scipy.stats import norm as gaussian_distribution
from scipy.stats import uniform as uniform_distribution
from scipy import signal

# used in colorbar
import matplotlib.colorbar

# used in scatter
from matplotlib.collections import PatchCollection

import sympy
import copy

# not used
# import scipy.optimize
# import scipy.stats.distributions as distributions

###################################################################################################
# Misc Info
###################################################################################################

# FUNCTIONS contained in this file:
# adjust_spines
# colorline
# histogram
# histogram2d (heatmap)
# boxplot
# colorbar (scale for colormap stuff), intended for just generating a colorbar for use in illustrator figure assembly


# useful links:
# colormaps: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps


cmap_bgy = matplotlib.colors.LinearSegmentedColormap.from_list("my_new_colormap", [(0, 0, .4), (0, .7, .7), (1, 1, .3)])


###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below)
def adjust_spines(ax, spines, spine_locations={}, smart_bounds=True, xticks=None, yticks=None, linewidth=1):
    if type(spines) is not list:
        spines = [spines]

    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()

    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]

    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none')  # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return

    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward', spine_locations_dict[loc]))  # outward by x points
            spine.set_linewidth(linewidth)
            spine.set_color('black')
        else:
            spine.set_color('none')  # don't draw spine

    # smart bounds, if possible
    if int(matplotlib.__version__[0]) > 0 and smart_bounds:
        for loc, spine in ax.spines.items():
            ticks = None
            if loc in ['left', 'right']:
                ticks = yticks
            if loc in ['top', 'bottom']:
                ticks = xticks
            if ticks is not None and len(ticks) > 0:
                spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)

    for line in ax.get_xticklines() + ax.get_yticklines():
        # line.set_markersize(6)
        line.set_markeredgewidth(linewidth)


###################################################################################################
# Map to Color
###################################################################################################

def get_color_transformer(norm=(0, 1), colormap='jet', clip=True):
    '''
    returns a function that will return a color value (4-tuple) from the given color map based on a single input, which is scaled to the range given by norm. clip is passed to plt.Normalize, default is True.
    '''

    def color_transformer(v):
        Norm = plt.Normalize(norm[0], norm[1], clip=clip)
        cmap = plt.get_cmap(colormap)
        return cmap(Norm(v))

    return color_transformer


###################################################################################################
# Colorline
###################################################################################################

# plot a line in x and y with changing colors defined by z, and optionally changing linewidths defined by linewidth
def colorline(ax, x, y, z, linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid', cmap=None,
              hide_nan_indices=True, hack_round_caps=False, axis_size_inches=None, cap_size_radius_adjustment=2):
    '''
    hack_round_caps - extend line segments so that line appears continuous. beta mode.
    axis_size_inches - used for hack_projected_cap (x,y). Not well implemented.
    '''
    if cmap is None:
        cmap = plt.get_cmap(colormap)

    if type(linewidth) is list or type(linewidth) is np.array or type(linewidth) is np.ndarray:
        linewidths = linewidth
    else:
        linewidths = np.ones_like(z) * linewidth

    if norm is None:
        norm = plt.Normalize(np.min(z), np.max(z))
    else:
        norm = plt.Normalize(norm[0], norm[1])

    '''
    if self.hide_colorbar is False:
        if self.cb is None:
            self.cb = matplotlib.colorbar.ColorbarBase(self.ax1, cmap=cmap, norm=norm, orientation='vertical', boundaries=None)
    '''

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2).astype(float)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ##print segments
    if hide_nan_indices == True:
        nanindices_x = np.where(np.isnan(x))[0].tolist()
        nanindices_y = np.where(np.isnan(y))[0].tolist()
        nanindices_x.extend(nanindices_y)
        nanindices = np.unique(nanindices_x)
        segments = np.delete(segments, nanindices, axis=0)
        z = np.delete(z, nanindices, axis=0)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    ordered_zorder = (zorder - 1)  # + z/float(len(z))
    if hasattr(linewidth, '__iter__'):
        lc = LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha,
                            linestyles=linestyle)
        lc.set_array(z)
        lc.set_zorder(ordered_zorder)  # .tolist()
        lc.set_linewidth(linewidth)
    else:
        lc = LineCollection(segments, linewidths=linewidth, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha,
                            linestyles=linestyle)
        lc.set_array(z)
        lc.set_zorder(ordered_zorder)  # .tolist()
    ax.add_collection(lc)

    if hack_round_caps:
        ax.scatter(x, y, color=cmap(norm(z)), s=linewidth ** 2, edgecolor='none', zorder=(z - 10).tolist())


###################################################################################################
# Colorline with heading
###################################################################################################

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet',
                                colornorm=None, size_radius_range=(0.01, .1), size_radius_norm=None, edgecolor='none',
                                alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation

    Outputs:
    Patch Collection

    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string.
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to,
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)

    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)

    indices_to_plot = np.arange(0, len(x), nskip + 1)

    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation * 180. / np.pi
    if flip:
        orientation += 180

    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray:
            r = size_radius_norm(size_radius[i]) * (size_radius_range[1] - size_radius_range[0]) + size_radius_range[0]
        else:
            r = size_radius

        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray:
            angle_swept = size_radius[i]
        else:
            angle_swept = size_radius
        theta1 = orientation[i] - size_angle / 2.
        theta2 = orientation[i] + size_angle / 2.

        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i] * np.pi / 180.) * r * center_offset_fraction
        center[1] -= np.sin(orientation[i] * np.pi / 180.) * r * center_offset_fraction

        wedge = patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)

    # add collection and color it
    pc = PatchCollection(flycons, cmap=cmap, norm=colornorm)

    # set properties for collection
    pc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)

    return pc


def colorline_with_heading(ax, x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None,
                           size_radius_range=(0.01, .1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True,
                           deg=True, nskip=0, use_center='center', show_centers=True, center_offset_fraction=0.75,
                           center_point_size=2):
    """
    Plots a trajectory with colored wedge shapes to indicate orientation.
    See function get_wedges_for_heading_plot for details

    Additional options:

    show_centers      - (bool) - show a black dot where the actual point is - shows where the center of the wedge is
    center_point_size - markersize for center, if show_centers
    """

    pc = get_wedges_for_heading_plot(x, y, color, orientation, size_radius=size_radius, size_angle=size_angle,
                                     colormap=colormap, colornorm=colornorm, size_radius_range=size_radius_range,
                                     size_radius_norm=size_radius_norm, edgecolor=edgecolor, alpha=alpha, flip=flip,
                                     deg=deg, nskip=nskip, center_offset_fraction=center_offset_fraction)

    ax.add_collection(pc)

    if show_centers:
        indices_to_plot = np.arange(0, len(x), nskip + 1)
        ax.plot(x[indices_to_plot], y[indices_to_plot], '.', color='black', markersize=center_point_size)


###################################################################################################
# Histograms
###################################################################################################

# first some helper functions
def custom_hist_rectangles(hist, leftedges, width, bottomedges=None, facecolor='green', edgecolor='none', alpha=1,
                           alignment='vertical'):
    linewidth = 1
    if edgecolor == 'none':
        linewidth = 0  # hack needed to remove edges in matplotlib.version 1.0+

    if bottomedges is None:
        bottomedges = np.zeros_like(leftedges)

    if type(width) is not list:
        width = [width for i in range(len(hist))]

    if type(facecolor) is not list:
        facecolor = [facecolor for i in range(len(hist))]
    if type(edgecolor) is not list:
        edgecolor = [edgecolor for i in range(len(hist))]

    rects = [None for i in range(len(hist))]

    if alignment == 'vertical':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle([leftedges[i], bottomedges[i]], width[i], hist[i], facecolor=facecolor[i],
                                         edgecolor=edgecolor[i], alpha=alpha, linewidth=linewidth)
    elif alignment == 'horizontal':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle([bottomedges[i], leftedges[i]], hist[i], width[i], facecolor=facecolor[i],
                                         edgecolor=edgecolor[i], alpha=alpha, linewidth=linewidth)

    return rects


def bootstrap_histogram(xdata, bins, normed=False, n=None, return_raw=False):
    if type(xdata) is not np.ndarray:
        xdata = np.array(xdata)

    if n is None:
        n = len(xdata)
    else:
        if n > len(xdata):
            n = len(xdata)
    hist_list = np.zeros([n, len(bins) - 1])

    for i in range(n):
        # Choose #sample_size members of d at random, with replacement
        choices = np.random.random_integers(0, len(xdata) - 1, n)
        xsample = xdata[choices]
        hist = np.histogram(xsample, bins, normed=normed)[0].astype(float)
        hist_list[i, :] = hist

    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)

    ##print 'bootstrapped: '
    ##print
    ##print hist_mean
    ##print
    ##print hist_std

    if return_raw:
        return hist_list
    else:
        return hist_mean, hist_std


def histogram(ax, data_list, bins=10, bin_width_ratio=0.6, colors='green', edgecolor='none', bar_alpha=0.7,
              curve_fill_alpha=0.4, curve_line_alpha=0.8, curve_butter_filter=[3, 0.3], return_vals=False,
              show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, bootstrap_line_width=0.5,
              exponential_histogram=False, smoothing_range=None, smoothing_bins_to_exclude=[], binweights=None,
              n_bootstrap_samples=None, alignment='vertical', peak_trace_alpha=0, show_peak_curve=False,
              data_from_which_to_calculate_binweights=None, data_to_which_calculated_binweights_should_apply='all',
              weight_distributions=None):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    data_to_which_calculated_binweights_should_apply -- list of indices corresponding to datasets which should be normalized by the binweights determined by data_from_which_to_calculate_binweights
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range

    if type(normed) is not list:
        normed = [normed for i in range(len(data_list))]

    if weight_distributions is None:
        weight_distributions = [1 for i in range(len(data_list))]

    if type(bar_alpha) is not list:
        bar_alpha = [bar_alpha for i in range(len(colors))]

    n_bars = float(len(data_list))
    if type(bins) is int:
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])

        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)

    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]

    bin_centers = np.diff(bins) / 2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1 - bin_width_ratio) * bin_width / 2.
    bar_width = (bin_width - 2 * bin_width_buff) / n_bars

    butter_b, butter_a = signal.butter(curve_butter_filter[0], curve_butter_filter[1])

    data_hist_list = []
    if return_vals:
        data_curve_list = []
        data_hist_std_list = []

    # first get max number of occurences
    max_occur = []
    for i, data in enumerate(data_list):
        data_hist = np.histogram(data, bins=bins, normed=normed[i], weights=None)[0].astype(float)
        max_occur.append(np.max(data_hist))
    max_occur = np.max(np.array(max_occur))

    if data_from_which_to_calculate_binweights is not None:
        ##print 'calculating bin weights'
        binweights = np.histogram(data_from_which_to_calculate_binweights, bins=bins, normed=False)[0].astype(float)
        binweights += np.min(binweights) * 1e-10  # to make sure we don't get divide by zero errors
        binweights = binweights ** -1
        if data_to_which_calculated_binweights_should_apply == 'all':
            binweights = [binweights for i in range(len(data_list))]
        else:
            tmp = []
            for i in range(len(data_list)):
                if i in data_to_which_calculated_binweights_should_apply:
                    tmp.append(binweights)
                else:
                    tmp.append(1)
            binweights = tmp

    for i, data in enumerate(data_list):

        if bootstrap_std:
            data_hist, data_hist_std = bootstrap_histogram(data, bins=bins, normed=normed[i], n=n_bootstrap_samples)
        else:
            data_hist = np.histogram(data, bins=bins, normed=normed[i])[0].astype(float)

        if binweights is not None:
            data_hist *= binweights[i]
            if normed[i]:
                data_hist /= np.sum(binweights[i])

        if exponential_histogram:
            data_hist = np.log(data_hist + 1)

        data_hist = data_hist / float(weight_distributions[i])

        if normed_occurences is not False:
            if normed_occurences == 'total':
                data_hist /= max_occur
                if bootstrap_std:
                    data_hist_std /= max_occur
            else:
                div = float(np.max(data_hist))
                ##print div
                data_hist /= div
                if bootstrap_std:
                    data_hist_std /= div

        rects = custom_hist_rectangles(data_hist, bins[0:-1] + bar_width * i + bin_width_buff, width=bar_width,
                                       facecolor=colors[i], edgecolor=edgecolor, alpha=bar_alpha[i],
                                       alignment=alignment)
        if bootstrap_std:
            for j, s in enumerate(data_hist_std):
                x = bins[j] + bar_width * i + bin_width_buff + bar_width / 2.
                # ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                # ax.plot(np.array([x,x]), np.array([data_hist[j], data_hist[j]+data_hist_std[j]]), alpha=bar_alpha, color=colors[i], linewidth=bootstrap_line_width)
                ax.vlines(x, data_hist[j], data_hist[j] + data_hist_std[j], color=colors[i],
                          linewidth=bootstrap_line_width)

                # ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                # ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i])
        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)

        if show_peak_curve:
            if curve_line_alpha > 0:
                ax.plot(bins[0:-1] + bar_width * i + bin_width_buff, data_hist, color=colors[i], alpha=curve_line_alpha)
            if curve_fill_alpha > 0:
                ax.fill_between(bins[0:-1] + bar_width * i + bin_width_buff, data_hist, np.zeros_like(data_hist),
                                color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')

        if show_smoothed:
            if smoothing_range is not None:  # in case you only want to smooth and show smoothing over a select range.
                indices_in_smoothing_range = \
                np.where((bin_centers > smoothing_range[0]) * (bin_centers < smoothing_range[-1]))[0].tolist()
            else:
                indices_in_smoothing_range = [bc for bc in range(len(bin_centers))]

            for b in smoothing_bins_to_exclude:
                try:
                    indices_in_smoothing_range.remove(b)
                except ValueError:
                    print('bin center not in indices list: ', b)
                    print('indices list: ', indices_in_smoothing_range)

            data_hist_filtered = signal.filtfilt(butter_b, butter_a, data_hist[indices_in_smoothing_range])
            interped_bin_centers = np.linspace(bin_centers[indices_in_smoothing_range[0]] - bin_width / 2.,
                                               bin_centers[indices_in_smoothing_range[-1]] + bin_width / 2., 100,
                                               endpoint=True)
            v = 100 / float(len(bin_centers))

            if alignment == 'vertical':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range],
                                                        data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b / v, butter_a / v, interped_data_hist_filtered)
                # ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_between(interped_bin_centers, interped_data_hist_filtered2,
                                    np.zeros_like(interped_data_hist_filtered2), color=colors[i],
                                    alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_bin_centers, interped_data_hist_filtered2, color=colors[i], alpha=curve_line_alpha)

            if alignment == 'horizontal':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range],
                                                        data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b / v, butter_a / v, interped_data_hist_filtered)
                # ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_betweenx(interped_bin_centers, interped_data_hist_filtered2,
                                     np.zeros_like(interped_data_hist_filtered2), color=colors[i],
                                     alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_data_hist_filtered2, interped_bin_centers, color=colors[i], alpha=curve_line_alpha)

        ax.plot(bin_centers, data_hist, color=colors[i], alpha=peak_trace_alpha)

        data_hist_list.append(data_hist)
        if return_vals:
            if bootstrap_std:
                data_hist_std_list.append(data_hist_std)

            if show_smoothed:
                data_curve_list.append([interped_bin_centers, interped_data_hist_filtered2])

    mins_of_data = [np.min(data) for data in data_list]
    maxs_of_data = [np.max(data) for data in data_list]

    mins_of_hist = [np.min(hist) for hist in data_hist_list]
    maxs_of_hist = [np.max(hist) for hist in data_hist_list]

    if alignment == 'vertical':
        ax.set_xlim(np.min(mins_of_data), np.max(maxs_of_data))
        ax.set_ylim(0, np.max(maxs_of_hist))
    elif alignment == 'horizontal':
        ax.set_ylim(np.min(mins_of_data), np.max(maxs_of_data))
        ax.set_xlim(0, np.max(maxs_of_hist))

    if return_vals and bootstrap_std is False:
        return bins, data_hist_list, data_curve_list
    elif return_vals and bootstrap_std is True:
        return bins, data_hist_list, data_hist_std_list, data_curve_list


###########

def histogram_stack(ax, data_list, bins=10, bin_width_ratio=0.8, colors='green', edgecolor='none', normed=True):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    normed - normalizes the SUM of all the stacked histograms
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range

    n_bars = float(len(data_list))
    if type(bins) is int:
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)

    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]

    bin_centers = np.diff(bins) / 2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1 - bin_width_ratio) * bin_width / 2.
    bar_width = (bin_width - bin_width_buff)

    data_hist_list = []

    all_data = []
    for data in data_list:
        all_data.extend(data.tolist())
    all_data_hist = np.histogram(all_data, bins=bins, normed=False)[0].astype(float)
    all_data_hist_normed = np.histogram(all_data, bins=bins, normed=True)[0].astype(float)
    binweights_for_normalizing = all_data_hist_normed / all_data_hist

    prev_data_hist = np.zeros_like(all_data_hist)
    for i, data in enumerate(data_list):

        data_hist = np.histogram(data, bins=bins, normed=False)[0].astype(float)

        if normed:
            data_hist *= binweights_for_normalizing

        rects = custom_hist_rectangles(data_hist, bins[0:-1] + bin_width_buff, bottomedges=prev_data_hist,
                                       width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=1)
        prev_data_hist += data_hist

        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)

        data_hist_list.append(data_hist)

    ax.set_xlim(bins[0], bins[-1])
    if normed:
        ax.set_ylim(0, np.max(all_data_hist_normed) + .1 * np.max(all_data_hist_normed))
    else:
        ax.set_ylim(0, np.max(all_data_hist) + .1 * np.max(all_data_hist))


###################################################################################################
# Boxplots
###################################################################################################

def boxplot(ax, x_data, y_data_list, nbins=50, colormap='YlOrRd', colorlinewidth=2, boxwidth=1, boxlinecolor='black',
            classic_linecolor='gray', usebins=None, boxlinewidth=0.5, outlier_limit=0.01, norm=None,
            use_distribution_for_linewidth=False, min_colorlinewidth=1, show_outliers=True, show_whiskers=True,
            logcolorscale=False, orientation='vertical'):
    # if colormap is None: show a line instead of the 1D histogram (ie. a normal boxplot)
    # use_distribution_for_linewidth will adjust the linewidth according to the histogram of the distribution
    # classic_linecolor: sets the color of the vertical line that shows the extent of the data, if colormap=None
    # outlier limit: decimal % of top and bottom data that is defined as outliers (0.01 = top 1% and bottom 1% are defined as outliers)
    # show_whiskers: toggle the whiskers, if colormap is None

    if usebins is None:
        usebins = nbins
        # usebins lets you assign the bins manually, but it's the same range for each x_coordinate

    for i, y_data in enumerate(y_data_list):
        ###print len(y_data)

        # calc boxplot statistics
        median = np.median(y_data)
        ind = np.where(y_data <= median)[0].tolist()
        first_quartile = np.median(y_data[ind])
        ind = np.where(y_data >= median)[0].tolist()
        last_quartile = np.median(y_data[ind])
        ###print first_quartile, median, last_quartile

        # find outliers
        ind_sorted = np.argsort(y_data)
        bottom_limit = int(len(ind_sorted) * (outlier_limit))
        top_limit = int(len(ind_sorted) * (1 - outlier_limit))
        indices_inrange = ind_sorted[bottom_limit:top_limit]
        outliers = ind_sorted[0:bottom_limit].tolist() + ind_sorted[top_limit:len(ind_sorted) - 1].tolist()
        y_data_inrange = y_data[indices_inrange]
        y_data_outliers = y_data[outliers]
        x = x_data[i]

        # plot colorline
        if colormap is not None:
            hist, bins = np.histogram(y_data_inrange, usebins, normed=True)
            hist = hist.astype(float)
            # hist /= np.max(hist)
            x_arr = np.ones_like(bins) * x

            if logcolorscale:
                hist = np.log(hist + 1)

            if use_distribution_for_linewidth:
                colorlinewidth = hist * colorlinewidth + min_colorlinewidth

            if orientation == 'vertical':
                colorline(ax, x_arr, bins, hist, colormap=colormap, norm=norm,
                          linewidth=colorlinewidth)  # the norm defaults make it so that at each x-coordinate the colormap/linewidth will be scaled to show the full color range. If you want to control the color range for all x-coordinate distributions so that they are the same, set the norm limits when calling boxplot().
            elif orientation == 'horizontal':
                colorline(ax, bins, x_arr, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth)

        elif show_whiskers:
            if orientation == 'vertical':
                ax.vlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.vlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.hlines([np.min(y_data_inrange), np.max(y_data_inrange)], x - boxwidth / 4., x + boxwidth / 4.,
                          color=classic_linecolor, linewidth=boxlinewidth / 2.)
            elif orientation == 'horizontal':
                ax.hlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.hlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.vlines([np.min(y_data_inrange), np.max(y_data_inrange)], x - boxwidth / 4., x + boxwidth / 4.,
                          color=classic_linecolor, linewidth=boxlinewidth / 2.)

        # plot boxplot
        if orientation == 'vertical':
            ax.hlines(median, x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.hlines([first_quartile, last_quartile], x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
            ax.vlines([x - boxwidth / 2., x + boxwidth / 2.], first_quartile, last_quartile, color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
        elif orientation == 'horizontal':
            ax.vlines(median, x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.vlines([first_quartile, last_quartile], x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
            ax.hlines([x - boxwidth / 2., x + boxwidth / 2.], first_quartile, last_quartile, color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)

        # plot outliers
        if show_outliers:
            if outlier_limit > 0:
                x_arr_outliers = x * np.ones_like(y_data_outliers)
                if orientation == 'vertical':
                    ax.plot(x_arr_outliers, y_data_outliers, '.', markerfacecolor='gray', markeredgecolor='none',
                            markersize=1)
                elif orientation == 'horizontal':
                    ax.plot(y_data_outliers, x_arr_outliers, '.', markerfacecolor='gray', markeredgecolor='none',
                            markersize=1)


def scatter_line(ax, x, lines, color=(0.001, 0.001, 0.001), shading='95conf', show_lines=False, use='median',
                 show_mean=False, alpha=0.3):
    if type(lines) is list:
        lines = np.array(lines)
    import flystat.resampling
    line_lo, line_hi = flystat.resampling.bootstrap_confidence_for_lines(lines, use='mean', iterations=1000)
    line_mean = np.mean(lines, axis=0)
    if show_mean:
        ax.plot(x, line_mean, color=color)
    ax.fill_between(x, line_lo, line_hi, facecolor=color, edgecolor='none', alpha=alpha)

    if show_lines:
        for line in lines:
            ax.plot(x, line, color=color, linewidth=0.5)


def get_optimized_scatter_distance(y_data, xwidth, y_scale_factor=1, seed=0, resolution=20):
    '''
    y_scale_factory - helps spread data about more
    '''
    xvals = [seed]
    r = np.linspace(-1 * xwidth / 2., xwidth / 2., resolution)

    y_data_copy = copy.copy(y_data)

    y_data_copy -= np.mean(y_data_copy)
    yrange = np.max(y_data_copy) - np.min(y_data_copy)
    factor = yrange / float(xwidth)
    y_data_copy /= factor
    y_data_copy *= y_scale_factor
    y_data_copy += 1

    for y in y_data_copy[1:]:
        q = sympy.symbols('q')
        pt = [q, y * y_scale_factor]

        all_points = np.vstack((xvals, y_data_copy[0:len(xvals)] * y_scale_factor)).T
        diff = sympy.Matrix(all_points - np.array(pt))

        distances = [(diff[i, :].norm(2)) ** 0.25 for i in range(diff.shape[0])]

        dist = np.sum(distances)
        # d_dist = sympy.diff(dist)
        v = [dist.subs({'q': ri}) for ri in r]
        rn = 0  # 2*(np.random.random()-0.5)*xwidth*0.1
        xvals.append(r[np.argmax(v)] + rn)

    return np.array(xvals)


def scatter_box(ax, x, y_data, xwidth=0.3, ywidth=0.1, color='black', edgecolor='none', flipxy=False, shading='95conf',
                alpha=0.3, markersize=5, linewidth=1, marker_linewidth=0, use='median', optimize_scatter_distance=False,
                optimize_scatter_distance_resolution=20, optimize_scatter_distance_y_scale=1, hide_markers=False,
                scatter_color=None, scatter_cmap='jet', scatter_norm_minmax=[0, 1], random_scatter=True,
                rasterized=True):
    '''
    shading - can show quartiles, or 95% conf, or none
    optimize_scatter_distance - maximize distance between points, instead of randomizing. May take a long time.
    '''
    if not hasattr(x, '__len__'):
        if use == 'median':
            mean = np.median(y_data)
        elif use == 'mean':
            mean = np.mean(y_data)
        y_data.sort()
        n = len(y_data)
        bottom_quartile = y_data[int(.25 * n)]
        top_quartile = y_data[int(.75 * n)]

        if random_scatter:
            if not optimize_scatter_distance:
                xvals = [x + np.random.random() * xwidth * 2 - xwidth for yi in range(len(y_data))]
            else:
                xvals = get_optimized_scatter_distance(y_data, xwidth, resolution=optimize_scatter_distance_resolution,
                                                       y_scale_factor=optimize_scatter_distance_y_scale)
                xvals += x
        else:
            xvals = [x + 0 for yi in range(len(y_data))]

        if shading == '95conf':
            import flystat.resampling
            conf_interval = flystat.resampling.bootstrap_confidence_intervals_from_data(y_data, use=use)

        if not flipxy:
            if shading != 'none':
                ax.hlines([mean], x - xwidth, x + xwidth, colors=[color], linewidth=linewidth)
            if shading == 'quartiles':
                ax.fill_between([x - xwidth, x + xwidth], [bottom_quartile, bottom_quartile],
                                [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
            elif shading == '95conf':
                ax.fill_between([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none', alpha=alpha)
            if not hide_markers:
                if scatter_color is not None:  # len is a check to rgb tuples
                    ax.scatter(xvals, y_data, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                               linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                               vmax=scatter_norm_minmax[1], rasterized=rasterized)
                else:
                    ax.plot(xvals, y_data, 'o', markerfacecolor=color, markeredgecolor=edgecolor, markersize=markersize,
                            rasterized=rasterized)
        else:
            if shading != 'none':
                ax.vlines([mean], x - xwidth, x + xwidth, colors=[color], linewidth=linewidth)
            if shading == 'quartiles':
                ax.fill_betweenx([x - xwidth, x + xwidth], [bottom_quartile, bottom_quartile],
                                 [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
            elif shading == '95conf':
                ax.fill_betweenx([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                 [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none', alpha=alpha)
            if not hide_markers:
                if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                    ax.scatter(y_data, xvals, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                               linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                               vmax=scatter_norm_minmax[1], rasterized=rasterized)
                else:
                    ax.plot(y_data, xvals, 'o', markerfacecolor=color, markeredgecolor=edgecolor, markersize=markersize,
                            rasterized=rasterized)

    else:
        for i in range(len(x)):
            if use == 'median':
                mean = np.median(y_data[i])
            elif use == 'mean':
                mean = np.mean(y_data[i])
            y_data[i].sort()
            n = len(y_data[i])
            bottom_quartile = y_data[i][int(.25 * n)]
            top_quartile = y_data[i][int(.75 * n)]

            if not optimize_scatter_distance:
                xvals = [x[i] + np.random.random() * xwidth * 2 - xwidth for yi in range(len(y_data))]
            else:
                xvals = get_optimized_scatter_distance(y_data, xwidth, resolution=optimize_scatter_distance_resolution,
                                                       y_scale_factor=optimize_scatter_distance_y_scale)
                xvals += x[i]

            if shading == '95conf':
                import flystat.resampling
                conf_interval = flystat.resampling.bootstrap_confidence_intervals_from_data(y_data[i], use=use)

            if not flipxy:
                if shading != 'none':
                    ax.hlines([mean], x[i] - xwidth, x[i] + xwidth, colors=[color], linewidth=linewidth)
                if shading == 'quartiles':
                    ax.fill_between([x[i] - xwidth, x[i] + xwidth], [bottom_quartile, bottom_quartile],
                                    [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
                elif shading == '95conf':
                    ax.fill_between([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                    [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none',
                                    alpha=alpha)
                if not hide_markers:
                    if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                        ax.scatter(xvals, y_data, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                                   linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                                   vmax=scatter_norm_minmax[1], rasterized=rasterized)
                    else:
                        ax.plot(xvals, y_data, 'o', markerfacecolor=color, markeredgecolor=edgecolor,
                                markersize=markersize, rasterized=rasterized)
            else:
                if shading != 'none':
                    ax.vlines([mean], x[i] - xwidth, x[i] + xwidth, colors=[color], linewidth=linewidth)
                if shading == 'quartiles':
                    ax.fill_betweenx([x[i] - xwidth, x[i] + xwidth], [bottom_quartile, bottom_quartile],
                                     [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
                elif shading == '95conf':
                    ax.fill_betweenx([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                     [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none',
                                     alpha=alpha)
                if not hide_markers:
                    if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                        ax.scatter(y_data, xvals, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                                   linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                                   vmax=scatter_norm_minmax[1], rasterized=rasterized)
                    else:
                        ax.plot(y_data, xvals, 'o', markerfacecolor=color, markeredgecolor=edgecolor,
                                markersize=markersize, rasterized=rasterized)


###################################################################################################
# 2D "heatmap" Histogram
###################################################################################################

def histogram2d(ax, x, y, bins=100, normed=False, histrange=None, weights=None, logcolorscale=False, colormap='jet',
                interpolation='nearest', colornorm=None, xextent=None, yextent=None, norm_rows=False,
                norm_columns=False, return_img=False):
    # the following paramters get passed straight to numpy.histogram2d
    # x, y, bins, normed, histrange, weights

    # from numpy.histogram2d:
    '''

    weights - if weights is not None, this function will plot a histogram of the weight values normalized by an unweighted histogram.

    Parameters
    ----------
    x : array_like, shape(N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:

        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).

    range : array_like, shape(2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like, shape(N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.
    '''
    hist, x_binned, y_binned = np.histogram2d(x, y, bins, normed=normed, range=histrange)

    if weights is not None:
        hist_weights, x, y = np.histogram2d(x, y, bins, normed=normed, range=histrange, weights=weights)
        indices_0 = np.where(np.array(hist) == 0)
        hist = hist_weights / hist
        hist[indices_0] = 0

    x = x_binned
    y = y_binned

    if logcolorscale:
        hist = np.log(hist + 1)  # the plus one solves bin=0 issues

    if xextent is None:
        xextent = [x[0], x[-1]]
    if yextent is None:
        yextent = [y[0], y[-1]]

    img = hist.T

    if norm_rows:
        for r in range(img.shape[0]):
            try:
                mi = np.min(img[r, :])
            except:
                mi = 0
            img[r, :] -= mi
            try:
                ma = np.max(img[r, :])
            except:
                ma = 0
                totalrow = 1
            if ma != 0:
                img[r, :] /= float(ma)

            # total = np.sum(img[r,:])
            # if total > 0:
            #    img[r,:] /= float(total)
            ###print mi, ma, np.min(img[r,:]), np.max(img[r,:])

    if norm_columns:
        for c in range(img.shape[1]):
            mi = np.min(img[:, c])
            img[:, c] -= mi
            ma = np.max(img[:, c])
            if ma != 0:
                img[:, c] /= ma
            ##print mi, ma, np.min(img[:,c]), np.max(img[:,c])

    if colornorm is not None:
        colornorm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    else:
        colornorm = matplotlib.colors.Normalize(np.min(np.min(img)), np.max(np.max(img)))
        ##print 'color norm: ', np.min(np.min(img)), np.max(np.max(img))

    # make the heatmap
    cmap = plt.get_cmap(colormap)
    ax.imshow(img,
              cmap=cmap,
              extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
              origin='lower',
              interpolation=interpolation,
              norm=colornorm)
    ax.set_aspect('auto')

    if return_img:
        return img


###################################################################################################
# Colorbar
###################################################################################################

def colorbar(ax=None, ticks=None, ticklabels=None, colormap='jet', aspect='auto', orientation='vertical', filename=None,
             flipspine=False, show_spine=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if ticks is None:
        ticks = np.linspace(-1, 1, 5, endpoint=True)

    if aspect != 'auto':
        ax.set_aspect('equal')
    else:
        ax.set_aspect('auto')

    # horizontal
    if orientation == 'horizontal':
        xlim = (ticks[0], ticks[-1])
        if aspect != 'auto':
            yrange = (ticks[-1] - ticks[0]) / float(aspect)
        else:
            yrange = 1
        ylim = (0, yrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad, grad))
        if show_spine:
            if not flipspine:
                adjust_spines(ax, ['bottom'], xticks=ticks)
            else:
                adjust_spines(ax, ['top'], xticks=ticks)
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)

    # vertical
    if orientation == 'vertical':
        ylim = (ticks[0], ticks[-1])
        if aspect != 'auto':
            xrange = (ticks[-1] - ticks[0]) / float(aspect)
        else:
            xrange = 1
        xlim = (0, xrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad, grad)).T
        if show_spine:
            if not flipspine:
                adjust_spines(ax, ['right'], yticks=ticks)
            else:
                adjust_spines(ax, ['left'], yticks=ticks)
        if ticklabels is not None:
            ax.set_yticklabels(ticklabels)

    if not show_spine:
        adjust_spines(ax, [])
    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(im,
              cmap=cmap,
              extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]),
              origin='lower',
              interpolation='bicubic')

    ax.set_xlim(xlim[0], xlim[-1])
    ax.set_ylim(ylim[0], ylim[-1])

    if aspect != 'auto':
        ax.set_aspect('equal')
    else:
        ax.set_aspect('auto')

    if filename is not None:
        fig.savefig(filename, format='pdf')


###################################################################################################
# Scatter Plot (with PatchCollections of circles) : more control than plotting with 'dotted' style with "plot"
###################################################################################################

def get_circles_for_scatter(x, y, color='black', edgecolor='none', colormap='jet', radius=0.01, colornorm=None, alpha=1,
                            radiusnorm=None, maxradius=1, minradius=0):
    cmap = plt.get_cmap(colormap)
    if colornorm is not None:
        colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)

    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    circles = [None for i in range(len(x))]
    for i, pt in enumerate(points):
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i]) * (maxradius - minradius) + minradius
        else:
            r = radius
        circles[i] = patches.Circle(pt, radius=r)

    # make a collection of those circles
    cc = PatchCollection(circles, cmap=cmap, norm=colornorm)  # potentially useful option: match_original=True

    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)

    return cc


def get_ellipses_for_scatter(ax, x, y, color='black', edgecolor='none', colormap='jet', cmap=None, radius=0.01,
                             colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0):
    # get ellipse size to make it a circle given axes
    x0, y0 = ax.transAxes.transform((ax.get_ylim()[0], ax.get_xlim()[0]))
    x1, y1 = ax.transAxes.transform((ax.get_ylim()[1], ax.get_xlim()[1]))
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)

    if cmap is None:
        cmap = plt.get_cmap(colormap)
        if colornorm is not None:
            colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)

    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    ellipses = [None for i in range(len(x))]
    for i, pt in enumerate(points):
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i]) * (maxradius - minradius) + minradius
        else:
            r = radius
        width = r * 2 * maxd / dx
        height = r * 2 * maxd / dy
        ellipses[i] = patches.Ellipse(pt, width, height)

    # make a collection of those circles
    cc = PatchCollection(ellipses, cmap=cmap, norm=colornorm)  # potentially useful option: match_original=True

    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)

    return cc


def scatter(ax, x, y, color='black', colormap='jet', cmap=None, edgecolor='none', radius=0.01, colornorm=None, alpha=1,
            radiusnorm=None, maxradius=1, minradius=0, xlim=None, ylim=None, use_ellipses=True, zorder=0):
    '''
    Make a colored scatter plot

    NOTE: the scatter points will only be circles if you use_ellipses=True, and if you do not change xlim/ylim or the relative size of the axes after the function has been called.

    x           -- np.array
    y           -- np.array
    color       -- matplotlib color (eg. string), or np.array of values
    colormap    -- matplotlib coloramp name (eg. 'jet')
    edgecolor   -- matplotlib color for edges (eg. string) - default is 'none', which means no edge
    radius      -- radius of circles to plot - in units of the axes - either a float, or np.array of floats
    colornorm   -- min and max you would like colors in the color array normalized to, eg [0,1], default is to scale to min/max of color array
    alpha       -- transparancy, float btwn 0 and 1
    radiusnorm  -- min/max you would like radius array to be normalized to
    maxradius   -- max radius size you would like
    minradius   -- min radius size you would like
    xlim/ylim   -- x and y limits of axes, default is scaled to min/max of x and y
    use_ellipses-- adjust scatter point so that they are circles, even if aspect is not equal. Only works if you do not change xlim/ylim or axes shape after calling this function

    '''
    # color can be array-like, or a matplotlib color
    # I can't figure out how to control alpha through the individual circle patches.. it seems to get overwritten by the collection. low priority!
    if type(x) is not list:
        x = x.flatten()
    if type(y) is not list:
        y = y.flatten()

    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if use_ellipses:
        cc = get_ellipses_for_scatter(ax, x, y, color=color, edgecolor=edgecolor, colormap=colormap, cmap=cmap,
                                      radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm,
                                      maxradius=maxradius, minradius=minradius)

    else:
        cc = get_circles_for_scatter(x, y, color=color, edgecolor=edgecolor, colormap=colormap, radius=radius,
                                     colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius,
                                     minradius=minradius)

    # add collection to axis
    ax.add_collection(cc)


def scattered_histogram(ax, bin_leftedges, data_list, bin_width=0.6, s=1, color='green', linewidths=None, alpha=1,
                        draw_median=False, median_color=None, fill_quartiles=True, quartile_alpha=0.5,
                        median_linewidth=2, draw_continuous_median=True, flip_xy=False, show_scatter=True,
                        lower_quartile=0.25, upper_quartile=0.75, medianmarkersize=5):
    '''
    data_list - should be a list of lists, equal in length to bin_leftedges. Each index [i] of data_list corresponds to bin_leftedges[i], and contains a list of the data that belongs in that bin.
    '''
    if median_color is None:
        median_color = color

    u = uniform_distribution(0, bin_width)

    scattered_points_x = []
    scattered_points_y = []

    for i, data in enumerate(data_list):
        for value in data:
            x = bin_leftedges[i] + u.rvs()
            y = value

            if not flip_xy:
                scattered_points_x.append(x)
                scattered_points_y.append(y)
            else:
                scattered_points_y.append(x)
                scattered_points_x.append(y)

    if show_scatter:
        ax.scatter(scattered_points_x, scattered_points_y, s=s, facecolor=color, edgecolor='none',
                   linewidths=linewidths, alpha=alpha, zorder=-100)

    if draw_median:
        bin_centers = bin_leftedges + bin_width / 2.

        for i, data in enumerate(data_list):
            if len(data) > 0:
                data.sort()
                median = np.median(data)
                ax.plot([bin_centers[i] - bin_width / 2., bin_centers[i] + bin_width / 2.], [median, median],
                        color=median_color, linewidth=median_linewidth)

                if fill_quartiles:
                    lower_quartile_index = int(len(data) * lower_quartile)
                    lower_quartile = data[lower_quartile_index]
                    upper_quartile_index = int(len(data) * upper_quartile)
                    upper_quartile = data[upper_quartile_index]
                    ax.fill_between([bin_centers[i] - bin_width / 2., bin_centers[i] + bin_width / 2.],
                                    [lower_quartile, lower_quartile], [upper_quartile, upper_quartile],
                                    facecolor=median_color, alpha=quartile_alpha, edgecolor='none')

    if draw_continuous_median:
        bin_centers = bin_leftedges + bin_width / 2.

        medians = []
        lower_quartiles = []
        upper_quartiles = []

        for i, data in enumerate(data_list):
            if len(data) > 0:
                data.sort()
                medians.append(np.median(data))

                lower_quartile_index = int(len(data) / 4.)
                lower_quartiles.append(data[lower_quartile_index])
                upper_quartile_index = int(len(data) * 3 / 4.)
                upper_quartiles.append(data[upper_quartile_index])
            else:
                medians.append(np.nan)
                lower_quartiles.append(np.nan)
                upper_quartiles.append(np.nan)

        if not flip_xy:
            ax.plot(bin_centers, medians, color=median_color, linewidth=median_linewidth)
        else:
            ax.plot(medians, bin_centers, color=median_color, linewidth=median_linewidth)
            ax.plot(medians, bin_centers, '.', markersize=medianmarkersize, markerfacecolor=median_color,
                    markeredgecolor=median_color)

        if fill_quartiles:
            if not flip_xy:
                ax.fill_between(bin_centers, lower_quartiles, upper_quartiles, facecolor=median_color,
                                alpha=quartile_alpha, edgecolor='none')
            else:
                ax.fill_betweenx(bin_centers, lower_quartiles, upper_quartiles, facecolor=median_color,
                                 alpha=quartile_alpha, edgecolor='none')


def plot_confidence_interval(ax, x, y, confidence_interval_95, confidence_interval_50=None, width=0.3, color='blue',
                             linewidth=3, alpha95=0.3, alpha50=0.5):
    xpts = [x - width / 2., x + width / 2.]
    ax.fill_between(xpts, confidence_interval_95[0], confidence_interval_95[-1], edgecolor='none', facecolor=color,
                    alpha=alpha95)
    if confidence_interval_50 is not None:
        ax.fill_between(xpts, confidence_interval_50[0], confidence_interval_50[-1], edgecolor='none', facecolor=color,
                        alpha=alpha50)
    # ax.fill_between(xpts, y-linewidth/2., y+linewidth/2., edgecolor='none', facecolor=color, alpha=1)
    ax.hlines([y], [x - width / 2.], [x + width / 2.], color=color, alpha=1, linewidth=linewidth)


# written by Floris van Breugel, with some help from Andrew Straw and Will Dickson
# dependencies for LaTex rendering: texlive, ghostscript, dvipng, texlive-latex-extra

# general imports
import matplotlib
##print matplotlib.__version__
##print 'recommended version: 1.1.1 or greater'

###################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# used for colorline
from matplotlib.collections import LineCollection

# used in histogram
from scipy.stats import norm as gaussian_distribution
from scipy.stats import uniform as uniform_distribution
from scipy import signal

# used in colorbar
import matplotlib.colorbar

# used in scatter
from matplotlib.collections import PatchCollection

import sympy
import copy

# not used
# import scipy.optimize
# import scipy.stats.distributions as distributions

###################################################################################################
# Misc Info
###################################################################################################

# FUNCTIONS contained in this file:
# adjust_spines
# colorline
# histogram
# histogram2d (heatmap)
# boxplot
# colorbar (scale for colormap stuff), intended for just generating a colorbar for use in illustrator figure assembly


# useful links:
# colormaps: http://www.scipy.org/Cookbook/Matplotlib/Show_colormaps


cmap_bgy = matplotlib.colors.LinearSegmentedColormap.from_list("my_new_colormap", [(0, 0, .4), (0, .7, .7), (1, 1, .3)])


###################################################################################################
# Adjust Spines (Dickinson style, thanks to Andrew Straw)
###################################################################################################

# NOTE: smart_bounds is disabled (commented out) in this function. It only works in matplotlib v >1.
# to fix this issue, try manually setting your tick marks (see example below)
def adjust_spines(ax, spines, spine_locations={}, smart_bounds=True, xticks=None, yticks=None, linewidth=1):
    if type(spines) is not list:
        spines = [spines]

    # get ticks
    if xticks is None:
        xticks = ax.get_xticks()
    if yticks is None:
        yticks = ax.get_yticks()

    spine_locations_dict = {'top': 10, 'right': 10, 'left': 10, 'bottom': 10}
    for key in spine_locations.keys():
        spine_locations_dict[key] = spine_locations[key]

    if 'none' in spines:
        for loc, spine in ax.spines.iteritems():
            spine.set_color('none')  # don't draw spine
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        return

    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            spine.set_position(('outward', spine_locations_dict[loc]))  # outward by x points
            spine.set_linewidth(linewidth)
            spine.set_color('black')
        else:
            spine.set_color('none')  # don't draw spine

    # smart bounds, if possible
    if int(matplotlib.__version__[0]) > 0 and smart_bounds:
        for loc, spine in ax.spines.items():
            ticks = None
            if loc in ['left', 'right']:
                ticks = yticks
            if loc in ['top', 'bottom']:
                ticks = xticks
            if ticks is not None and len(ticks) > 0:
                spine.set_bounds(ticks[0], ticks[-1])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    if 'top' in spines:
        ax.xaxis.set_ticks_position('top')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

    if 'left' in spines or 'right' in spines:
        ax.set_yticks(yticks)
    if 'top' in spines or 'bottom' in spines:
        ax.set_xticks(xticks)

    for line in ax.get_xticklines() + ax.get_yticklines():
        # line.set_markersize(6)
        line.set_markeredgewidth(linewidth)


###################################################################################################
# Map to Color
###################################################################################################

def get_color_transformer(norm=(0, 1), colormap='jet', clip=True):
    '''
    returns a function that will return a color value (4-tuple) from the given color map based on a single input, which is scaled to the range given by norm. clip is passed to plt.Normalize, default is True.
    '''

    def color_transformer(v):
        Norm = plt.Normalize(norm[0], norm[1], clip=clip)
        cmap = plt.get_cmap(colormap)
        return cmap(Norm(v))

    return color_transformer


###################################################################################################
# Colorline
###################################################################################################

# plot a line in x and y with changing colors defined by z, and optionally changing linewidths defined by linewidth
def colorline(ax, x, y, z, linewidth=1, colormap='jet', norm=None, zorder=1, alpha=1, linestyle='solid', cmap=None,
              hide_nan_indices=True, hack_round_caps=False, axis_size_inches=None, cap_size_radius_adjustment=2):
    '''
    hack_round_caps - extend line segments so that line appears continuous. beta mode.
    axis_size_inches - used for hack_projected_cap (x,y). Not well implemented.
    '''
    if cmap is None:
        cmap = plt.get_cmap(colormap)

    if type(linewidth) is list or type(linewidth) is np.array or type(linewidth) is np.ndarray:
        linewidths = linewidth
    else:
        linewidths = np.ones_like(z) * linewidth

    if norm is None:
        norm = plt.Normalize(np.min(z), np.max(z))
    else:
        norm = plt.Normalize(norm[0], norm[1])

    '''
    if self.hide_colorbar is False:
        if self.cb is None:
            self.cb = matplotlib.colorbar.ColorbarBase(self.ax1, cmap=cmap, norm=norm, orientation='vertical', boundaries=None)
    '''

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be numlines x points per line x 2 (x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2).astype(float)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    ##print segments
    if hide_nan_indices == True:
        nanindices_x = np.where(np.isnan(x))[0].tolist()
        nanindices_y = np.where(np.isnan(y))[0].tolist()
        nanindices_x.extend(nanindices_y)
        nanindices = np.unique(nanindices_x)
        segments = np.delete(segments, nanindices, axis=0)
        z = np.delete(z, nanindices, axis=0)

    # Create the line collection object, setting the colormapping parameters.
    # Have to set the actual values used for colormapping separately.
    ordered_zorder = (zorder - 1)  # + z/float(len(z))
    if hasattr(linewidth, '__iter__'):
        lc = LineCollection(segments, linewidths=linewidths, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha,
                            linestyles=linestyle)
        lc.set_array(z)
        lc.set_zorder(ordered_zorder)  # .tolist()
        lc.set_linewidth(linewidth)
    else:
        lc = LineCollection(segments, linewidths=linewidth, cmap=cmap, norm=norm, zorder=zorder, alpha=alpha,
                            linestyles=linestyle)
        lc.set_array(z)
        lc.set_zorder(ordered_zorder)  # .tolist()
    ax.add_collection(lc)

    if hack_round_caps:
        ax.scatter(x, y, color=cmap(norm(z)), s=linewidth ** 2, edgecolor='none', zorder=(z - 10).tolist())


###################################################################################################
# Colorline with heading
###################################################################################################

def get_wedges_for_heading_plot(x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet',
                                colornorm=None, size_radius_range=(0.01, .1), size_radius_norm=None, edgecolor='none',
                                alpha=1, flip=True, deg=True, nskip=0, center_offset_fraction=0.75):
    '''
    Returns a Patch Collection of Wedges, with arbitrary color and orientation

    Outputs:
    Patch Collection

    Inputs:
    x, y        - x and y positions (np.array or list, each of length N)
    color       - values to color wedges by (np.array or list, length N), OR color string.
       colormap - specifies colormap to use (string, eg. 'jet')
       norm     - specifies range you'd like to normalize to,
                  if none, scales to min/max of color array (2-tuple, eg. (0,1) )
    orientation - angles are in degrees, use deg=False to convert radians to degrees
    size_radius - radius of wedge, in same units as x, y. Can be list or np.array, length N, for changing sizes
       size_radius_norm - specifies range you'd like to normalize size_radius to, if size_radius is a list/array
                  should be tuple, eg. (0.01, .1)
    size_angle  - angular extent of wedge, degrees. Can be list or np.array, length N, for changing sizes
    edgecolor   - color for lineedges, string or np.array of length N
    alpha       - transparency (single value, between 0 and 1)
    flip        - flip orientations by 180 degrees, default = True
    nskip       - allows you to skip between points to make the points clearer, nskip=1 skips every other point
    center_offset_fraction  - (float in range (0,1) ) - 0 means (x,y) is at the tip, 1 means (x,y) is at the edge
    '''
    cmap = plt.get_cmap(colormap)

    # norms
    if colornorm is None and type(color) is not str:
        colornorm = plt.Normalize(np.min(color), np.max(color))
    elif type(color) is not str:
        colornorm = plt.Normalize(colornorm[0], colornorm[1])
    if size_radius_norm is None:
        size_radius_norm = plt.Normalize(np.min(size_radius), np.max(size_radius), clip=True)
    else:
        size_radius_norm = plt.Normalize(size_radius_norm[0], size_radius_norm[1], clip=True)

    indices_to_plot = np.arange(0, len(x), nskip + 1)

    # fix orientations
    if type(orientation) is list:
        orientation = np.array(orientation)
    if deg is False:
        orientation = orientation * 180. / np.pi
    if flip:
        orientation += 180

    flycons = []
    n = 0
    for i in indices_to_plot:
        # wedge parameters
        if type(size_radius) is list or type(size_radius) is np.array or type(size_radius) is np.ndarray:
            r = size_radius_norm(size_radius[i]) * (size_radius_range[1] - size_radius_range[0]) + size_radius_range[0]
        else:
            r = size_radius

        if type(size_angle) is list or type(size_angle) is np.array or type(size_angle) is np.ndarray:
            angle_swept = size_radius[i]
        else:
            angle_swept = size_radius
        theta1 = orientation[i] - size_angle / 2.
        theta2 = orientation[i] + size_angle / 2.

        center = [x[i], y[i]]
        center[0] -= np.cos(orientation[i] * np.pi / 180.) * r * center_offset_fraction
        center[1] -= np.sin(orientation[i] * np.pi / 180.) * r * center_offset_fraction

        wedge = patches.Wedge(center, r, theta1, theta2)
        flycons.append(wedge)

    # add collection and color it
    pc = PatchCollection(flycons, cmap=cmap, norm=colornorm)

    # set properties for collection
    pc.set_edgecolors(edgecolor)
    pc.set_linewidth(0.25)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        if type(color) is list:
            color = np.asarray(color)
        pc.set_array(color[indices_to_plot])
    else:
        pc.set_facecolors(color)
    pc.set_alpha(alpha)

    return pc


def colorline_with_heading(ax, x, y, color, orientation, size_radius=0.1, size_angle=20, colormap='jet', colornorm=None,
                           size_radius_range=(0.01, .1), size_radius_norm=None, edgecolor='none', alpha=1, flip=True,
                           deg=True, nskip=0, use_center='center', show_centers=True, center_offset_fraction=0.75,
                           center_point_size=2):
    """
    Plots a trajectory with colored wedge shapes to indicate orientation.
    See function get_wedges_for_heading_plot for details

    Additional options:

    show_centers      - (bool) - show a black dot where the actual point is - shows where the center of the wedge is
    center_point_size - markersize for center, if show_centers
    """

    pc = get_wedges_for_heading_plot(x, y, color, orientation, size_radius=size_radius, size_angle=size_angle,
                                     colormap=colormap, colornorm=colornorm, size_radius_range=size_radius_range,
                                     size_radius_norm=size_radius_norm, edgecolor=edgecolor, alpha=alpha, flip=flip,
                                     deg=deg, nskip=nskip, center_offset_fraction=center_offset_fraction)

    ax.add_collection(pc)

    if show_centers:
        indices_to_plot = np.arange(0, len(x), nskip + 1)
        ax.plot(x[indices_to_plot], y[indices_to_plot], '.', color='black', markersize=center_point_size)


###################################################################################################
# Histograms
###################################################################################################

# first some helper functions
def custom_hist_rectangles(hist, leftedges, width, bottomedges=None, facecolor='green', edgecolor='none', alpha=1,
                           alignment='vertical'):
    linewidth = 1
    if edgecolor == 'none':
        linewidth = 0  # hack needed to remove edges in matplotlib.version 1.0+

    if bottomedges is None:
        bottomedges = np.zeros_like(leftedges)

    if type(width) is not list:
        width = [width for i in range(len(hist))]

    if type(facecolor) is not list:
        facecolor = [facecolor for i in range(len(hist))]
    if type(edgecolor) is not list:
        edgecolor = [edgecolor for i in range(len(hist))]

    rects = [None for i in range(len(hist))]

    if alignment == 'vertical':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle([leftedges[i], bottomedges[i]], width[i], hist[i], facecolor=facecolor[i],
                                         edgecolor=edgecolor[i], alpha=alpha, linewidth=linewidth)
    elif alignment == 'horizontal':
        for i in range(len(hist)):
            rects[i] = patches.Rectangle([bottomedges[i], leftedges[i]], hist[i], width[i], facecolor=facecolor[i],
                                         edgecolor=edgecolor[i], alpha=alpha, linewidth=linewidth)

    return rects


def bootstrap_histogram(xdata, bins, normed=False, n=None, return_raw=False):
    if type(xdata) is not np.ndarray:
        xdata = np.array(xdata)

    if n is None:
        n = len(xdata)
    else:
        if n > len(xdata):
            n = len(xdata)
    hist_list = np.zeros([n, len(bins) - 1])

    for i in range(n):
        # Choose #sample_size members of d at random, with replacement
        choices = np.random.random_integers(0, len(xdata) - 1, n)
        xsample = xdata[choices]
        hist = np.histogram(xsample, bins, normed=normed)[0].astype(float)
        hist_list[i, :] = hist

    hist_mean = np.mean(hist_list, axis=0)
    hist_std = np.std(hist_list, axis=0)

    ##print 'bootstrapped: '
    ##print
    ##print hist_mean
    ##print
    ##print hist_std

    if return_raw:
        return hist_list
    else:
        return hist_mean, hist_std


def histogram(ax, data_list, bins=10, bin_width_ratio=0.6, colors='green', edgecolor='none', bar_alpha=0.7,
              curve_fill_alpha=0.4, curve_line_alpha=0.8, curve_butter_filter=[3, 0.3], return_vals=False,
              show_smoothed=True, normed=False, normed_occurences=False, bootstrap_std=False, bootstrap_line_width=0.5,
              exponential_histogram=False, smoothing_range=None, smoothing_bins_to_exclude=[], binweights=None,
              n_bootstrap_samples=None, alignment='vertical', peak_trace_alpha=0, show_peak_curve=False,
              data_from_which_to_calculate_binweights=None, data_to_which_calculated_binweights_should_apply='all',
              weight_distributions=None):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    data_to_which_calculated_binweights_should_apply -- list of indices corresponding to datasets which should be normalized by the binweights determined by data_from_which_to_calculate_binweights
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range

    if type(normed) is not list:
        normed = [normed for i in range(len(data_list))]

    if weight_distributions is None:
        weight_distributions = [1 for i in range(len(data_list))]

    if type(bar_alpha) is not list:
        bar_alpha = [bar_alpha for i in range(len(colors))]

    n_bars = float(len(data_list))
    if type(bins) is int:
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])

        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)

    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]

    bin_centers = np.diff(bins) / 2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1 - bin_width_ratio) * bin_width / 2.
    bar_width = (bin_width - 2 * bin_width_buff) / n_bars

    butter_b, butter_a = signal.butter(curve_butter_filter[0], curve_butter_filter[1])

    data_hist_list = []
    if return_vals:
        data_curve_list = []
        data_hist_std_list = []

    # first get max number of occurences
    max_occur = []
    for i, data in enumerate(data_list):
        data_hist = np.histogram(data, bins=bins, normed=normed[i], weights=None)[0].astype(float)
        max_occur.append(np.max(data_hist))
    max_occur = np.max(np.array(max_occur))

    if data_from_which_to_calculate_binweights is not None:
        ##print 'calculating bin weights'
        binweights = np.histogram(data_from_which_to_calculate_binweights, bins=bins, normed=False)[0].astype(float)
        binweights += np.min(binweights) * 1e-10  # to make sure we don't get divide by zero errors
        binweights = binweights ** -1
        if data_to_which_calculated_binweights_should_apply == 'all':
            binweights = [binweights for i in range(len(data_list))]
        else:
            tmp = []
            for i in range(len(data_list)):
                if i in data_to_which_calculated_binweights_should_apply:
                    tmp.append(binweights)
                else:
                    tmp.append(1)
            binweights = tmp

    for i, data in enumerate(data_list):

        if bootstrap_std:
            data_hist, data_hist_std = bootstrap_histogram(data, bins=bins, normed=normed[i], n=n_bootstrap_samples)
        else:
            data_hist = np.histogram(data, bins=bins, normed=normed[i])[0].astype(float)

        if binweights is not None:
            data_hist *= binweights[i]
            if normed[i]:
                data_hist /= np.sum(binweights[i])

        if exponential_histogram:
            data_hist = np.log(data_hist + 1)

        data_hist = data_hist / float(weight_distributions[i])

        if normed_occurences is not False:
            if normed_occurences == 'total':
                data_hist /= max_occur
                if bootstrap_std:
                    data_hist_std /= max_occur
            else:
                div = float(np.max(data_hist))
                ##print div
                data_hist /= div
                if bootstrap_std:
                    data_hist_std /= div

        rects = custom_hist_rectangles(data_hist, bins[0:-1] + bar_width * i + bin_width_buff, width=bar_width,
                                       facecolor=colors[i], edgecolor=edgecolor, alpha=bar_alpha[i],
                                       alignment=alignment)
        if bootstrap_std:
            for j, s in enumerate(data_hist_std):
                x = bins[j] + bar_width * i + bin_width_buff + bar_width / 2.
                # ax.plot([x,x], [data_hist[j], data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                # ax.plot(np.array([x,x]), np.array([data_hist[j], data_hist[j]+data_hist_std[j]]), alpha=bar_alpha, color=colors[i], linewidth=bootstrap_line_width)
                ax.vlines(x, data_hist[j], data_hist[j] + data_hist_std[j], color=colors[i],
                          linewidth=bootstrap_line_width)

                # ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=1, color='w')
                # ax.plot([x-bar_width/3., x+bar_width/3.], [data_hist[j]+data_hist_std[j],data_hist[j]+data_hist_std[j]], alpha=bar_alpha, color=colors[i])
        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)

        if show_peak_curve:
            if curve_line_alpha > 0:
                ax.plot(bins[0:-1] + bar_width * i + bin_width_buff, data_hist, color=colors[i], alpha=curve_line_alpha)
            if curve_fill_alpha > 0:
                ax.fill_between(bins[0:-1] + bar_width * i + bin_width_buff, data_hist, np.zeros_like(data_hist),
                                color=colors[i], alpha=curve_fill_alpha, zorder=-100, edgecolor='none')

        if show_smoothed:
            if smoothing_range is not None:  # in case you only want to smooth and show smoothing over a select range.
                indices_in_smoothing_range = \
                np.where((bin_centers > smoothing_range[0]) * (bin_centers < smoothing_range[-1]))[0].tolist()
            else:
                indices_in_smoothing_range = [bc for bc in range(len(bin_centers))]

            for b in smoothing_bins_to_exclude:
                try:
                    indices_in_smoothing_range.remove(b)
                except ValueError:
                    print('bin center not in indices list: ', b)
                    print('indices list: ', indices_in_smoothing_range)

            data_hist_filtered = signal.filtfilt(butter_b, butter_a, data_hist[indices_in_smoothing_range])
            interped_bin_centers = np.linspace(bin_centers[indices_in_smoothing_range[0]] - bin_width / 2.,
                                               bin_centers[indices_in_smoothing_range[-1]] + bin_width / 2., 100,
                                               endpoint=True)
            v = 100 / float(len(bin_centers))

            if alignment == 'vertical':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range],
                                                        data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b / v, butter_a / v, interped_data_hist_filtered)
                # ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_between(interped_bin_centers, interped_data_hist_filtered2,
                                    np.zeros_like(interped_data_hist_filtered2), color=colors[i],
                                    alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_bin_centers, interped_data_hist_filtered2, color=colors[i], alpha=curve_line_alpha)

            if alignment == 'horizontal':
                interped_data_hist_filtered = np.interp(interped_bin_centers, bin_centers[indices_in_smoothing_range],
                                                        data_hist_filtered)
                interped_data_hist_filtered2 = signal.filtfilt(butter_b / v, butter_a / v, interped_data_hist_filtered)
                # ax.plot(bin_centers, data_hist_filtered, color=facecolor[i])
                if curve_fill_alpha > 0:
                    ax.fill_betweenx(interped_bin_centers, interped_data_hist_filtered2,
                                     np.zeros_like(interped_data_hist_filtered2), color=colors[i],
                                     alpha=curve_fill_alpha, zorder=-100, edgecolor='none')
                if curve_line_alpha:
                    ax.plot(interped_data_hist_filtered2, interped_bin_centers, color=colors[i], alpha=curve_line_alpha)

        ax.plot(bin_centers, data_hist, color=colors[i], alpha=peak_trace_alpha)

        data_hist_list.append(data_hist)
        if return_vals:
            if bootstrap_std:
                data_hist_std_list.append(data_hist_std)

            if show_smoothed:
                data_curve_list.append([interped_bin_centers, interped_data_hist_filtered2])

    mins_of_data = [np.min(data) for data in data_list]
    maxs_of_data = [np.max(data) for data in data_list]

    mins_of_hist = [np.min(hist) for hist in data_hist_list]
    maxs_of_hist = [np.max(hist) for hist in data_hist_list]

    if alignment == 'vertical':
        ax.set_xlim(np.min(mins_of_data), np.max(maxs_of_data))
        ax.set_ylim(0, np.max(maxs_of_hist))
    elif alignment == 'horizontal':
        ax.set_ylim(np.min(mins_of_data), np.max(maxs_of_data))
        ax.set_xlim(0, np.max(maxs_of_hist))

    if return_vals and bootstrap_std is False:
        return bins, data_hist_list, data_curve_list
    elif return_vals and bootstrap_std is True:
        return bins, data_hist_list, data_hist_std_list, data_curve_list


###########

def histogram_stack(ax, data_list, bins=10, bin_width_ratio=0.8, colors='green', edgecolor='none', normed=True):
    '''
    ax          -- matplotlib axis
    data_list   -- list of data collections to histogram - if just one, either give an np.array, or soemthing like [data], where data is a list itself
    normed - normalizes the SUM of all the stacked histograms
    '''
    # smoothing_range: tuple or list or sequence, eg. (1,100). Use if you only want to smooth and show smoothing over a specific range

    n_bars = float(len(data_list))
    if type(bins) is int:
        mia = np.array([np.min(d) for d in data_list])
        maa = np.array([np.max(d) for d in data_list])
        bins = np.linspace(np.min(mia), np.max(maa), bins, endpoint=True)

    if type(colors) is not list:
        colors = [colors]
    if len(colors) != n_bars:
        colors = [colors[0] for i in range(n_bars)]

    bin_centers = np.diff(bins) / 2. + bins[0:-1]
    bin_width = np.mean(np.diff(bins))
    bin_width_buff = (1 - bin_width_ratio) * bin_width / 2.
    bar_width = (bin_width - bin_width_buff)

    data_hist_list = []

    all_data = []
    for data in data_list:
        all_data.extend(data.tolist())
    all_data_hist = np.histogram(all_data, bins=bins, normed=False)[0].astype(float)
    all_data_hist_normed = np.histogram(all_data, bins=bins, normed=True)[0].astype(float)
    binweights_for_normalizing = all_data_hist_normed / all_data_hist

    prev_data_hist = np.zeros_like(all_data_hist)
    for i, data in enumerate(data_list):

        data_hist = np.histogram(data, bins=bins, normed=False)[0].astype(float)

        if normed:
            data_hist *= binweights_for_normalizing

        rects = custom_hist_rectangles(data_hist, bins[0:-1] + bin_width_buff, bottomedges=prev_data_hist,
                                       width=bar_width, facecolor=colors[i], edgecolor=edgecolor, alpha=1)
        prev_data_hist += data_hist

        for rect in rects:
            rect.set_zorder(1)
            ax.add_artist(rect)

        data_hist_list.append(data_hist)

    ax.set_xlim(bins[0], bins[-1])
    if normed:
        ax.set_ylim(0, np.max(all_data_hist_normed) + .1 * np.max(all_data_hist_normed))
    else:
        ax.set_ylim(0, np.max(all_data_hist) + .1 * np.max(all_data_hist))


###################################################################################################
# Boxplots
###################################################################################################

def boxplot(ax, x_data, y_data_list, nbins=50, colormap='YlOrRd', colorlinewidth=2, boxwidth=1, boxlinecolor='black',
            classic_linecolor='gray', usebins=None, boxlinewidth=0.5, outlier_limit=0.01, norm=None,
            use_distribution_for_linewidth=False, min_colorlinewidth=1, show_outliers=True, show_whiskers=True,
            logcolorscale=False, orientation='vertical'):
    # if colormap is None: show a line instead of the 1D histogram (ie. a normal boxplot)
    # use_distribution_for_linewidth will adjust the linewidth according to the histogram of the distribution
    # classic_linecolor: sets the color of the vertical line that shows the extent of the data, if colormap=None
    # outlier limit: decimal % of top and bottom data that is defined as outliers (0.01 = top 1% and bottom 1% are defined as outliers)
    # show_whiskers: toggle the whiskers, if colormap is None

    if usebins is None:
        usebins = nbins
        # usebins lets you assign the bins manually, but it's the same range for each x_coordinate

    for i, y_data in enumerate(y_data_list):
        ###print len(y_data)

        # calc boxplot statistics
        median = np.median(y_data)
        ind = np.where(y_data <= median)[0].tolist()
        first_quartile = np.median(y_data[ind])
        ind = np.where(y_data >= median)[0].tolist()
        last_quartile = np.median(y_data[ind])
        ###print first_quartile, median, last_quartile

        # find outliers
        ind_sorted = np.argsort(y_data)
        bottom_limit = int(len(ind_sorted) * (outlier_limit))
        top_limit = int(len(ind_sorted) * (1 - outlier_limit))
        indices_inrange = ind_sorted[bottom_limit:top_limit]
        outliers = ind_sorted[0:bottom_limit].tolist() + ind_sorted[top_limit:len(ind_sorted) - 1].tolist()
        y_data_inrange = y_data[indices_inrange]
        y_data_outliers = y_data[outliers]
        x = x_data[i]

        # plot colorline
        if colormap is not None:
            hist, bins = np.histogram(y_data_inrange, usebins, normed=True)
            hist = hist.astype(float)
            # hist /= np.max(hist)
            x_arr = np.ones_like(bins) * x

            if logcolorscale:
                hist = np.log(hist + 1)

            if use_distribution_for_linewidth:
                colorlinewidth = hist * colorlinewidth + min_colorlinewidth

            if orientation == 'vertical':
                colorline(ax, x_arr, bins, hist, colormap=colormap, norm=norm,
                          linewidth=colorlinewidth)  # the norm defaults make it so that at each x-coordinate the colormap/linewidth will be scaled to show the full color range. If you want to control the color range for all x-coordinate distributions so that they are the same, set the norm limits when calling boxplot().
            elif orientation == 'horizontal':
                colorline(ax, bins, x_arr, hist, colormap=colormap, norm=norm, linewidth=colorlinewidth)

        elif show_whiskers:
            if orientation == 'vertical':
                ax.vlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.vlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.hlines([np.min(y_data_inrange), np.max(y_data_inrange)], x - boxwidth / 4., x + boxwidth / 4.,
                          color=classic_linecolor, linewidth=boxlinewidth / 2.)
            elif orientation == 'horizontal':
                ax.hlines(x, last_quartile, np.max(y_data_inrange), color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.hlines(x, np.min(y_data_inrange), first_quartile, color=classic_linecolor, linestyle=('-'),
                          linewidth=boxlinewidth / 2.)
                ax.vlines([np.min(y_data_inrange), np.max(y_data_inrange)], x - boxwidth / 4., x + boxwidth / 4.,
                          color=classic_linecolor, linewidth=boxlinewidth / 2.)

        # plot boxplot
        if orientation == 'vertical':
            ax.hlines(median, x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.hlines([first_quartile, last_quartile], x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
            ax.vlines([x - boxwidth / 2., x + boxwidth / 2.], first_quartile, last_quartile, color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
        elif orientation == 'horizontal':
            ax.vlines(median, x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor, linewidth=boxlinewidth)
            ax.vlines([first_quartile, last_quartile], x - boxwidth / 2., x + boxwidth / 2., color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)
            ax.hlines([x - boxwidth / 2., x + boxwidth / 2.], first_quartile, last_quartile, color=boxlinecolor,
                      linewidth=boxlinewidth / 2.)

        # plot outliers
        if show_outliers:
            if outlier_limit > 0:
                x_arr_outliers = x * np.ones_like(y_data_outliers)
                if orientation == 'vertical':
                    ax.plot(x_arr_outliers, y_data_outliers, '.', markerfacecolor='gray', markeredgecolor='none',
                            markersize=1)
                elif orientation == 'horizontal':
                    ax.plot(y_data_outliers, x_arr_outliers, '.', markerfacecolor='gray', markeredgecolor='none',
                            markersize=1)


def scatter_line(ax, x, lines, color=(0.001, 0.001, 0.001), shading='95conf', show_lines=False, use='median',
                 show_mean=False, alpha=0.3):
    if type(lines) is list:
        lines = np.array(lines)
    import flystat.resampling
    line_lo, line_hi = flystat.resampling.bootstrap_confidence_for_lines(lines, use='mean', iterations=1000)
    line_mean = np.mean(lines, axis=0)
    if show_mean:
        ax.plot(x, line_mean, color=color)
    ax.fill_between(x, line_lo, line_hi, facecolor=color, edgecolor='none', alpha=alpha)

    if show_lines:
        for line in lines:
            ax.plot(x, line, color=color, linewidth=0.5)


def get_optimized_scatter_distance(y_data, xwidth, y_scale_factor=1, seed=0, resolution=20):
    '''
    y_scale_factory - helps spread data about more
    '''
    xvals = [seed]
    r = np.linspace(-1 * xwidth / 2., xwidth / 2., resolution)

    y_data_copy = copy.copy(y_data)

    y_data_copy -= np.mean(y_data_copy)
    yrange = np.max(y_data_copy) - np.min(y_data_copy)
    factor = yrange / float(xwidth)
    y_data_copy /= factor
    y_data_copy *= y_scale_factor
    y_data_copy += 1

    for y in y_data_copy[1:]:
        q = sympy.symbols('q')
        pt = [q, y * y_scale_factor]

        all_points = np.vstack((xvals, y_data_copy[0:len(xvals)] * y_scale_factor)).T
        diff = sympy.Matrix(all_points - np.array(pt))

        distances = [(diff[i, :].norm(2)) ** 0.25 for i in range(diff.shape[0])]

        dist = np.sum(distances)
        # d_dist = sympy.diff(dist)
        v = [dist.subs({'q': ri}) for ri in r]
        rn = 0  # 2*(np.random.random()-0.5)*xwidth*0.1
        xvals.append(r[np.argmax(v)] + rn)

    return np.array(xvals)


def scatter_box(ax, x, y_data, xwidth=0.3, ywidth=0.1, color='black', edgecolor='none', flipxy=False, shading='95conf',
                alpha=0.3, markersize=5, linewidth=1, marker_linewidth=0, use='median', optimize_scatter_distance=False,
                optimize_scatter_distance_resolution=20, optimize_scatter_distance_y_scale=1, hide_markers=False,
                scatter_color=None, scatter_cmap='jet', scatter_norm_minmax=[0, 1], random_scatter=True,
                rasterized=True):
    '''
    shading - can show quartiles, or 95% conf, or none
    optimize_scatter_distance - maximize distance between points, instead of randomizing. May take a long time.
    '''
    if not hasattr(x, '__len__'):
        if use == 'median':
            mean = np.median(y_data)
        elif use == 'mean':
            mean = np.mean(y_data)
        y_data.sort()
        n = len(y_data)
        bottom_quartile = y_data[int(.25 * n)]
        top_quartile = y_data[int(.75 * n)]

        if random_scatter:
            if not optimize_scatter_distance:
                xvals = [x + np.random.random() * xwidth * 2 - xwidth for yi in range(len(y_data))]
            else:
                xvals = get_optimized_scatter_distance(y_data, xwidth, resolution=optimize_scatter_distance_resolution,
                                                       y_scale_factor=optimize_scatter_distance_y_scale)
                xvals += x
        else:
            xvals = [x + 0 for yi in range(len(y_data))]

        if shading == '95conf':
            import flystat.resampling
            conf_interval = flystat.resampling.bootstrap_confidence_intervals_from_data(y_data, use=use)

        if not flipxy:
            if shading != 'none':
                ax.hlines([mean], x - xwidth, x + xwidth, colors=[color], linewidth=linewidth)
            if shading == 'quartiles':
                ax.fill_between([x - xwidth, x + xwidth], [bottom_quartile, bottom_quartile],
                                [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
            elif shading == '95conf':
                ax.fill_between([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none', alpha=alpha)
            if not hide_markers:
                if scatter_color is not None:  # len is a check to rgb tuples
                    ax.scatter(xvals, y_data, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                               linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                               vmax=scatter_norm_minmax[1], rasterized=rasterized)
                else:
                    ax.plot(xvals, y_data, 'o', markerfacecolor=color, markeredgecolor=edgecolor, markersize=markersize,
                            rasterized=rasterized)
        else:
            if shading != 'none':
                ax.vlines([mean], x - xwidth, x + xwidth, colors=[color], linewidth=linewidth)
            if shading == 'quartiles':
                ax.fill_betweenx([x - xwidth, x + xwidth], [bottom_quartile, bottom_quartile],
                                 [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
            elif shading == '95conf':
                ax.fill_betweenx([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                 [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none', alpha=alpha)
            if not hide_markers:
                if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                    ax.scatter(y_data, xvals, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                               linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                               vmax=scatter_norm_minmax[1], rasterized=rasterized)
                else:
                    ax.plot(y_data, xvals, 'o', markerfacecolor=color, markeredgecolor=edgecolor, markersize=markersize,
                            rasterized=rasterized)

    else:
        for i in range(len(x)):
            if use == 'median':
                mean = np.median(y_data[i])
            elif use == 'mean':
                mean = np.mean(y_data[i])
            y_data[i].sort()
            n = len(y_data[i])
            bottom_quartile = y_data[i][int(.25 * n)]
            top_quartile = y_data[i][int(.75 * n)]

            if not optimize_scatter_distance:
                xvals = [x[i] + np.random.random() * xwidth * 2 - xwidth for yi in range(len(y_data))]
            else:
                xvals = get_optimized_scatter_distance(y_data, xwidth, resolution=optimize_scatter_distance_resolution,
                                                       y_scale_factor=optimize_scatter_distance_y_scale)
                xvals += x[i]

            if shading == '95conf':
                import flystat.resampling
                conf_interval = flystat.resampling.bootstrap_confidence_intervals_from_data(y_data[i], use=use)

            if not flipxy:
                if shading != 'none':
                    ax.hlines([mean], x[i] - xwidth, x[i] + xwidth, colors=[color], linewidth=linewidth)
                if shading == 'quartiles':
                    ax.fill_between([x[i] - xwidth, x[i] + xwidth], [bottom_quartile, bottom_quartile],
                                    [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
                elif shading == '95conf':
                    ax.fill_between([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                    [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none',
                                    alpha=alpha)
                if not hide_markers:
                    if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                        ax.scatter(xvals, y_data, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                                   linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                                   vmax=scatter_norm_minmax[1], rasterized=rasterized)
                    else:
                        ax.plot(xvals, y_data, 'o', markerfacecolor=color, markeredgecolor=edgecolor,
                                markersize=markersize, rasterized=rasterized)
            else:
                if shading != 'none':
                    ax.vlines([mean], x[i] - xwidth, x[i] + xwidth, colors=[color], linewidth=linewidth)
                if shading == 'quartiles':
                    ax.fill_betweenx([x[i] - xwidth, x[i] + xwidth], [bottom_quartile, bottom_quartile],
                                     [top_quartile, top_quartile], facecolor=color, edgecolor='none', alpha=alpha)
                elif shading == '95conf':
                    ax.fill_betweenx([x - xwidth, x + xwidth], [conf_interval[0], conf_interval[0]],
                                     [conf_interval[1], conf_interval[1]], facecolor=color, edgecolor='none',
                                     alpha=alpha)
                if not hide_markers:
                    if hasattr(color, '__iter__') and len(color) > 3:  # len is a check to rgb tuples
                        ax.scatter(y_data, xvals, s=markersize, c=scatter_color, marker='o', cmap=scatter_cmap,
                                   linewidths=marker_linewidth, edgecolors=edgecolor, vmin=scatter_norm_minmax[0],
                                   vmax=scatter_norm_minmax[1], rasterized=rasterized)
                    else:
                        ax.plot(y_data, xvals, 'o', markerfacecolor=color, markeredgecolor=edgecolor,
                                markersize=markersize, rasterized=rasterized)


###################################################################################################
# 2D "heatmap" Histogram
###################################################################################################

def histogram2d(ax, x, y, bins=100, normed=False, histrange=None, weights=None, logcolorscale=False, colormap='jet',
                interpolation='nearest', colornorm=None, xextent=None, yextent=None, norm_rows=False,
                norm_columns=False, return_img=False):
    # the following paramters get passed straight to numpy.histogram2d
    # x, y, bins, normed, histrange, weights

    # from numpy.histogram2d:
    '''

    weights - if weights is not None, this function will plot a histogram of the weight values normalized by an unweighted histogram.

    Parameters
    ----------
    x : array_like, shape(N,)
      A sequence of values to be histogrammed along the first dimension.
    y : array_like, shape(M,)
      A sequence of values to be histogrammed along the second dimension.
    bins : int or [int, int] or array-like or [array, array], optional
      The bin specification:

        * the number of bins for the two dimensions (nx=ny=bins),
        * the number of bins in each dimension (nx, ny = bins),
        * the bin edges for the two dimensions (x_edges=y_edges=bins),
        * the bin edges in each dimension (x_edges, y_edges = bins).

    range : array_like, shape(2,2), optional
      The leftmost and rightmost edges of the bins along each dimension
      (if not specified explicitly in the `bins` parameters):
      [[xmin, xmax], [ymin, ymax]]. All values outside of this range will be
      considered outliers and not tallied in the histogram.
    normed : boolean, optional
      If False, returns the number of samples in each bin. If True, returns
      the bin density, ie, the bin count divided by the bin area.
    weights : array-like, shape(N,), optional
      An array of values `w_i` weighing each sample `(x_i, y_i)`. Weights are
      normalized to 1 if normed is True. If normed is False, the values of the
      returned histogram are equal to the sum of the weights belonging to the
      samples falling into each bin.
    '''
    hist, x_binned, y_binned = np.histogram2d(x, y, bins, normed=normed, range=histrange)

    if weights is not None:
        hist_weights, x, y = np.histogram2d(x, y, bins, normed=normed, range=histrange, weights=weights)
        indices_0 = np.where(np.array(hist) == 0)
        hist = hist_weights / hist
        hist[indices_0] = 0

    x = x_binned
    y = y_binned

    if logcolorscale:
        hist = np.log(hist + 1)  # the plus one solves bin=0 issues

    if xextent is None:
        xextent = [x[0], x[-1]]
    if yextent is None:
        yextent = [y[0], y[-1]]

    img = hist.T

    if norm_rows:
        for r in range(img.shape[0]):
            try:
                mi = np.min(img[r, :])
            except:
                mi = 0
            img[r, :] -= mi
            try:
                ma = np.max(img[r, :])
            except:
                ma = 0
                totalrow = 1
            if ma != 0:
                img[r, :] /= float(ma)

            # total = np.sum(img[r,:])
            # if total > 0:
            #    img[r,:] /= float(total)
            ###print mi, ma, np.min(img[r,:]), np.max(img[r,:])

    if norm_columns:
        for c in range(img.shape[1]):
            mi = np.min(img[:, c])
            img[:, c] -= mi
            ma = np.max(img[:, c])
            if ma != 0:
                img[:, c] /= ma
            ##print mi, ma, np.min(img[:,c]), np.max(img[:,c])

    if colornorm is not None:
        colornorm = matplotlib.colors.Normalize(colornorm[0], colornorm[1])
    else:
        colornorm = matplotlib.colors.Normalize(np.min(np.min(img)), np.max(np.max(img)))
        ##print 'color norm: ', np.min(np.min(img)), np.max(np.max(img))

    # make the heatmap
    cmap = plt.get_cmap(colormap)
    ax.imshow(img,
              cmap=cmap,
              extent=(xextent[0], xextent[1], yextent[0], yextent[1]),
              origin='lower',
              interpolation=interpolation,
              norm=colornorm)
    ax.set_aspect('auto')

    if return_img:
        return img


###################################################################################################
# Colorbar
###################################################################################################

def colorbar(ax=None, ticks=None, ticklabels=None, colormap='jet', aspect='auto', orientation='vertical', filename=None,
             flipspine=False, show_spine=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    if ticks is None:
        ticks = np.linspace(-1, 1, 5, endpoint=True)

    if aspect != 'auto':
        ax.set_aspect('equal')
    else:
        ax.set_aspect('auto')

    # horizontal
    if orientation == 'horizontal':
        xlim = (ticks[0], ticks[-1])
        if aspect != 'auto':
            yrange = (ticks[-1] - ticks[0]) / float(aspect)
        else:
            yrange = 1
        ylim = (0, yrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad, grad))
        if show_spine:
            if not flipspine:
                adjust_spines(ax, ['bottom'], xticks=ticks)
            else:
                adjust_spines(ax, ['top'], xticks=ticks)
        if ticklabels is not None:
            ax.set_xticklabels(ticklabels)

    # vertical
    if orientation == 'vertical':
        ylim = (ticks[0], ticks[-1])
        if aspect != 'auto':
            xrange = (ticks[-1] - ticks[0]) / float(aspect)
        else:
            xrange = 1
        xlim = (0, xrange)
        grad = np.linspace(ticks[0], ticks[-1], 500, endpoint=True)
        im = np.vstack((grad, grad)).T
        if show_spine:
            if not flipspine:
                adjust_spines(ax, ['right'], yticks=ticks)
            else:
                adjust_spines(ax, ['left'], yticks=ticks)
        if ticklabels is not None:
            ax.set_yticklabels(ticklabels)

    if not show_spine:
        adjust_spines(ax, [])
    # make image
    cmap = plt.get_cmap(colormap)
    ax.imshow(im,
              cmap=cmap,
              extent=(xlim[0], xlim[-1], ylim[0], ylim[-1]),
              origin='lower',
              interpolation='bicubic')

    ax.set_xlim(xlim[0], xlim[-1])
    ax.set_ylim(ylim[0], ylim[-1])

    if aspect != 'auto':
        ax.set_aspect('equal')
    else:
        ax.set_aspect('auto')

    if filename is not None:
        fig.savefig(filename, format='pdf')


###################################################################################################
# Scatter Plot (with PatchCollections of circles) : more control than plotting with 'dotted' style with "plot"
###################################################################################################

def get_circles_for_scatter(x, y, color='black', edgecolor='none', colormap='jet', radius=0.01, colornorm=None, alpha=1,
                            radiusnorm=None, maxradius=1, minradius=0):
    cmap = plt.get_cmap(colormap)
    if colornorm is not None:
        colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)

    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    circles = [None for i in range(len(x))]
    for i, pt in enumerate(points):
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i]) * (maxradius - minradius) + minradius
        else:
            r = radius
        circles[i] = patches.Circle(pt, radius=r)

    # make a collection of those circles
    cc = PatchCollection(circles, cmap=cmap, norm=colornorm)  # potentially useful option: match_original=True

    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)

    return cc


def get_ellipses_for_scatter(ax, x, y, color='black', edgecolor='none', colormap='jet', cmap=None, radius=0.01,
                             colornorm=None, alpha=1, radiusnorm=None, maxradius=1, minradius=0):
    # get ellipse size to make it a circle given axes
    x0, y0 = ax.transAxes.transform((ax.get_ylim()[0], ax.get_xlim()[0]))
    x1, y1 = ax.transAxes.transform((ax.get_ylim()[1], ax.get_xlim()[1]))
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)

    if cmap is None:
        cmap = plt.get_cmap(colormap)
        if colornorm is not None:
            colornorm = plt.Normalize(colornorm[0], colornorm[1], clip=True)

    # setup normalizing for radius scale factor (if used)
    if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
        if radiusnorm is None:
            radiusnorm = matplotlib.colors.Normalize(np.min(radius), np.max(radius), clip=True)
        else:
            radiusnorm = matplotlib.colors.Normalize(radiusnorm[0], radiusnorm[1], clip=True)

    # make circles
    points = np.array([x, y]).T
    ellipses = [None for i in range(len(x))]
    for i, pt in enumerate(points):
        if type(radius) is list or type(radius) is np.array or type(radius) is np.ndarray:
            r = radiusnorm(radius[i]) * (maxradius - minradius) + minradius
        else:
            r = radius
        width = r * 2 * maxd / dx
        height = r * 2 * maxd / dy
        ellipses[i] = patches.Ellipse(pt, width, height)

    # make a collection of those circles
    cc = PatchCollection(ellipses, cmap=cmap, norm=colornorm)  # potentially useful option: match_original=True

    # set properties for collection
    cc.set_edgecolors(edgecolor)
    if type(color) is list or type(color) is np.array or type(color) is np.ndarray:
        cc.set_array(color)
    else:
        cc.set_facecolors(color)
    cc.set_alpha(alpha)

    return cc


def scatter(ax, x, y, color='black', colormap='jet', cmap=None, edgecolor='none', radius=0.01, colornorm=None, alpha=1,
            radiusnorm=None, maxradius=1, minradius=0, xlim=None, ylim=None, use_ellipses=True, zorder=0):
    '''
    Make a colored scatter plot

    NOTE: the scatter points will only be circles if you use_ellipses=True, and if you do not change xlim/ylim or the relative size of the axes after the function has been called.

    x           -- np.array
    y           -- np.array
    color       -- matplotlib color (eg. string), or np.array of values
    colormap    -- matplotlib coloramp name (eg. 'jet')
    edgecolor   -- matplotlib color for edges (eg. string) - default is 'none', which means no edge
    radius      -- radius of circles to plot - in units of the axes - either a float, or np.array of floats
    colornorm   -- min and max you would like colors in the color array normalized to, eg [0,1], default is to scale to min/max of color array
    alpha       -- transparancy, float btwn 0 and 1
    radiusnorm  -- min/max you would like radius array to be normalized to
    maxradius   -- max radius size you would like
    minradius   -- min radius size you would like
    xlim/ylim   -- x and y limits of axes, default is scaled to min/max of x and y
    use_ellipses-- adjust scatter point so that they are circles, even if aspect is not equal. Only works if you do not change xlim/ylim or axes shape after calling this function

    '''
    # color can be array-like, or a matplotlib color
    # I can't figure out how to control alpha through the individual circle patches.. it seems to get overwritten by the collection. low priority!
    if type(x) is not list:
        x = x.flatten()
    if type(y) is not list:
        y = y.flatten()

    if xlim is None:
        xlim = [np.min(x), np.max(x)]
    if ylim is None:
        ylim = [np.min(y), np.max(y)]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if use_ellipses:
        cc = get_ellipses_for_scatter(ax, x, y, color=color, edgecolor=edgecolor, colormap=colormap, cmap=cmap,
                                      radius=radius, colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm,
                                      maxradius=maxradius, minradius=minradius)

    else:
        cc = get_circles_for_scatter(x, y, color=color, edgecolor=edgecolor, colormap=colormap, radius=radius,
                                     colornorm=colornorm, alpha=alpha, radiusnorm=radiusnorm, maxradius=maxradius,
                                     minradius=minradius)

    # add collection to axis
    ax.add_collection(cc)


def scattered_histogram(ax, bin_leftedges, data_list, bin_width=0.6, s=1, color='green', linewidths=None, alpha=1,
                        draw_median=False, median_color=None, fill_quartiles=True, quartile_alpha=0.5,
                        median_linewidth=2, draw_continuous_median=True, flip_xy=False, show_scatter=True,
                        lower_quartile=0.25, upper_quartile=0.75, medianmarkersize=5):
    '''
    data_list - should be a list of lists, equal in length to bin_leftedges. Each index [i] of data_list corresponds to bin_leftedges[i], and contains a list of the data that belongs in that bin.
    '''
    if median_color is None:
        median_color = color

    u = uniform_distribution(0, bin_width)

    scattered_points_x = []
    scattered_points_y = []

    for i, data in enumerate(data_list):
        for value in data:
            x = bin_leftedges[i] + u.rvs()
            y = value

            if not flip_xy:
                scattered_points_x.append(x)
                scattered_points_y.append(y)
            else:
                scattered_points_y.append(x)
                scattered_points_x.append(y)

    if show_scatter:
        ax.scatter(scattered_points_x, scattered_points_y, s=s, facecolor=color, edgecolor='none',
                   linewidths=linewidths, alpha=alpha, zorder=-100)

    if draw_median:
        bin_centers = bin_leftedges + bin_width / 2.

        for i, data in enumerate(data_list):
            if len(data) > 0:
                data.sort()
                median = np.median(data)
                ax.plot([bin_centers[i] - bin_width / 2., bin_centers[i] + bin_width / 2.], [median, median],
                        color=median_color, linewidth=median_linewidth)

                if fill_quartiles:
                    lower_quartile_index = int(len(data) * lower_quartile)
                    lower_quartile = data[lower_quartile_index]
                    upper_quartile_index = int(len(data) * upper_quartile)
                    upper_quartile = data[upper_quartile_index]
                    ax.fill_between([bin_centers[i] - bin_width / 2., bin_centers[i] + bin_width / 2.],
                                    [lower_quartile, lower_quartile], [upper_quartile, upper_quartile],
                                    facecolor=median_color, alpha=quartile_alpha, edgecolor='none')

    if draw_continuous_median:
        bin_centers = bin_leftedges + bin_width / 2.

        medians = []
        lower_quartiles = []
        upper_quartiles = []

        for i, data in enumerate(data_list):
            if len(data) > 0:
                data.sort()
                medians.append(np.median(data))

                lower_quartile_index = int(len(data) / 4.)
                lower_quartiles.append(data[lower_quartile_index])
                upper_quartile_index = int(len(data) * 3 / 4.)
                upper_quartiles.append(data[upper_quartile_index])
            else:
                medians.append(np.nan)
                lower_quartiles.append(np.nan)
                upper_quartiles.append(np.nan)

        if not flip_xy:
            ax.plot(bin_centers, medians, color=median_color, linewidth=median_linewidth)
        else:
            ax.plot(medians, bin_centers, color=median_color, linewidth=median_linewidth)
            ax.plot(medians, bin_centers, '.', markersize=medianmarkersize, markerfacecolor=median_color,
                    markeredgecolor=median_color)

        if fill_quartiles:
            if not flip_xy:
                ax.fill_between(bin_centers, lower_quartiles, upper_quartiles, facecolor=median_color,
                                alpha=quartile_alpha, edgecolor='none')
            else:
                ax.fill_betweenx(bin_centers, lower_quartiles, upper_quartiles, facecolor=median_color,
                                 alpha=quartile_alpha, edgecolor='none')


def plot_confidence_interval(ax, x, y, confidence_interval_95, confidence_interval_50=None, width=0.3, color='blue',
                             linewidth=3, alpha95=0.3, alpha50=0.5):
    xpts = [x - width / 2., x + width / 2.]
    ax.fill_between(xpts, confidence_interval_95[0], confidence_interval_95[-1], edgecolor='none', facecolor=color,
                    alpha=alpha95)
    if confidence_interval_50 is not None:
        ax.fill_between(xpts, confidence_interval_50[0], confidence_interval_50[-1], edgecolor='none', facecolor=color,
                        alpha=alpha50)
    # ax.fill_between(xpts, y-linewidth/2., y+linewidth/2., edgecolor='none', facecolor=color, alpha=1)
    ax.hlines([y], [x - width / 2.], [x + width / 2.], color=color, alpha=1, linewidth=linewidth)
