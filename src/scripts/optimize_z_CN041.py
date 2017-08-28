"""
    This file is part of b26_toolkit, a PyLabControl add-on for experiments in Harvard LISE B26.
    Copyright (C) <2016>  Arthur Safira, Jan Gieseler, Aaron Kabcenell

    Foobar is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from copy import deepcopy

from PyLabControl.src.core import Script, Parameter
from b26_toolkit.src.plotting.plots_1d import plot_counts
from b26_toolkit.src.scripts.confocal_scan_CN041 import ConfocalScan
from b26_toolkit.src.scripts.set_confocal_CN041 import SetConfocal
from b26_toolkit.src.instruments import CN041PulseBlaster

class optimizeZ(Script):

    _DEFAULT_SETTINGS = [
        Parameter('scan_range', 3, float, 'z-range for optimizing scan [V]'),
        Parameter('num_points', 50, int, 'number of z points to scan'),
        Parameter('time_per_pt', .5, [.25, .5, 1.], 'time in s to measure at each point for 1D z-scans only')]

    _INSTRUMENTS = {'PB': CN041PulseBlaster}
    # _INSTRUMENTS = {}

    _SCRIPTS = {'scan_z': ConfocalScan, 'set_focus': SetConfocal}

    def __init__(self, scripts, name = None, settings = None, instruments=None, log_function = None, timeout = 1000000000, data_path = None):

        Script.__init__(self, name, scripts = scripts, settings=settings, instruments = instruments, log_function=log_function, data_path = data_path)

        self.scripts['scan_z'].update({'scan_axes': 'z'})
        self.scripts['scan_z'].update({'RoI_mode': 'center'})

    def _function(self):

        # turn laser on
        self.instruments['PB']['instance'].update({'laser': {'status': True}})

        initial_point = self.scripts['scan_z'].instruments['NI6259']['instance'].get_analog_voltages([self.scripts['scan_z'].settings['DAQ_channels']['x_ao_channel'],self.scripts['scan_z'].settings['DAQ_channels']['y_ao_channel'],self.scripts['scan_z'].settings['DAQ_channels']['z_ao_channel']])

        self.data = {'maximum_point': None,
                     'initial_point': initial_point[2],
                     'fluor_vector': [],
                     'extent': [],
                     'max_fluor': None
                     }

        self.scripts['scan_z'].settings['point_a'].update({'z': initial_point[2]})
        self.scripts['scan_z'].settings['point_b'].update({'z': self.settings['scan_range']})
        self.scripts['scan_z'].settings['num_points'].update({'z': self.settings['num_points']})
        self.scripts['scan_z'].settings['time_per_pt'].update({'z-piezo': self.settings['time_per_pt']})
        self.scripts['scan_z'].run()

        self.data['fluor_vector'] = deepcopy(self.scripts['scan_z'].data['image_data'])
        self.data['extent'] = deepcopy(self.scripts['scan_z'].data['bounds'])

        self.data['max_fluor'] = np.amax(self.data['fluor_vector'])
        self.data['maximum_point'] = self.data['extent'][0]+(self.data['extent'][1]-self.data['extent'][0])/(len(self.data['fluor_vector'])-1)*float(np.argmax(self.data['fluor_vector']))

        self.scripts['set_focus'].settings['point'].update({'x': initial_point[0], 'y': initial_point[1], 'z': self.data['maximum_point']})
        self.scripts['set_focus'].run()

        self.log('set z = {:f}'.format(self.data['maximum_point']))
        # turn laser off
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.log('Laser is off.')

    @staticmethod
    def plot_data(axes_list, data):
        plot_counts(axes_list[0], data['fluor_vector'], np.linspace(data['extent'][0], data['extent'][1], len(data['fluor_vector'])), 'z [V]')
        if data['maximum_point'] and data['max_fluor']:
            axes_list[0].hold(True)
            axes_list[0].plot(data['maximum_point'], data['max_fluor'], 'ro')
            axes_list[0].hold(False)

    def _plot(self, axes_list, data=None):
        """
        Plots the confocal scan image
        Args:
            axes_list: list of axes objects on which to plot the galvo scan on the first axes object
            data: data (dictionary that contains keys image_data, extent) if not provided use self.data
        """
        if data is None:
            data = self.data

        if self._current_subscript_stage['current_subscript'] == self.scripts['scan_z']:
            self.scripts['scan_z']._plot(axes_list)
        else:
            self.plot_data(axes_list, data)

        # plot_counts(axes_list[0], data['fluor_vector'],np.linspace(data['extent'][0],data['extent'][1],len(data['fluor_vector'])),'z [V]')
        # axes_list[0].hold(True)
        # axes_list[0].plot(data['maximum_point'], data['max_fluor'], 'ro')
        # axes_list[0].hold(False)


    def _update_plot(self, axes_list):
        """
        updates the galvo scan image
        Args:
            axes_list: list of axes objects on which to plot plots the esr on the first axes object
        """

        if self._current_subscript_stage['current_subscript'] == self.scripts['scan_z']:
            self.scripts['scan_z']._update_plot(axes_list)

        if self.data['maximum_point'] and self.data['max_fluor']:
            axes_list[0].hold(True)
            axes_list[0].plot(self.data['maximum_point'], self.data['max_fluor'], 'ro')
            axes_list[0].hold(False)

    def get_axes_layout(self, figure_list):
        """
        returns the axes objects the script needs to plot its data
        the default creates a single axes object on each figure
        This can/should be overwritten in a child script if more axes objects are needed
        Args:
            figure_list: a list of figure objects
        Returns:
            axes_list: a list of axes objects

        """

        # only pick the first figure from the figure list, this avoids that get_axes_layout clears all the figures
        return super(optimizeZ, self).get_axes_layout([figure_list[1]])


if __name__ == '__main__':
    script, failed, instruments = Script.load_and_append(script_dict={'ConfocalScan': 'ConfocalScan'})

    print(script)
    print(failed)
    # print(instruments)

