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
from matplotlib import patches

from b26_toolkit.src.instruments import DAQ
from PyLabControl.src.core import Script, Parameter


class SetLaser(Script):
    """
This script points the laser to a point
    """

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('x', -0.4, float, 'x-coordinate'),
                   Parameter('y', -0.4, float, 'y-coordinate')
                   ]),
        Parameter('DAQ_channels',
            [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for x voltage analog output'),
            Parameter('y_ao_channel', 'ao3', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for y voltage analog output')
            ])
    ]

    _INSTRUMENTS = {'daq':  DAQ}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        pt = (self.settings['point']['x'], self.settings['point']['y'])
        pt = np.transpose(np.column_stack((pt[0],pt[1])))
        pt = (np.repeat(pt, 2, axis=1))

        self.instruments['daq']['instance'].AO_init([self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], pt)
        self.instruments['daq']['instance'].AO_run()
        self.instruments['daq']['instance'].AO_waitToFinish()
        self.instruments['daq']['instance'].AO_stop()

    #must be passed figure with galvo plot on first axis
    def plot(self, figure_list):
        axes_Image = figure_list[0].axes[0]

        # removes patches
        [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]

        patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), .0005, fc='r')
        axes_Image.add_patch(patch)


if __name__ == '__main__':
    from PyLabControl.src.core import Instrument

    instruments, instruments_failed = Instrument.load_and_append({'daq':  'DAQ'})

    script, failed, instruments = Script.load_and_append(script_dict={'SetLaser':'SetLaser'}, instruments = instruments)

    print(script)
    print(failed)
    # print(instruments)