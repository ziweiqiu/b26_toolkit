"""
    This file is part of b26_toolkit, a PyLabControl add-on for experiments in Harvard LISE B26.
    Copyright (C) <2016>  Arthur Safira, Jan Gieseler, Aaron Kabcenell

    b26_toolkit is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    b26_toolkit is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with b26_toolkit.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from matplotlib import patches

from b26_toolkit.src.instruments import NI6259, NI9263
from PyLabControl.src.core import Script, Parameter


class SetLaser(Script):
    """
This script points the laser to a point
    """

    _DEFAULT_SETTINGS = [
        Parameter('point',
                  [Parameter('x', 0, float, 'x-coordinate'),
                   Parameter('y', 0, float, 'y-coordinate')
                   ]),
        Parameter('patch_size', 0.05, [0.005, 0.05, 0.5], 'size of the red circle'),
        Parameter('DAQ_channels',
            [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for x voltage analog output'),
            Parameter('y_ao_channel', 'ao1', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for y voltage analog output')
            ]),
        Parameter('daq_type', 'PCI', ['PCI', 'cDAQ'], 'Type of daq to use for scan')
    ]

    _INSTRUMENTS = {'NI6259':  NI6259, 'NI9263': NI9263}

    _SCRIPTS = {}


    def __init__(self, instruments = None, scripts = None, name = None, settings = None, log_function = None, data_path = None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings = settings, instruments = instruments, scripts = scripts, log_function= log_function, data_path = data_path)
        if self.settings['daq_type'] == 'PCI':
            self.daq_out = self.instruments['NI6259']['instance']
        elif self.settings['daq_type'] == 'cDAQ':
            self.daq_out = self.instruments['NI9263']['instance']

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """
        pt = (self.settings['point']['x'], self.settings['point']['y'])

        # daq API only accepts either one point and one channel or multiple points and multiple channels
        pt = np.transpose(np.column_stack((pt[0],pt[1])))
        pt = (np.repeat(pt, 2, axis=1))

        task = self.daq_out.setup_AO([self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], pt)
        self.daq_out.run(task)
        self.daq_out.waitToFinish(task)
        self.daq_out.stop(task)
        self.log('laser set to Vx={:.4}, Vy={:.4}'.format(self.settings['point']['x'], self.settings['point']['y']))

    #must be passed figure with galvo plot on first axis
    def plot(self, figure_list):
        axes_Image = figure_list[0].axes[0]

        # removes patches
        [child.remove() for child in axes_Image.get_children() if isinstance(child, patches.Circle)]

        patch = patches.Circle((self.settings['point']['x'], self.settings['point']['y']), self.settings['patch_size'], fc='y')
        axes_Image.add_patch(patch)

# class SetLaser_cDAQ(SetLaser):
#     """
# This script points the laser to a point
#     """
#
#     _DEFAULT_SETTINGS = [
#         Parameter('point',
#                   [Parameter('x', -0.4, float, 'x-coordinate'),
#                    Parameter('y', -0.4, float, 'y-coordinate'),
#                    ]),
#         Parameter('DAQ_channels',
#             [Parameter('x_ao_channel', 'ao0', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for x voltage analog output'),
#             Parameter('y_ao_channel', 'ao3', ['ao0', 'ao1', 'ao2', 'ao3'], 'Daq channel used for y voltage analog output')
#             ])
#     ]
#
#     _INSTRUMENTS = {'daq_out':  NI9263}
#
#     def _function(self):
#         """
#         This is the actual function that will be executed. It uses only information that is provided in the settings property
#         will be overwritten in the __init__
#         """
#         pt = (self.settings['point']['x'], self.settings['point']['y'])
#
#         # daq API only accepts either one point and one channel or multiple points and multiple channels
#         pt = np.transpose(np.column_stack((pt[0],pt[1])))
#         pt = (np.repeat(pt, 2, axis=1))
#
#         task = self.instruments['daq_out']['instance'].setup_AO([self.settings['DAQ_channels']['x_ao_channel'], self.settings['DAQ_channels']['y_ao_channel']], pt)
#         self.instruments['daq_out']['instance'].run(task)
#         self.instruments['daq_out']['instance'].waitToFinish(task)
#         self.instruments['daq_out']['instance'].stop(task)
#         self.log('laser set to Vx={:.4}, Vy={:.4}'.format(self.settings['point']['x'], self.settings['point']['y']))
#
# if __name__ == '__main__':
#     from PyLabControl.src.core import Instrument
#
#     # instruments, instruments_failed = Instrument.load_and_append({'daq':  'NI6259'})
#
#     script, failed, instruments = Script.load_and_append(script_dict={'SetLaser_cDAQ': 'SetLaser_cDAQ'})
#
#     print(script)
#     print(failed)
#     # print(instruments)