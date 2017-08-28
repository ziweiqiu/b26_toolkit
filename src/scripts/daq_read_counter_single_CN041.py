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

import time
import numpy as np
from collections import deque

from b26_toolkit.src.instruments import NI6259
from b26_toolkit.src.plotting.plots_1d import plot_counts
from PyLabControl.src.core import Parameter, Script
from b26_toolkit.src.instruments import CN041PulseBlaster

class Daq_Read_Counter_Single(Script):
    """
This script reads the Counter input from the DAQ and plots it (fixed number of samples).
    """
    _DEFAULT_SETTINGS = [
        Parameter('integration_time', 0.25, float, 'total time to collect counts for'),
        Parameter('N_samps', 100, int, 'Number of samples for estimating error bar'),
        Parameter('counter_channel', 'ctr0', ['ctr0', 'ctr1'], 'Daq channel used for counter')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster}

    _SCRIPTS = {

    }

    def __init__(self, instruments, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        """
        Example of a script that emits a QT signal for the gui
        Args:
            name (optional): name of script, if empty same as class name
            settings (optional): settings for this script, if empty same as default settings
        """
        Script.__init__(self, name, settings=settings, scripts=scripts, instruments=instruments,
                        log_function=log_function, data_path=data_path)

        self.data = {}

    def _function(self):
        """
        This is the actual function that will be executed. It uses only information that is provided in the settings property
        will be overwritten in the __init__
        """

        # turn laser on
        self.instruments['PB']['instance'].update({'laser': {'status': True}})

        sample_num = self.settings['N_samps']

        # initialize numpy arrays to store data:
        self.data = {'counts': np.zeros(sample_num-1), 'time': np.zeros(sample_num-1)}

        # normalization to get kcounts/sec:
        normalization = self.settings['integration_time']/ sample_num / .001

        # set sample rate:
        sample_rate = sample_num / self.settings['integration_time']
        self.instruments['daq']['instance'].settings['digital_input'][self.settings['counter_channel']]['sample_rate'] = sample_rate

        # start counter acquisiotion:
        task = self.instruments['daq']['instance'].setup_counter("ctr0", sample_num, continuous_acquisition=False)
        self.instruments['daq']['instance'].run(task)
        raw_data, num_read = self.instruments['daq']['instance'].read(task)

        # parse cumulative counter into a numpy array of counts collected in subsequent windows:
        self.data['counts'] = np.array([raw_data[i+1]-raw_data[i] for i in range(len(raw_data)-1)])/normalization

        # create time vector:
        self.data['time'] = np.arange(0,len(raw_data)-1)/sample_rate

        time.sleep(2.0 / sample_rate)

        # clean up APD tasks
        self.instruments['daq']['instance'].stop(task)

        # turn laser off
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.log('Laser is off.')

    def plot(self, figure_list):
        # COMMENT_ME
        super(Daq_Read_Counter_Single, self).plot([figure_list[1]])

    def _plot(self, axes_list, data = None):
        # COMMENT_ME

        if data is None:
            data = self.data

        if data:
            plot_counts(axes_list[0], data['counts'], data['time'],'time [sec]')



if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'Daq_Read_Cntr': 'Daq_Read_Cntr'}, script, instr)

    print(script)
    print(failed)
    print(instr)