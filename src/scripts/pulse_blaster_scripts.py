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
from b26_toolkit.src.scripts.pulse_blaster_base_script import PulseBlasterBaseScript
from b26_toolkit.src.instruments import DAQ, B26PulseBlaster, MicrowaveGenerator, Pulse
from b26_toolkit.src.plotting.plots_1d import plot_esr, plot_pulses, update_pulse_plot, plot_1d_simple_timetrace_ns, update_1d_simple
from PyLabControl.src.core import Parameter, Script
from src.data_processing.fit_functions import fit_rabi_decay, cose_with_decay

class PulsedESR(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power and durations for varying frequencies
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_power', -45.0, float, 'microwave power in dB'),
        Parameter('tau_mw', 200, float, 'the time duration of the microwaves (in ns)'),
        Parameter('meas_time', 300, float, 'measurement time after rabi sequence (in ns)'),
        Parameter('num_averages', 1000000, int, 'number of averages'),
        Parameter('reset_time', 1000000, int, 'time with laser on at the beginning to reset state'),
        Parameter('freq_start', 2.82e9, float, 'start frequency of scan in Hz'),
        Parameter('freq_stop', 2.92e9, float, 'end frequency of scan in Hz'),
        Parameter('freq_points', 100, int, 'number of frequencies in scan in Hz'),
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        assert self.settings['freq_start'] < self.settings['freq_stop']

        self.data = {'mw_frequencies': np.linspace(self.settings['freq_start'], self.settings['freq_stop'],
                                                   self.settings['freq_points']), 'esr_counts': []}

        for i, mw_frequency in enumerate(self.data['mw_frequencies']):
            self._loop_count = i
            self.instruments['mw_gen']['instance'].update({'frequency': float(mw_frequency)})
            super(PulsedESR, self)._function(self.data)
            self.data['esr_counts'].append(self.data['counts'])

    def _calc_progress(self):
        #COMMENT_ME
        # todo: change to _calc_progress(self, index):
        progress = int(100. * (self._loop_count) / self.settings['freq_points'])
        return progress

    def _plot(self, axes_list, data = None):
        '''
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first 2)
            data (optional) dataset to plot, if not provided use self.data
        '''
        if data is None:
            data = self.data

        mw_frequencies = data['mw_frequencies']
        esr_counts = data['esr_counts']
        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
        axis2 = axes_list[1]
        plot_pulses(axis2, self.pulse_sequences[0])

    def _update_plot(self, axes_list):
        mw_frequencies = self.data['mw_frequencies']
        esr_counts = self.data['esr_counts']
        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
            # axis2 = axes_list[1]
            # update_pulse_plot(axis2, self.pulse_sequences[0])

    def _create_pulse_sequences(self):

        '''

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''

        reset_time = self.settings['reset_time']
        tau = self.settings['tau_mw']
        pulse_sequences = [[Pulse('laser', 0, reset_time),
                            Pulse('microwave_i', reset_time, tau),
                            Pulse('laser', reset_time + tau, self.settings['meas_time']),
                            Pulse('apd_readout', reset_time + tau, self.settings['meas_time'])
                            ]]

        tau_list = [tau]
        end_time_max = 0
        for pulse_sequence in pulse_sequences:
            for pulse in pulse_sequence:
                end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        for pulse_sequence in pulse_sequences:
            pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']


class Rabi(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power for varying durations to measure Rabi Oscillations
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_power', -45.0, float, 'microwave power in dB'),
        Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
        Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of rabi pulse duration (in ns)'),
        Parameter('time', 200, float, 'total time of rabi oscillations (in ns)'),
        Parameter('meas_time', 300, float, 'measurement time after rabi sequence (in ns)'),
        Parameter('delay_init_mw', 100, int, 'delay between initialization and mw (in ns)'),
        Parameter('delay_mw_readout', 100, int, 'delay between mw and readout (in ns)'),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('reset_time', 3000, int, 'time with laser on at the beginning to reset state'),
        Parameter('skip_invalid_sequences', False, bool, 'Skips any sequences with <15ns commands'),
        Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)'),
        Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits'] = None
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        super(Rabi, self)._function(self.data)

        counts = self.data['counts'][:, 1] / self.data['counts'][:, 0]
        tau = self.data['tau']


        try:
            fits = fit_rabi_decay(tau, counts, varibale_phase=True)
            self.data['fits'] = fits
        except:
            self.data['fits'] = None
            self.log('rabi fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        tau_list = range(int(max(15, self.settings['time_step'])), int(self.settings['time'] + 15),
                         self.settings['time_step'])
        reset_time = self.settings['reset_time']
        microwave_channel = 'microwave_' + self.settings['microwave_channel']

        for tau in tau_list:
            pulse_sequences.append([Pulse('laser', 0, reset_time - self.settings['ref_meas_off_time'] - 15 - self.settings['meas_time']),
                                    Pulse('apd_readout', reset_time - 15 - self.settings['meas_time'],self.settings['meas_time']),
                                    Pulse('laser', reset_time - 15 - self.settings['meas_time'],self.settings['meas_time']),
                                    Pulse(microwave_channel, reset_time + self.settings['delay_init_mw'], tau),
                                    Pulse('laser', reset_time + self.settings['delay_init_mw'] + tau + self.settings[
                                        'delay_mw_readout'], self.settings['meas_time']),
                                    Pulse('apd_readout',reset_time + self.settings['delay_init_mw'] + tau + self.settings['delay_mw_readout'], self.settings['meas_time'])
                                    ])


        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']



    def _plot(self, axislist, data = None):
        '''
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first 2)
            data (optional) dataset to plot (dictionary that contains keys counts, tau, fits), if not provided use self.data
        '''

        if data is None:
            data = self.data

        if data['fits'] is not None:
            counts = data['counts'][:,1]/ data['counts'][:,0]
            tau = data['tau']
            fits = data['fits']

            axislist[0].plot(tau, counts, 'b')
            axislist[0].hold(True)

            axislist[0].plot(tau, cose_with_decay(tau, *fits), 'k', lw=3)
            pi_time = 2*np.pi / fits[1] / 2
            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz, pi-time: {:2.0f}ns'.format(self.settings['mw_power'], self.settings['mw_frequency']*1e-9, pi_time))
        else:
            super(Rabi, self)._plot(axislist)
            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz'.format(self.settings['mw_power'], self.settings['mw_frequency']*1e-9))
            axislist[0].legend(labels=('Ref Fluorescence', 'Rabi Data'), fontsize=8)




class Rabi_Power_Sweep_Single_Tau(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power for varying durations to measure Rabi Oscillations
    """
    _DEFAULT_SETTINGS = [
        Parameter('min_mw_power', -45.0, float, 'minimum microwave power in dB'),
        Parameter('max_mw_power', -45.0, float, 'maximum microwave power in dB'),
        Parameter('mw_power_step', 1.0, float, 'power to step by in dB'),
        Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
        Parameter('mw_time', 200, float, 'total time of rabi oscillations (in ns)'),
        Parameter('meas_time', 300, float, 'measurement time after rabi sequence (in ns)'),
        Parameter('num_averages', 1000000, int, 'number of averages'),
        Parameter('reset_time', 10000, int, 'time with laser on at the beginning to reset state'),
        Parameter('skip_invalid_sequences', False, bool, 'Skips any sequences with <15ns commands')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        mw_power_values = np.arange(self.settings['min_mw_power'],
                                    self.settings['max_mw_power'] + self.settings['mw_power_step'],
                                    self.settings['mw_power_step'])

        print(mw_power_values)
        self.data = {'mw_power_values': mw_power_values, 'counts_for_mw': np.zeros(len(mw_power_values))}
        for index, power in enumerate(mw_power_values):
            self.instruments['mw_gen']['instance'].update({'amplitude': float(power)})
            super(Rabi_Power_Sweep_Single_Tau, self)._function(self.data)
            self.data['counts_for_mw'][index] = self.data['counts'][0]

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        reset_time = self.settings['reset_time']
        mw_time = self.settings['mw_time']
        pulse_sequences.append([Pulse('laser', 0, reset_time),
                                Pulse('microwave_i', reset_time + 200, mw_time),
                                Pulse('laser', reset_time + mw_time + 300, self.settings['meas_time']),
                                Pulse('apd_readout', reset_time + mw_time + 300, self.settings['meas_time'])
                                ])

        end_time_max = 0
        for pulse_sequence in pulse_sequences:
            for pulse in pulse_sequence:
                end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        for pulse_sequence in pulse_sequences:
            pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], [mw_time], self.settings['meas_time']

    def _plot(self, axes_list, data = None):
        '''
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first 2)
            data (optional) dataset to plot (dictionary that contains keys counts_for_mw, mw_power_values), if not provided use self.data
        '''
        if data is None:
            data = self.data

        counts = data['counts_for_mw']
        x_data = data['mw_power_values']
        axis1 = axes_list[0]
        if not counts == []:
            plot_1d_simple_timetrace_ns(axis1, x_data, [counts], x_label='microwave power (dBm)')
        axis2 = axes_list[1]
        plot_pulses(axis2, self.pulse_sequences[self.sequence_index])

    def _update_plot(self, axes_list):
        '''
        Updates plots specified in _plot above
        Args:
            axes_list: list of axes to write plots to (uses first 2)

        '''
        counts = self.data['counts_for_mw']
        x_data = self.data['mw_power_values']
        axis1 = axes_list[0]
        if not counts == []:
            update_1d_simple(axis1, x_data, [counts])
        axis2 = axes_list[1]
        update_pulse_plot(axis2, self.pulse_sequences[self.sequence_index])


# class Pulsed_ESR(ExecutePulseBlasterSequence):
#     """
# This script applies a microwave pulse at fixed power for varying durations to measure Rabi Oscillations
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('mw_power', -45.0, float, 'microwave power in dB'),
#         Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
#         Parameter('delay_until_mw', 100, float, 'total time of rabi oscillations (in ns)'),
#         Parameter('mw_duration', 200, float, 'total time of rabi oscillations (in ns)'),
#         Parameter('time_step', 15, float,
#                   'time step increment of rabi pulse duration (in ns)'),
#         Parameter('time', 400, float, 'total time of rabi oscillations (in ns)'),
#         Parameter('meas_time', 15, float, 'measurement time after rabi sequence (in ns)'),
#         Parameter('num_averages', 1000000, int, 'number of averages'),
#         Parameter('reset_time', 1000000, int, 'time with laser on at the beginning to reset state')
#     ]
#
#     _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}
#
#     def _function(self):
#         self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
#         self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
#         self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
#         super(Pulsed_ESR, self)._function()
#
#     def _create_pulse_sequences(self):
#         '''
#
#         Returns: pulse_sequences, num_averages, tau_list
#             pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
#             scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
#             sequence must have the same number of daq read pulses
#             num_averages: the number of times to repeat each pulse sequence
#             tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
#             meas_time: the width (in ns) of the daq measurement
#
#         '''
#         pulse_sequences = []
#         tau_list = range(int(max(15, self.settings['time_step'])), int(self.settings['time'] + 15),
#                          int(self.settings['time_step']))
#         reset_time = self.settings['reset_time']
#         for tau in tau_list:
#             pulse_sequences.append([Pulse('laser', 0, reset_time + max(tau + self.settings['meas_time'],
#                                                                        self.settings['delay_until_mw'] + self.settings[
#                                                                            'mw_duration'])),
#                                     Pulse('microwave_i', reset_time + self.settings['delay_until_mw'],
#                                           self.settings['mw_duration']),
#                                     Pulse('apd_readout', reset_time + tau, self.settings['meas_time'])
#                                     ])
#         end_time_max = 0
#         for pulse_sequence in pulse_sequences:
#             for pulse in pulse_sequence:
#                 end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
#         for pulse_sequence in pulse_sequences:
#             pulse_sequence[0] = Pulse('laser', 0, end_time_max)
#
#         return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']

class Pulsed_ESR_Pulsed_Laser(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power for varying durations to measure Rabi Oscillations
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_power', -45.0, float, 'microwave power in dB'),
        Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
        Parameter('delay_until_mw', 100, float, 'total time of rabi oscillations (in ns)'),
        Parameter('mw_duration', 200, float, 'total time of rabi oscillations (in ns)'),
        Parameter('time_step', 15, float,
                  'time step increment of rabi pulse duration (in ns)'),
        Parameter('time', 400, float, 'total time of rabi oscillations (in ns)'),
        Parameter('meas_time', 15, float, 'measurement time after rabi sequence (in ns)'),
        Parameter('num_averages', 1000000, int, 'number of averages'),
        Parameter('reset_time', 1000000, int, 'time with laser on at the beginning to reset state')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        super(Pulsed_ESR_Pulsed_Laser, self)._function()

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        tau_list = range(int(max(15, self.settings['time_step'])), int(self.settings['time'] + 15),
                         int(self.settings['time_step']))
        reset_time = self.settings['reset_time']
        for tau in tau_list:
            if tau < self.settings['delay_until_mw']:
                pulse_sequences.append([Pulse('laser', 0, reset_time),
                                        Pulse('microwave_i', reset_time + self.settings['delay_until_mw'],
                                              self.settings['mw_duration']),
                                        Pulse('apd_readout', reset_time + tau, self.settings['meas_time'])
                                        ])
            else:
                pulse_sequences.append([Pulse('laser', 0, reset_time + max(tau + self.settings['meas_time'],
                                                                           self.settings['delay_until_mw'] +
                                                                           self.settings[
                                                                               'mw_duration'])),
                                        Pulse('microwave_i', reset_time + self.settings['delay_until_mw'],
                                              self.settings['mw_duration']),
                                        Pulse('apd_readout', reset_time + tau, self.settings['meas_time'])
                                        ])
        end_time_max = 0
        for pulse_sequence in pulse_sequences:
            for pulse in pulse_sequence:
                end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        for pulse_sequence in pulse_sequences:
            pulse_sequence[0] = Pulse('laser', 0, end_time_max)

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']


class CalibrateMeasurementWindow(PulseBlasterBaseScript):
    """
This script find the optimal duration of the measurment window.
It applies a sliding measurement window with respect to a readout from the NV 0 state and the NV 1 state.
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_power', -45.0, float, 'microwave power in dB'),
        Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
        Parameter('pi_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
        Parameter('readout_window_incremement', 10, [5, 10, 20, 50, 100], 'time step increment of measurement duration (in ns)'),
        Parameter('initial_readout_displacement', -80, int, 'min time of measurement duration (in ns)'),
        Parameter('final_readout_displacement', 450, int, 'max time of measurement duration (in ns)'),
        Parameter('reset_time', 3000, int, 'time with laser on at the beginning to reset state'),
        Parameter('delay_init_mw', 200, int, 'time delay before pi pulse after NV reset'),
        Parameter('delay_mw_readout', 200, int, 'time delay before readout after pi pulse'),
        Parameter('measurement_window_width', 20, int, 'the width of the sliding readout window'),
        Parameter('laser_on_time', 500, range(100, 1201, 100), 'time laser is on for readout'),
        Parameter('ref_meas_off_time', 1000, int, 'time reset laser is turned off before reference measurement is made'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('num_averages', 1000000, int, 'number of averages')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        super(CalibrateMeasurementWindow, self)._function()

    def _create_pulse_sequences(self):
        """

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        """
        pulse_sequences = []
        tau_list = range(self.settings['initial_readout_displacement'],
                         self.settings['final_readout_displacement'],
                         self.settings['readout_window_incremement'])
        reset_time = self.settings['reset_time']

        for tau in tau_list:
            pulse_sequences.append([Pulse('laser', 0, reset_time - self.settings['ref_meas_off_time'] - self.settings['laser_on_time']),
                                    Pulse('apd_readout', reset_time - self.settings['laser_on_time'] + tau, self.settings['measurement_window_width']),
                                    Pulse('laser', reset_time - self.settings['laser_on_time'], self.settings['laser_on_time']),
                                    Pulse('microwave_i', reset_time + self.settings['delay_init_mw'], self.settings['pi_pulse_time']),
                                    Pulse('laser', reset_time + self.settings['delay_init_mw'] + self.settings['pi_pulse_time'] + self.settings[
                                        'delay_mw_readout'], self.settings['laser_on_time']),
                                    Pulse('apd_readout', reset_time + self.settings['delay_init_mw'] + self.settings['pi_pulse_time'] +
                                          self.settings['delay_mw_readout'] + tau, self.settings['measurement_window_width'])
                                    ])

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['measurement_window_width']

    def _plot(self, axes_list, data = None):
        """
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first)
            data (optional): dataset to plot (dictionary that contains keys counts, tau), if not provided use self.data
        """

        super(CalibrateMeasurementWindow, self)._plot(axes_list, data)
        axes_list[0].set_title('Measurement Calibration')
        axes_list[0].legend(labels=('|0> State Fluorescence', '|1> State Fluoresence'), fontsize=8)



class XY8(PulseBlasterBaseScript):
    """
This script runs a CPMG pulse sequence.
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses',[
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            # Parameter('mw_switch_extra_time', 15, int, 'Time to add before and after microwave switch is turned on'),
            Parameter('pi_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
            Parameter('number_of_pulse_blocks', 1, range(1, 17), 'number of alternating x-y-x-y-y-x-y-x pulses'),
        ]),
        Parameter('tau_times',[
            Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000],
                      'time step increment of time between pulses (in ns)'),
            Parameter('min_time', 100, float, 'minimum time between pulses (in ns)'),
            Parameter('max_time', 1000, float, 'maximum time between pulses (in ns)'),
        ]),
        Parameter('read_out',[
            Parameter('delay_mw_init', 1000, int, 'delay between initialization and mw (in ns)'),
            Parameter('delay_mw_readout', 200, int, 'delay between mw and readout (in ns)'),
            Parameter('meas_time', 250, float, 'measurement time after CPMG sequence (in ns)'),
            Parameter('nv_reset_time', 3000, int, 'time with laser on at the beginning to reset state'),
            Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)')
        ]),
        Parameter('num_averages', 1000, int, 'number of averages (should be less than a million)'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}
    _SCRIPTS = {}

    def _function(self):
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        super(XY8, self)._function()

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        # tau_list = range(int(max(15, self.settings['min_delay_time'])), int(self.settings['max_delay_time'] + 15),
        #                  self.settings['delay_time_step'])

        # JG: changed the previous because the 15ns is taken care of later
        tau_list = range(int(self.settings['tau_times']['min_time']),
                         int(self.settings['tau_times']['max_time']),
                         self.settings['tau_times']['time_step']
                         )

        reset_time = self.settings['read_out']['nv_reset_time']
        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = pi_time/2.0

        ref_meas_off_time = self.settings['read_out']['ref_meas_off_time']
        meas_time = self.settings['read_out']['meas_time']
        delay_mw_init = self.settings['read_out']['delay_mw_init']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        number_of_pulse_blocks = self.settings['mw_pulses']['number_of_pulse_blocks']


        for tau in tau_list:

            pulse_sequence = []

            #initialize and pi/2 pulse
            pulse_sequence.extend([Pulse('laser', 0, reset_time - ref_meas_off_time - 15 - meas_time),
                                   Pulse('apd_readout', reset_time - 15 - meas_time, meas_time),
                                   Pulse('laser', reset_time - 15 - meas_time, meas_time),
                                   Pulse('microwave_i', reset_time + delay_mw_init, pi_half_time)
                                   ])

            #CPMG xyxyyxyx loops added number_of_pulse_blocks times
            section_begin_time = reset_time + delay_mw_init + pi_half_time - tau/2 #for the first pulse, only wait tau/2
            # JG 16-08-19 - begin changed to pi time instead of pi/2
            # section_begin_time = reset_time + delay_mw_init + pi_time
            # JG 16-08-19 - end

            for i in range(0, number_of_pulse_blocks):
                pulse_sequence.extend([Pulse('microwave_i', section_begin_time + 1*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 2*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 3*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 4*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 5*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 6*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 7*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 8*tau - pi_half_time, pi_time)
                                      ])
                section_begin_time += 8*tau

            #pi/2 and readout
            pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau/2, pi_half_time),
                                   Pulse('laser',       section_begin_time + tau/2 + pi_half_time + delay_mw_readout, meas_time),
                                   Pulse('apd_readout', section_begin_time + tau/2 + pi_half_time + delay_mw_readout, meas_time)])

            # JG 16-08-19 - begin changed to pi time instead of pi/2
            # pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau, pi_half_time),
            #                        Pulse('laser',       section_begin_time + tau + pi_half_time + delay_mw_readout, meas_time),
            #                        Pulse('apd_readout', section_begin_time + tau + pi_half_time + delay_mw_readout, meas_time)])
            # JG 16-08-19 - end


            pulse_sequences.append(pulse_sequence)


        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time


    def _plot(self, axislist, data = None):
        """
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first 2)
            data (optional) dataset to plot (dictionary that contains keys counts, tau), if not provided use self.data
        """

        super(XY8, self)._plot(axislist, data)
        axislist[0].set_title('XY8')
        axislist[0].legend(labels=('Ref Fluorescence', 'XY8 data'), fontsize=8)

class PDD(PulseBlasterBaseScript):
    """
This script runs a PDD ( Periodic Dynamical Decoupling) sequence for different number of pi pulses.
For a single pi-pulse this is a Hahn-echo sequence.

The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2

Tau/2 is the time between the center of the pulses!


    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -2, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('pi_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
            Parameter('number_of_pi_pulses', 1, range(1, 17), 'number of pi pulses')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 15, float, 'min value for tau, the free evolution time in between pulses (in ns)'),
            Parameter('max_time', 30, float, 'max value for tau, the free evolution time in between pulses (in ns)'),
            Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000], 'step size for tau, the free evolution time in between pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 250, float, 'measurement time after CPMG sequence (in ns)'),
            Parameter('nv_reset_time', 3000, int, 'time duration of the green laser to reset the spin state'),
            Parameter('ref_meas_off_time', 1000, int, 'laser off time before taking reference measurement at the end of init (ns)'),
            Parameter('delay_mw_init', 1000, int, 'delay between initialization and mw (in ns)'),
            Parameter('delay_mw_readout', 100, int, 'delay between mw and readout (in ns)')
        ]),
        Parameter('num_averages', 1000, int, 'number of averages (should be less than a million)'),
        Parameter('skip_invalid_sequences', False, bool, 'Skips any sequences with <15ns commands')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    _SCRIPTS = {}


    def __init__(self, instruments, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        Script.__init__(self, name, settings=settings, scripts=scripts, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        super(PDD, self)._function()


    def _create_pulse_sequences(self):
        '''
        creates the pulse sequence for the Hahn echo /
        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []


        tau_list = range(int(self.settings['tau_times']['min_time']),
                         int(self.settings['tau_times']['max_time'] + self.settings['tau_times']['time_step']),
                         self.settings['tau_times']['time_step']
                         )



        reset_time = self.settings['read_out']['nv_reset_time']
        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = pi_time/2.0

        ref_meas_off_time = self.settings['read_out']['ref_meas_off_time']
        meas_time = self.settings['read_out']['meas_time']
        delay_mw_init = self.settings['read_out']['delay_mw_init']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']
        number_of_pi_pulses = self.settings['mw_pulses']['number_of_pi_pulses']


        for tau in tau_list:

            # pulse_sequence = [Pulse('laser', 0, reset_time - ref_meas_off_time - 15 - meas_time),
            #                   Pulse('apd_readout', reset_time - 15 - meas_time, meas_time),
            #                   Pulse('laser', reset_time - 15 - meas_time, meas_time),
            #                   Pulse('microwave_i', reset_time + delay_mw_init, pi_half_time)
            #                   ]
            # # 16-08-25 JG: changed :
            pulse_sequence = [Pulse('laser', 0, reset_time - ref_meas_off_time - 15 - meas_time),
                              Pulse('apd_readout', reset_time - 15 - meas_time, meas_time),
                              Pulse('laser', reset_time - 15 - meas_time, meas_time),
                              Pulse('microwave_i', reset_time + delay_mw_init-pi_half_time/2, pi_half_time)
                              ]


            # next_pi_pulse_time = reset_time + delay_mw_init + pi_half_time + tau
            # # 16-08-19 JG: changed :
            # next_pi_pulse_time = reset_time + delay_mw_init + tau/2
            # # 16-08-25 JG: changed :
            next_pi_pulse_time = reset_time + delay_mw_init - pi_half_time / 2 + tau / 2

            for n in range(1, number_of_pi_pulses + 1):
                pulse_sequence.extend([Pulse('microwave_q', next_pi_pulse_time,pi_time)])
                # next_pi_pulse_time += tau*2 + pi_time
                # 16-08-19 JG: changed:
                # next_pi_pulse_time += tau
                # 16 - 08 -24 JG: changed
                next_pi_pulse_time += tau/2

            # pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time-tau, pi_half_time),
            #                         Pulse('laser', next_pi_pulse_time-tau + delay_mw_readout + pi_half_time, meas_time),
            #                         Pulse('apd_readout',next_pi_pulse_time-tau + delay_mw_readout + pi_half_time, meas_time)
            #                         ])
            # 16-08-19 JG: changed:
            # pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time-tau/2 + pi_half_time, pi_half_time),
            #                         Pulse('laser',      next_pi_pulse_time-tau/2 + pi_time + delay_mw_readout, meas_time),
            #                         Pulse('apd_readout',next_pi_pulse_time-tau/2 + pi_time + delay_mw_readout, meas_time)
            #                         ])
            # pulse_sequences.append(pulse_sequence)
            # 16 - 08 -24 JG: changed
            pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time + pi_half_time, pi_half_time),
                                   Pulse('laser', next_pi_pulse_time + pi_time + delay_mw_readout, meas_time),
                                   Pulse('apd_readout', next_pi_pulse_time + pi_time + delay_mw_readout,
                                         meas_time)
                                   ])

            # # 16-08-25 JG: changed :
            pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time - pi_half_time/2, pi_half_time),
                                   Pulse('laser', next_pi_pulse_time + pi_half_time + delay_mw_readout, meas_time),
                                   Pulse('apd_readout', next_pi_pulse_time + pi_half_time + delay_mw_readout, meas_time)
                                   ])

            pulse_sequences.append(pulse_sequence)

        # TEMPORATTY: THIS IS TO SEE IF THE OVERALL TIME OF A SEQUENCE SHOULD ALWAYS BE THE SAME
        # IF WE WANT TO KEEP THIS ADD ADDITIONAL PARAMETER TO THE SCRIPT SETTINGS
        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

class XY(PulseBlasterBaseScript):
    """
This script runs a XY sequence for different number of pi pulses. Without pi-pulse this is a Ramsey sequence.
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_power', -45.0, float, 'microwave power in dB'),
        Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
        Parameter('pi_half_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
        Parameter('number_of__pi_pulses', 0, range(0,17), 'number of pi pulses'),
        Parameter('tau', [
            Parameter('min', 15, float, 'min value for tau, the free evolution time in between pulses (in ns)'),
            Parameter('max', 30, float, 'max value for tau, the free evolution time in between pulses (in ns)'),
            Parameter('step', 5, float, 'step size for tau, the free evolution time in between pulses (in ns)'),
        ]),
        Parameter('meas_time', 300, float, 'measurement time after CPMG sequence (in ns)'),
        Parameter('num_averages', 1000, int, 'number of averages (should be less than a million)'),
        Parameter('reset_time', 1000, int, 'time duration of the green laser to reset the spin state'),
        Parameter('delay_init_mw', 100, int, 'delay between initialization and mw (in ns)'),
        Parameter('delay_mw_readout', 100, int, 'delay between mw and readout (in ns)'),
        Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)'),
        Parameter('skip_invalid_sequences', False, bool, 'Skips any sequences with <15ns commands')
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    _SCRIPTS = {}


    def __init__(self, instruments, scripts=None, name=None, settings=None, log_function=None, data_path=None):
        Script.__init__(self, name, settings=settings, scripts=scripts, instruments=instruments,
                        log_function=log_function, data_path=data_path)

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        super(XY, self)._function()


    def _create_pulse_sequences(self):
        '''
        creates the pulse sequence for the Hahn echo /
        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(max(15,self.settings['tau']['min'])), int(self.settings['tau']['max']),int(self.settings['tau']['step']))
        reset_time = self.settings['reset_time']
        mw_delay_time = self.settings['delay_init_mw']
        delay_after_mw = self.settings['delay_mw_readout']
        pi_half_pulse_time = self.settings['pi_half_pulse_time']
        meas_time  = self.settings['meas_time']
        number_of__pi_pulses =  self.settings['number_of__pi_pulses']

        for tau in tau_list:
            # if number_of__pi_pulses == 0:
            #     pulse_sequences.append([Pulse('laser', 0, reset_time),
            #                             Pulse('microwave_i', reset_time+ mw_delay_time, pi_half_pulse_time),
            #                             Pulse('microwave_i', reset_time + mw_delay_time+ pi_half_pulse_time + tau, pi_half_pulse_time),
            #                             Pulse('laser', reset_time + mw_delay_time+ pi_half_pulse_time + tau + pi_half_pulse_time, meas_time),
            #                             Pulse('apd_readout', reset_time + mw_delay_time+ pi_half_pulse_time + tau + pi_half_pulse_time, meas_time)
            #                             ])
            # else:

            pulse_sequence = []

            pulse_sequence.extend([Pulse('laser', 0, reset_time - self.settings['ref_meas_off_time'] - 15 - self.settings['meas_time']),
                                    Pulse('apd_readout', reset_time - 15 - self.settings['meas_time'], self.settings['meas_time']),
                                    Pulse('laser', reset_time - 15 - self.settings['meas_time'], self.settings['meas_time']),
                                    Pulse('microwave_i', reset_time + mw_delay_time, pi_half_pulse_time)
                                    ])

            next_pi_pulse_time = reset_time + mw_delay_time + pi_half_pulse_time + tau

            for n in range(1, number_of__pi_pulses + 1):
                pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time, 2*pi_half_pulse_time)])
                pulse_sequence.extend([Pulse('microwave_q', next_pi_pulse_time + 2*pi_half_pulse_time + 2*tau, 2*pi_half_pulse_time)])
                next_pi_pulse_time += tau*4 + 4*pi_half_pulse_time

            pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time-tau,pi_half_pulse_time),
                                    Pulse('laser', next_pi_pulse_time-tau + delay_after_mw + pi_half_pulse_time, meas_time),
                                    Pulse('apd_readout',next_pi_pulse_time-tau + delay_after_mw + pi_half_pulse_time, meas_time)
                                    ])

            pulse_sequences.append(pulse_sequence)


        # TEMPORATTY: THIS IS TO SEE IF THE OVERALL TIME OF A SEQUENCE SHOULD ALWAYS BE THE SAME
        # IF WE WANT TO KEEP THIS ADD ADDITIONAL PARAMETER TO THE SCRIPT SETTINGS
        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']


class T1(PulseBlasterBaseScript):
    """
This script measures the relaxation time of an NV center
    """
    _DEFAULT_SETTINGS = [
        Parameter('time_step', 1000, int, 'time step increment of T1 measurement (ns)'),
        Parameter('max_time', 200000, float, 'total time of T1 measurement (ns)'),
        Parameter('meas_time', 300, float, 'measurement time of fluorescence counts (ns)'),
        Parameter('num_averages', 1000000, int, 'number of averages'),
        Parameter('nv_reset_time', 3000, int, 'time with laser on at the beginning to reset state (ns)'),
        Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15 ns commands'),
        Parameter('tau_scale', 'linear', ['linear', 'logarithmic'])
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster}

    def _create_pulse_sequences(self):
        """
        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a  pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        """

        pulse_sequences = []
        if self.settings['time_step'] % 5 != 0:
            raise AttributeError('given time_step is not a multiple of 5')

        tau_list = range(0, int(self.settings['max_time'] + self.settings['time_step']), self.settings['time_step'])
        reset_time = self.settings['nv_reset_time']

        # reduce the initialization time by 15 ns to avoid touching DAQ pulses
        # (they are problematic because the DAQ expects two pulse but get only one because they get merged by the pulse blaster)
        for tau in tau_list:
            pulse_sequences.append(
                [Pulse('laser', 0, reset_time - self.settings['ref_meas_off_time'] - 15 - self.settings['meas_time']),
                 Pulse('apd_readout', reset_time - 15 - self.settings['meas_time'], self.settings['meas_time']),
                 Pulse('laser', reset_time - 15 - self.settings['meas_time'], self.settings['meas_time']),
                 Pulse('apd_readout', reset_time + tau, self.settings['meas_time']),
                 Pulse('laser', reset_time + tau, self.settings['meas_time']),
                 ])

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['meas_time']

    def _plot(self, axislist, data = None):
        """
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first)
            data (optional): dataset to plot (dictionary that contains keys counts, tau), if not provided use self.data
        """
        super(T1, self)._plot(axislist, data)
        axislist[0].set_title('T1')
        axislist[0].legend(labels=( 'Ref Fluorescence', 'T1 data'), fontsize=8)


class T1SpinFlip(PulseBlasterBaseScript):
    """
This script measures the relaxation time of an NV center.
Optionally a microwave pulse is applied as part of the initialization to prepare the system in a different state
    """
    _DEFAULT_SETTINGS = [
        Parameter(
            'tau_times',
            [
                Parameter('time_step', 1000, int, 'time step increment of T1 measurement (ns)'),
                Parameter('max_time', 200000, float, 'max time of T1 measurement (ns)'),
                Parameter('min_time', 0, float, 'min time of T1 measurement (ns)')
            ]
        ),
        Parameter('num_averages', 1000000, int, 'number of averages'),
        Parameter(
            'read_out',
            [
                Parameter('meas_time', 700, float, 'measurement time of fluorescence counts (ns)'),
                Parameter('nv_reset_time', 3000, int, 'time with laser on at the beginning to reset state (ns)'),
                Parameter('ref_meas_off_time', 1000, int, 'laser off time before taking reference measurement at the end of init (ns)')
            ]
        ),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15 ns commands'),
        Parameter('apply mw-pulse', True, bool, 'if true a pi pulse is at the beginning of the measurement'),
        Parameter('mw-pulse',
                  [
                      Parameter('mw_frequency', 2.87e9, float, 'microwave frequency of pi pulse (Hz)'),
                      Parameter('mw_duration', 300, int, 'pi pulse duration (ns)'),
                      Parameter('mw_channel', 'i', ['i', 'q'], 'select i or q channel for i pulse'),
                      Parameter('mw_power', -2, float, 'microwave power (dB)')
                  ]
                  )
    ]

    _INSTRUMENTS = {'daq': DAQ, 'PB': B26PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({
            'modulation_type': 'IQ',
            'amplitude': self.settings['mw-pulse']['mw_power'],
            'frequency': self.settings['mw-pulse']['mw_frequency']
        })
        super(T1SpinFlip, self)._function()

    def _create_pulse_sequences(self):
        """
        Returns: pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a  pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        """

        pulse_sequences = []
        if self.settings['tau_times']['time_step'] % 5 != 0:
            raise AttributeError('given time_step is not a multiple of 5')

        tau_list = range(int(self.settings['tau_times']['min_time']),
                         int(self.settings['tau_times']['max_time'] + self.settings['tau_times']['time_step']),
                         self.settings['tau_times']['time_step']
                         )


        # if self.settings['apply mw-pulse']:
        #     ref_meas_off_time = self.settings['read_out']['ref_meas_off_time'] + self.settings['mw-pulse']['mw_duration']
        # else:
        ref_meas_off_time = self.settings['read_out']['ref_meas_off_time']
        reset_time = self.settings['read_out']['nv_reset_time']
        meas_time = self.settings['read_out']['meas_time']

        microwave_channel = 'microwave_' + self.settings['mw-pulse']['mw_channel']
        microwave_duration = self.settings['mw-pulse']['mw_duration']

        # reduce the initialization time by 15 ns to avoid touching DAQ pulses
        # (they are problematic because the DAQ expects two pulse but get only one because they get merged by the pulse blaster)
        def build_sequence(tau):
            """
            builds the sequence for a given tau
            Args:
                tau: the time after the initialization at which to measure the population

            Returns: the sequence for tau

            """

            if self.settings['apply mw-pulse']:
                sequence = [
                    Pulse('laser', 0,       reset_time - ref_meas_off_time - meas_time - 15 - ref_meas_off_time- microwave_duration),
                    Pulse('apd_readout',    reset_time - meas_time - 15 - ref_meas_off_time- microwave_duration, meas_time),
                    Pulse('laser',          reset_time - meas_time - 15 - ref_meas_off_time- microwave_duration, meas_time),
                    Pulse(microwave_channel,reset_time - 15 - ref_meas_off_time/2.- microwave_duration, microwave_duration)
                ]
            else:
                sequence = [
                    Pulse('laser', 0, reset_time - ref_meas_off_time - meas_time - 15),
                    Pulse('apd_readout', reset_time - 15 - meas_time, meas_time),
                    Pulse('laser', reset_time - 15 - meas_time, meas_time)
                ]

            sequence += [
                Pulse('apd_readout', reset_time + tau, meas_time),
                Pulse('laser',       reset_time + tau, meas_time)
            ]

            return sequence

        pulse_sequences = [build_sequence(tau) for tau in tau_list]

        return pulse_sequences, self.settings['num_averages'], tau_list, self.settings['read_out']['meas_time']

    def _plot(self, axislist, data = None):
        """
        Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
        received for each time
        Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
        performed

        Args:
            axes_list: list of axes to write plots to (uses first)
            data (optional): dataset to plot (dictionary that contains keys counts, tau), if not provided use self.data
        """
        super(T1SpinFlip, self)._plot(axislist, data)
        axislist[0].set_title('T1')
        axislist[0].legend(labels=( 'Ref Fluorescence', 'T1 data'), fontsize=8)

if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'CalibrateMeasurementWindow': 'CalibrateMeasurementWindow'}, script, instr)

    print(script)
    print('failed', failed)
    print(instr)