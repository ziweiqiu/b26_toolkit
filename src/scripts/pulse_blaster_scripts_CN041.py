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
import random

from copy import deepcopy
from b26_toolkit.src.scripts.pulse_blaster_base_script import PulseBlasterBaseScript
from b26_toolkit.src.instruments import NI6259, CN041PulseBlaster, MicrowaveGenerator, R8SMicrowaveGenerator, Pulse
from b26_toolkit.src.plotting.plots_1d import plot_esr, plot_pulses, update_pulse_plot, plot_1d_simple_timetrace_ns, update_1d_simple
from PyLabControl.src.core import Parameter, Script
from b26_toolkit.src.data_processing.fit_functions import fit_rabi_decay, cose_with_decay, fit_exp_decay, exp_offset

MAX_AVERAGES_PER_SCAN = 100000

class Rabi(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power for varying durations to measure Rabi Oscillations
==> Last edited by Alexei Bylinskii 06/28/2017 for use in CN041 sensing lab
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', 3, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.64e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 15, float, 'minimum time for rabi oscillations (in ns)'),
            Parameter('max_time', 200, float, 'total time of rabi oscillations (in ns)'),
            Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 100000, 500000],
                  'time step increment of rabi pulse duration (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, int, '[ns] APD window to count photons at the beginning of optical pulse'),
            Parameter('nv_reset_time', 2000, float, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int, '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 500000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0,20,30,40], '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):

        self.data['fits'] = None
        self.data['pi_time'] = 0
        self.data['pi_half_time'] = 0
        self.data['three_pi_half_time'] = 0

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(Rabi, self)._function(self.data)

        ### Turn off green light:
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        counts = (self.data['counts'][:, 1]-self.data['counts'][:, 0]) / self.data['counts'][:, 0]
        tau = self.data['tau']

        try:
            fits = fit_rabi_decay(tau, counts, varibale_phase=True)
            self.data['fits'] = fits
        except:
            self.data['fits'] = None
            self.log('rabi fit failed')

        if self.data['fits'] is not None:
            counts = self.data['counts'][:,1]/ self.data['counts'][:,0]
            tau =self.data['tau']
            fits = self.data['fits']

            #RabiT = 2*np.pi / fits[1]
            RabiT = abs(2 * np.pi / fits[1]) # avoid negative rabi period ZQ 8/25/2017
            T2star = fits[4]
            phaseoffs = fits[2]
            pi_time = RabiT / 2 - phaseoffs * RabiT / (2 * np.pi)
            pi_half_time = RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
            three_pi_half_time = 3 * RabiT / 4 - phaseoffs * RabiT / (2 * np.pi)
            # pi_time = RabiT / 2
            # pi_half_time = RabiT / 4
            # three_pi_half_time = 3 * RabiT / 4
            self.data['phaseoffs'] = phaseoffs * RabiT / (2 * np.pi)
            self.data['pi_time'] = pi_time
            self.data['pi_half_time'] = pi_half_time
            self.data['three_pi_half_time'] = three_pi_half_time

            # # round float number to integer to avoid short pulse validation failure, which happens sometimes without clear reason
            # # ZQ 8/26/2017
            # self.data['pi_time'] = round(pi_time,0)
            # self.data['pi_half_time'] = round(pi_half_time,0)
            # self.data['three_pi_half_time'] = round(three_pi_half_time,0)


    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        # JG 16-08-25 changed (15ns min spacing is taken care of later):
        # tau_list = range(0, int(self.settings['tau_times']['max_time']), self.settings['tau_times']['time_step'])
        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']
        mw_sw_buffer = self.settings['mw_switch_extra_time']

        for tau in tau_list:
            pulse_sequence = \
                [Pulse('laser', laser_off_time + tau + 2*mw_sw_buffer, nv_reset_time),
                 Pulse('apd_readout', laser_off_time + tau + 2*mw_sw_buffer + delay_readout, meas_time),
                 ]
            # if tau is 0 there is actually no mw pulse
            if tau > 0:
                pulse_sequence += [Pulse(microwave_channel, laser_off_time + tau + 2*mw_sw_buffer + nv_reset_time + laser_off_time, tau)]

            pulse_sequence += [
                Pulse('laser', laser_off_time + tau + 2*mw_sw_buffer + nv_reset_time + laser_off_time + tau + 2*mw_sw_buffer + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', laser_off_time + tau + 2*mw_sw_buffer + nv_reset_time + laser_off_time + tau + 2*mw_sw_buffer + delay_mw_readout + delay_readout, meas_time)
            ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time


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
            counts = (data['counts'][:, 1]-data['counts'][:, 0]) / data['counts'][:, 0]
            tau = data['tau']
            fits = data['fits']

            axislist[0].plot(tau, counts, 'b')
            axislist[0].hold(True)

            tauinterp = np.linspace(0.,np.max(tau),100)
            axislist[0].plot(tauinterp, cose_with_decay(tauinterp, *fits), 'k', lw=3)
            RabiT = 2*np.pi / fits[1]
            T2star = fits[4]
            phaseoffs = fits[2]
            phaseoffs = self.data['phaseoffs']
            pi_time = self.data['pi_time']
            pi_half_time = self.data['pi_half_time']
            three_pi_half_time = self.data['three_pi_half_time']

            axislist[0].plot(pi_time, cose_with_decay(pi_time, *fits), 'ro', lw=3)
            axislist[0].annotate('$\pi$={:0.1f}'.format(pi_time), xy = (pi_time, cose_with_decay(pi_time, *fits)), xytext = (pi_time - 10., cose_with_decay(pi_time, *fits)-.01), xycoords = 'data')
            axislist[0].plot(pi_half_time, cose_with_decay(pi_half_time, *fits), 'ro', lw=3)
            axislist[0].annotate('$\pi/2$={:0.1f}'.format(pi_half_time), xy = (pi_half_time, cose_with_decay(pi_half_time, *fits)), xytext = (pi_half_time + 10., cose_with_decay(pi_half_time, *fits)), xycoords = 'data')
            axislist[0].plot(three_pi_half_time, cose_with_decay(three_pi_half_time, *fits), 'ro', lw=3)
            axislist[0].annotate('$3\pi/2$={:0.1f}'.format(three_pi_half_time), xy = (three_pi_half_time, cose_with_decay(three_pi_half_time, *fits)), xytext = (three_pi_half_time + 10., cose_with_decay(three_pi_half_time, *fits)), xycoords = 'data')
            axislist[0].plot(-phaseoffs, cose_with_decay(-phaseoffs, *fits), 'gd', lw=3)
            axislist[0].annotate('$start$={:0.1f}'.format(-phaseoffs), xy=(-phaseoffs, cose_with_decay(-phaseoffs, *fits)), xytext=(-phaseoffs + 10., cose_with_decay(-phaseoffs, *fits)), xycoords='data')

            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz, Rabi-period: {:2.1f}ns, T2*: {:2.1f}ns'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9, RabiT, T2star))
        else:
            super(Rabi, self)._plot(axislist)
            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9))
            axislist[0].legend(labels=('Ref Fluorescence', 'Rabi Data'), fontsize=8)

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(Rabi, self)._update_plot(axislist)

            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz'.format(self.settings['mw_pulses']['mw_power'],
                                                                                         self.settings['mw_pulses'][
                                                                                             'mw_frequency'] * 1e-9))
            axislist[0].legend(labels=('Ref Fluorescence', 'Rabi Data'), fontsize=8)

class GrDelayMeas(PulseBlasterBaseScript):
    """
This script measures the green laser delay during AOM turn ON and turn OFF
==> Last edited by Alexei Bylinskii 06/28/2017 for use in CN041 sensing lab
    """
    _DEFAULT_SETTINGS = [
        Parameter('tau_times', [
            Parameter('min_time', 15, float, 'minimum green delay (in ns)'),
            Parameter('max_time', 600, float, 'max green delay (in ns)'),
            Parameter('time_step', 15, [15,20,30,50,100],'time step increment of green delay (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 30, int, '[ns] APD window to count photons'),
            Parameter('green_time', 300, float, '[ns] duration of green pulse '),
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster}

    def _function(self):
        super(GrDelayMeas, self)._function(self.data)
        counts = self.data['counts']
        tau = self.data['tau']

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        # JG 16-08-25 changed (15ns min spacing is taken care of later):
        # tau_list = range(0, int(self.settings['tau_times']['max_time']), self.settings['tau_times']['time_step'])
        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        meas_time = self.settings['read_out']['meas_time']
        green_time = self.settings['read_out']['green_time']

        for tau in tau_list:
            pulse_sequence = \
                [Pulse('laser', 1000, green_time),
                 Pulse('apd_readout', 1000+tau, meas_time),
                 ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time


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

        super(GrDelayMeas, self)._plot(axislist)

class HahnEcho(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a Hahn echo on the NV to find the Hahn echo T2.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 5, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 300, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 1000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 1000, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 30, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 10, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        super(HahnEcho, self)._function(self.data)

        counts = (- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        tau = self.data['tau']

        try:
            fits = fit_exp_decay(tau, counts, offset = True, verbose = True)
            self.data['fits'] = fits
        except:
            self.data['fits'] = None
            self.log('t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']


        for tau in tau_list:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time



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
            counts = (-data['counts'][:,1] + data['counts'][:,0])/ (data['counts'][:,0] + data['counts'][:,1])
            tau = data['tau']
            fits = data['fits']

            axislist[0].plot(tau, counts, 'b')
            axislist[0].hold(True)

            axislist[0].plot(tau, exp_offset(tau, fits[0], fits[1], fits[2]))
            axislist[0].set_title('T2 decay time (simple exponential, p = 1): {:2.1f} ns'.format(fits[1]))
        else:
            super(HahnEcho, self)._plot(axislist)
            axislist[0].set_title('Rabi mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9))
            axislist[0].legend(labels=('Ref Fluorescence', 'T2 Data'), fontsize=8)

class DEER(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a Hahn echo on the NV to find the Hahn echo T2.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -3.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -15.0, float, 'microwave power in dB'),
            Parameter('RF_frequency', 224e6, float, 'microwave frequency in Hz')
            #Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 500, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']
        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
            self.data['fits_deer'] = fits
        except:
            self.data['fits_deer'] = None
            self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']


        for tau in tau_list:
            #ECHO SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            #DEER SEQUENCE

            start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

    # # Ziwei 7/27/17
    # def _create_pulse_sequences(self):
    #     '''
    #
    #     Returns: pulse_sequences, num_averages, tau_list, meas_time
    #         pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
    #         scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
    #         sequence must have the same number of daq read pulses
    #         num_averages: the number of times to repeat each pulse sequence
    #         tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
    #         meas_time: the width (in ns) of the daq measurement
    #
    #     '''
    #     pulse_sequences = []
    #
    #     tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])
    #
    #     # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
    #     tau_list = [x for x in tau_list if x == 0 or x >= 15]
    #     print('tau_list', tau_list)
    #
    #     microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']
    #
    #     meas_time = self.settings['read_out']['meas_time']
    #     nv_reset_time = self.settings['read_out']['nv_reset_time']
    #     delay_readout = self.settings['read_out']['delay_readout']
    #     laser_off_time = self.settings['read_out']['laser_off_time']
    #     delay_mw_readout = self.settings['read_out']['delay_mw_readout']
    #
    #     pi_time = self.settings['mw_pulses']['pi_pulse_time']
    #     pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
    #     three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
    #
    #     mw_sw_buffer = self.settings['mw_switch_extra_time']
    #
    #     #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
    #     #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
    #     #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']
    #
    #
    #     for tau in tau_list:
    #         #ECHO SEQUENCE:
    #         pulse_sequence = \
    #         [
    #             Pulse(microwave_channel, laser_off_time, pi_half_time),
    #             Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
    #             Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
    #         ]
    #
    #         end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time
    #
    #         pulse_sequence += [
    #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
    #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
    #              ]
    #
    #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
    #
    #         pulse_sequence += \
    #         [
    #             Pulse(microwave_channel, start_of_second_HE, three_pi_half_time),
    #             Pulse(microwave_channel, start_of_second_HE + three_pi_half_time/6. + tau - pi_time/2., pi_time),
    #             Pulse(microwave_channel, start_of_second_HE + three_pi_half_time/6. + tau + tau - pi_half_time/2., pi_half_time)
    #         ]
    #
    #         end_of_second_HE = start_of_second_HE + three_pi_half_time/6. + tau + tau - pi_half_time/2. + pi_half_time
    #
    #         pulse_sequence += [
    #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
    #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
    #         ]
    #
    #         #DEER SEQUENCE
    #
    #         start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
    #         pulse_sequence += \
    #         [
    #             Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
    #             Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
    #             Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - pi_time / 2., pi_time),
    #             Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
    #         ]
    #
    #         end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time
    #
    #         pulse_sequence += [
    #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
    #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
    #              ]
    #
    #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
    #
    #         pulse_sequence += \
    #         [
    #             Pulse(microwave_channel, start_of_second_HE, three_pi_half_time),
    #             Pulse(microwave_channel, start_of_second_HE + three_pi_half_time/6. + tau - pi_time/2., pi_time),
    #             Pulse('RF_switch', start_of_second_HE + three_pi_half_time / 6. + tau - pi_time / 2., pi_time),
    #             Pulse(microwave_channel, start_of_second_HE + three_pi_half_time/6. + tau + tau - pi_half_time/2., pi_half_time)
    #         ]
    #
    #         end_of_second_HE = start_of_second_HE + three_pi_half_time/6. + tau + tau - pi_half_time/2. + pi_half_time
    #
    #         pulse_sequence += [
    #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
    #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
    #         ]
    #
    #         pulse_sequences.append(pulse_sequence)
    #
    #     print('number of sequences before validation ', len(pulse_sequences))
    #     return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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
            tau = data['tau']

        if data['fits_echo'] is not None and data['fits_deer'] is not None:
            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
            axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)
        else:
            super(DEER, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)

            axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER, self)._update_plot(axislist)
            axislist[0].legend(labels=('Echo up ',
                                       'Echo down ',
                                       'DEER up ',
                                       'DEER down'), fontsize=8)

            axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))


class DEER_XYn(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a DEER sequence on the NV with different decoupling sequence options.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
==> last edited by Ziwei Qiu 8/25/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', 3, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.64e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for pi/2 pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -7, float, 'microwave power in dB'),
            Parameter('RF_frequency', 224e6, float, 'microwave frequency in Hz'),
            Parameter('RF_pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)')
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi/2 pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi/2 pulses'),
            Parameter('time_step', 500, [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 500000],
                  'time step increment of time between pi/2 pulses (in ns)')
        ]),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo',['spin_echo','XY4','XY8','CPMG'], 'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 1200000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_XYn, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']
        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
            self.data['fits_deer'] = fits
        except:
            self.data['fits_deer'] = None
            self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        if self.settings['mw_pulses']['microwave_channel'] == 'i':
            microwave_channel_1 = 'microwave_i'
            microwave_channel_2 = 'microwave_q'
        else:
            microwave_channel_1 = 'microwave_q'
            microwave_channel_2 = 'microwave_i'

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
        RF_pi_time = self.settings['RF_pulses']['RF_pi_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']
        number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']

        # if self.settings['decoupling_seq']['type'] == 'spin_echo':
        #     for tau_total in tau_list:
        #         tau = tau_total/2
        #         #ECHO SEQUENCE:
        #         pulse_sequence = \
        #         [
        #             Pulse(microwave_channel_1, laser_off_time, pi_half_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE = laser_off_time + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2 * tau, three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2 * tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         #DEER SEQUENCE
        #
        #         start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_DEER , pi_half_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_DEER + pi_half_time + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE =  start_of_DEER + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_second_HE + pi_half_time  + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2*tau , three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2*tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         pulse_sequences.append(pulse_sequence)
        #
        #     print('number of sequences before validation ', len(pulse_sequences))
        #     return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

        if self.settings['decoupling_seq']['type'] == 'spin_echo':
            for tau_total in tau_list:
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY4':
            for tau_total in tau_list:
                tau = tau_total/ (4*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau/2 #for the first pulse, only wait tau/2
                for i in range(0,number_of_pulse_blocks):
                    pulse_sequence.extend([Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time/2, pi_time)
                                           ])
                    section_begin_time += 4 * tau

                #the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau/2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau/2 + pi_half_time + delay_mw_readout, nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau/2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau/2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])


                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY8':
            for tau_total in tau_list:
                tau = tau_total/ (8*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'CPMG':
            for tau_total in tau_list:
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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
            tau = data['tau']

        if data['fits_echo'] is not None and data['fits_deer'] is not None:

            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            axislist[0].set_title('DEER {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz \n T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6, fits_echo[1],fits_deer[1]))
            axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)


            # super(DEER_XYn, self)._plot(axislist)
            # axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

        else:
            super(DEER_XYn, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            axislist[0].set_title('DEER {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER_XYn, self)._update_plot(axislist)

            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_title('DEER {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

class DEER_XYn_RFpwrsw(PulseBlasterBaseScript): # ER 5.25.2017

    """
        This script runs a DEER sequence on the NV scanning over RF power.
        There are different decoupling sequence options.
        To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.

        ==> last edited by Ziwei Qiu 8/25/2017
    """

    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', 3, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.64e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for pi/2 pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -6.0, float, 'RF center power in dBm'),
            Parameter('RF_power_range', 10, float, 'RF power range in dBm'),
            Parameter('RF_power_points', 11, int, 'number of powers in the scan'),
            Parameter('RF_frequency', 224e6, float, 'RF frequency in Hz'),
            Parameter('RF_pi_pulse_time', 50.0, float, 'time duration of an RF pi pulse (in ns)')
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_time', 4000, float, '[ns] tau time between two pi/2 pulses'),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo',['spin_echo','XY4','XY8','CPMG'], 'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse blocks.')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 1200000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['norm_echo'] = None
        self.data['norm_deer'] = None


        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_XYn_RFpwrsw, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']
        # no fitting for RF power scan
        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
        #     self.data['fits_echo'] = fits
        # except:
        #     self.data['fits_echo'] = None
        #     self.log('ECHO t2 fit failed')
        #
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
        #     self.data['fits_deer'] = fits
        # except:
        #     self.data['fits_deer'] = None
        #     self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        RF_power_start = self.settings['RF_pulses']['RF_power'] - self.settings['RF_pulses']['RF_power_range']/2.0
        RF_power_stop = self.settings['RF_pulses']['RF_power'] + self.settings['RF_pulses']['RF_power_range']/2.0
        RF_power_list = np.linspace(RF_power_start, RF_power_stop, num = self.settings['RF_pulses']['RF_power_points']).tolist()
        print('RF_power_list:', RF_power_list)
        # tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        # tau_list = [x for x in tau_list if x == 0 or x >= 15]


        if self.settings['mw_pulses']['microwave_channel'] == 'i':
            microwave_channel_1 = 'microwave_i'
            microwave_channel_2 = 'microwave_q'
        else:
            microwave_channel_1 = 'microwave_q'
            microwave_channel_2 = 'microwave_i'

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
        RF_pi_time = self.settings['RF_pulses']['RF_pi_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']
        number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
        tau_total = self.settings['tau_time']

        # if self.settings['decoupling_seq']['type'] == 'spin_echo':
        #     for RF_power_current in RF_power_list:
        #         # self.instruments['RF_gen']['instance'].update({'power': RF_power_current})
        #         tau = tau_total/2
        #         #ECHO SEQUENCE:
        #         pulse_sequence = \
        #         [
        #             Pulse(microwave_channel_1, laser_off_time, pi_half_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE = laser_off_time + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2 * tau, three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2 * tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         #DEER SEQUENCE
        #
        #         start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_DEER , pi_half_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_DEER + pi_half_time + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE =  start_of_DEER + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_second_HE + pi_half_time  + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2*tau , three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2*tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         pulse_sequences.append(pulse_sequence)
        #
        #     print('number of sequences before validation ', len(pulse_sequences))
        #     return pulse_sequences, self.settings['num_averages'], RF_power_list, meas_time
        if self.settings['decoupling_seq']['type'] == 'spin_echo':
            for RF_power_current in RF_power_list:
                # self.instruments['RF_gen']['instance'].update({'power': RF_power_current})
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_power_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY4':
            for RF_power_current in RF_power_list:
                # self.instruments['RF_gen']['instance'].update({'power': RF_power_current})
                tau = tau_total/ (4*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau/2 #for the first pulse, only wait tau/2
                for i in range(0,number_of_pulse_blocks):
                    pulse_sequence.extend([Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time/2, pi_time)
                                           ])
                    section_begin_time += 4 * tau

                #the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau/2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau/2 + pi_half_time + delay_mw_readout, nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau/2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau/2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])


                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_power_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY8':
            for RF_power_current in RF_power_list:
                # self.instruments['RF_gen']['instance'].update({'power': RF_power_current})
                tau = tau_total/ (8*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_power_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'CPMG':
            for RF_power_current in RF_power_list:
                # self.instruments['RF_gen']['instance'].update({'power': RF_power_current})
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_power_list, meas_time

    def _run_sweep(self, pulse_sequences, num_loops_sweep, num_daq_reads, tau_list):
        '''
        Each pulse sequence specified in pulse_sequences is run num_loops_sweep consecutive times.

        Args:
            pulse_sequences: a list of pulse sequences to run, each corresponding to a different value of tau. Each
                             sequence is a list of Pulse objects specifying a given pulse sequence
            num_loops_sweep: number of times to repeat each sequence before moving on to the next one
            num_daq_reads: number of times the daq must read for each sequence (generally 1, 2, or 3)

        Poststate: self.data['counts'] and self.data['shot_noise'] are updated with the acquired data

        '''

        # randomize the indexes of the pulse sequences to run, to reduce heating. ER 5/25/2017
        rand_indexes = []
        for i in range(0, len(pulse_sequences)):
            rand_indexes.append(i)
        if self.settings['randomize']:
            random.shuffle(rand_indexes)
        for index, sequence in enumerate(pulse_sequences):
            rand_index = rand_indexes[index]

            self.instruments['RF_gen']['instance'].update({'power': tau_list[rand_index]})

            if self._abort:
                break

            result = self._single_sequence(pulse_sequences[rand_index], num_loops_sweep,
                                           num_daq_reads)  # keep entire array

            counts_temp = 0
            if not result:
                print('Throwing away results of sequence {}'.format(rand_index))
                self.log('Throwing away results of sequence {}'.format(rand_index))
                self.nfails[rand_index] += 1
            else:
                self.count_data[rand_index] = self.count_data[rand_index] + result
                counts_to_check = self._normalize_to_kCounts(np.array(result), self.measurement_gate_width,
                                                             num_loops_sweep)
                self.data['counts'][rand_index] = self._normalize_to_kCounts(self.count_data[rand_index],
                                                                             self.measurement_gate_width,
                                                                             self.current_averages - self.nfails[
                                                                                 rand_index] * MAX_AVERAGES_PER_SCAN)
                self.data['shot_noise'][rand_index] = np.reciprocal(np.sqrt(self.count_data[rand_index]))
                counts_temp = counts_to_check[0]

            self.sequence_index = rand_index

            # throw error if tracking is on and you haven't ran find nv ER 6/2/17
            if self.settings['Tracking']['on/off']:
                if self.scripts['find_nv'].data['fluorescence']:
                    self.data['init_fluor'] = deepcopy(self.scripts['find_nv'].data['fluorescence'])
                else:
                    raise AttributeError('need to run find NV first for tracking!')

            # track to the NV if necessary ER 5/31/17
            if (self.settings['Tracking']['on/off']):
                if (self.settings['Tracking']['threshold'] * self.data['init_fluor'] > counts_temp or
                                (2 - self.settings['Tracking']['threshold']) * self.data['init_fluor'] < counts_temp):
                    #      self._plot_refresh = True
                    print('TRACKING TO NV...')
                    self.scripts['find_nv'].run()
                    #     self._plot_refresh = True
                    self.scripts['find_nv'].settings['initial_point'] = self.scripts['find_nv'].data['maximum_point']
            self.updateProgress.emit(self._calc_progress(index))

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
            tau = data['tau']

        if data['norm_echo'] is not None and data['norm_deer'] is not None:

            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            axislist[0].legend(labels=('Echo', 'DEER'), fontsize=8)

            axislist[0].set_title(
                'DEER (RF power scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                    self.settings['tau_time'],
                    self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))
            axislist[0].set_xlabel('RF power [dBm]')
            # super(DEER_XYn, self)._plot(axislist)
            # axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

        else:
            super(DEER_XYn_RFpwrsw, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            axislist[0].set_title('DEER (RF power scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['tau_time'], self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_xlabel('RF power [dBm]')
    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER_XYn_RFpwrsw, self)._update_plot(axislist)

            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_title('DEER (RF power scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['tau_time'], self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_xlabel('RF power [dBm]')

class DEER_XYn_RFfreqsw(PulseBlasterBaseScript): # ER 5.25.2017

    """
        This script runs a DEER sequence on the NV scanning over RF frequency.
        There are different decoupling sequence options.
        To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.

        ==> last edited by Ziwei Qiu 8/25/2017
    """

    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', 3.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.64e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for pi/2 pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -7.0, float, 'RF center power in dBm'),
            Parameter('RF_frequency', 224e6, float, 'RF frequency in Hz'),
            Parameter('RF_freq_range', 100e6, float, 'RF frequency range in Hz'),
            Parameter('RF_freq_points', 21, int, 'number of frequency points in the scan'),
            Parameter('RF_pi_pulse_time', 50.0, float, 'time duration of an RF pi pulse (in ns)')
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_time', 4000, float, '[ns] tau time between two pi/2 pulses'),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo',['spin_echo','XY4','XY8','CPMG'], 'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int, 'number of pulse pulses.')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 1200000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        self.data['norm_echo'] = None
        self.data['norm_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_XYn_RFfreqsw, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']

        # no fitting for RF frequency scan
        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
        #     self.data['fits_echo'] = fits
        # except:
        #     self.data['fits_echo'] = None
        #     self.log('ECHO t2 fit failed')
        #
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
        #     self.data['fits_deer'] = fits
        # except:
        #     self.data['fits_deer'] = None
        #     self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        RF_freq_start = self.settings['RF_pulses']['RF_frequency'] - self.settings['RF_pulses']['RF_freq_range']/2.0
        RF_freq_stop = self.settings['RF_pulses']['RF_frequency'] + self.settings['RF_pulses']['RF_freq_range']/2.0
        RF_freq_list = np.linspace(RF_freq_start, RF_freq_stop, num = self.settings['RF_pulses']['RF_freq_points']).tolist()
        print('RF_freq_list:', RF_freq_list)
        # tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        # tau_list = [x for x in tau_list if x == 0 or x >= 15]


        if self.settings['mw_pulses']['microwave_channel'] == 'i':
            microwave_channel_1 = 'microwave_i'
            microwave_channel_2 = 'microwave_q'
        else:
            microwave_channel_1 = 'microwave_q'
            microwave_channel_2 = 'microwave_i'

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
        RF_pi_time = self.settings['RF_pulses']['RF_pi_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']
        number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
        tau_total = self.settings['tau_time']

        # if self.settings['decoupling_seq']['type'] == 'spin_echo':
        #     for RF_freq_current in RF_freq_list:
        #
        #         tau = tau_total/2
        #         #ECHO SEQUENCE:
        #         pulse_sequence = \
        #         [
        #             Pulse(microwave_channel_1, laser_off_time, pi_half_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, laser_off_time + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE = laser_off_time + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2 * tau, three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2 * tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         #DEER SEQUENCE
        #
        #         start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_DEER , pi_half_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_DEER + pi_half_time + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_DEER + pi_half_time + 2 * tau , pi_half_time)
        #         ]
        #
        #         end_of_first_HE =  start_of_DEER + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #              Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #              Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #              ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #         [
        #             Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time/2., pi_time),
        #             Pulse('RF_switch', start_of_second_HE + pi_half_time  + tau - RF_pi_time / 2., RF_pi_time),
        #             Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2*tau , three_pi_half_time)
        #         ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2*tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         pulse_sequences.append(pulse_sequence)
        #
        #     print('number of sequences before validation ', len(pulse_sequences))
        #     return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        if self.settings['decoupling_seq']['type'] == 'spin_echo':
            for RF_freq_current in RF_freq_list:

                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY4':
            for RF_freq_current in RF_freq_list:

                tau = tau_total/ (4*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau/2 #for the first pulse, only wait tau/2
                for i in range(0,number_of_pulse_blocks):
                    pulse_sequence.extend([Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time/2, pi_time),
                                           Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time/2, pi_time)
                                           ])
                    section_begin_time += 4 * tau

                #the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau/2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau/2 + pi_half_time + delay_mw_readout, nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau/2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau/2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])


                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                                 nv_reset_time),
                                       Pulse('apd_readout', section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout, meas_time)
                                      ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY8':
            for RF_freq_current in RF_freq_list:

                tau = tau_total/ (8*self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'CPMG':
            for RF_freq_current in RF_freq_list:

                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time) ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time


    def _run_sweep(self, pulse_sequences, num_loops_sweep, num_daq_reads, tau_list):
        '''
        Each pulse sequence specified in pulse_sequences is run num_loops_sweep consecutive times.

        Args:
            pulse_sequences: a list of pulse sequences to run, each corresponding to a different value of tau. Each
                             sequence is a list of Pulse objects specifying a given pulse sequence
            num_loops_sweep: number of times to repeat each sequence before moving on to the next one
            num_daq_reads: number of times the daq must read for each sequence (generally 1, 2, or 3)

        Poststate: self.data['counts'] and self.data['shot_noise'] are updated with the acquired data

        '''

        # randomize the indexes of the pulse sequences to run, to reduce heating. ER 5/25/2017
        rand_indexes = []
        for i in range(0, len(pulse_sequences)):
            rand_indexes.append(i)
        if self.settings['randomize']:
            random.shuffle(rand_indexes)
        for index, sequence in enumerate(pulse_sequences):
            # print('current index', index)
            rand_index = rand_indexes[index]
            # print('current rand_index', rand_index)
            self.instruments['RF_gen']['instance'].update({'frequency': tau_list[rand_index]})
            # print('current RF frequency [Hz]', tau_list[rand_index])
            if self._abort:
                break

            result = self._single_sequence(pulse_sequences[rand_index], num_loops_sweep,
                                           num_daq_reads)  # keep entire array

            counts_temp = 0
            if not result:
                print('Throwing away results of sequence {}'.format(rand_index))
                self.log('Throwing away results of sequence {}'.format(rand_index))
                self.nfails[rand_index] += 1
            else:
                self.count_data[rand_index] = self.count_data[rand_index] + result
                counts_to_check = self._normalize_to_kCounts(np.array(result), self.measurement_gate_width,
                                                             num_loops_sweep)
                self.data['counts'][rand_index] = self._normalize_to_kCounts(self.count_data[rand_index],
                                                                             self.measurement_gate_width,
                                                                             self.current_averages - self.nfails[
                                                                                 rand_index] * MAX_AVERAGES_PER_SCAN)
                self.data['shot_noise'][rand_index] = np.reciprocal(np.sqrt(self.count_data[rand_index]))
                counts_temp = counts_to_check[0]

            self.sequence_index = rand_index

            # throw error if tracking is on and you haven't ran find nv ER 6/2/17
            if self.settings['Tracking']['on/off']:
                if self.scripts['find_nv'].data['fluorescence']:
                    self.data['init_fluor'] = deepcopy(self.scripts['find_nv'].data['fluorescence'])
                else:
                    raise AttributeError('need to run find NV first for tracking!')

            # track to the NV if necessary ER 5/31/17
            if (self.settings['Tracking']['on/off']):
                if (self.settings['Tracking']['threshold'] * self.data['init_fluor'] > counts_temp or
                                (2 - self.settings['Tracking']['threshold']) * self.data['init_fluor'] < counts_temp):
                    #      self._plot_refresh = True
                    print('TRACKING TO NV...')
                    self.scripts['find_nv'].run()
                    #     self._plot_refresh = True
                    self.scripts['find_nv'].settings['initial_point'] = self.scripts['find_nv'].data['maximum_point']
            self.updateProgress.emit(self._calc_progress(index))

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
            tau = data['tau']

        if data['norm_echo'] is not None and data['norm_deer'] is not None:

            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            # axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            # axislist[0].hold(True)
            # axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])
            #
            # tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            # axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            # axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            # axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
            # axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)


            # super(DEER_XYn, self)._plot(axislist)
            # axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            axislist[0].legend(labels=('Echo', 'DEER'), fontsize=8)

            axislist[0].set_title(
                'DEER (RF frequency scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                    self.settings['tau_time'],
                    self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))
            axislist[0].set_xlabel('RF frequency [Hz]')

        else:
            super(DEER_XYn_RFfreqsw, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            axislist[0].set_title('DEER (RF frequency scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['tau_time'], self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_xlabel('RF frequency [Hz]')
    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER_XYn_RFfreqsw, self)._update_plot(axislist)

            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_title('DEER (RF frequency scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['tau_time'], self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'], self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].set_xlabel('RF frequency [Hz]')

class DEER_XYn_RFpitimesw(PulseBlasterBaseScript):  # ER 5.25.2017

    """
        This script runs a DEER sequence on the NV scanning over RF pi pulse time.
        There are different decoupling sequence options.
        To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.

        ==> last edited by Ziwei Qiu 8/25/2017
    """

    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', 3.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.6424e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for pi/2 pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -7.0, float, 'RF center power in dBm'),
            Parameter('RF_frequency', 224e6, float, 'RF frequency in Hz'),
            Parameter('RF_pi_pulse_time', 50.0, float, 'time duration of an RF pi pulse (in ns)'),
            Parameter('RF_pi_time_range', 40.0, float, 'RF pi pulse range in ns'),
            Parameter('RF_pi_time_points', 21, int, 'number of RF pi pulse points in the scan')
            # Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            # Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_time', 4000, float, '[ns] tau time between two pi/2 pulses'),
        Parameter('decoupling_seq', [
            Parameter('type', 'spin_echo', ['spin_echo', 'XY4', 'XY8', 'CPMG'],
                      'type of dynamical decoupling sequences'),
            Parameter('num_of_pulse_blocks', 1, int,
                      'number of pulse blocks.')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int,
                      '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 1200000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator,
                    'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        # COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        self.data['norm_echo'] = None
        self.data['norm_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_XYn_RFpitimesw, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2. * (- self.data['counts'][:, 1] + self.data['counts'][:, 0]) / (
        self.data['counts'][:, 1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2. * (- self.data['counts'][:, 3] + self.data['counts'][:, 2]) / (
        self.data['counts'][:, 3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2 * (self.data['counts'][:, 1] * self.data['counts'][:, 0]) / np.square(
            self.data['counts'][:, 1] + self.data['counts'][:, 0]) * np.sqrt(
            np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2 * (self.data['counts'][:, 3] * self.data['counts'][:, 2]) / np.square(
            self.data['counts'][:, 3] + self.data['counts'][:, 2]) * np.sqrt(
            np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']

        # no fitting for RF pi time scan
        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
        #     self.data['fits_echo'] = fits
        # except:
        #     self.data['fits_echo'] = None
        #     self.log('ECHO t2 fit failed')
        #
        # try:
        #     fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
        #     self.data['fits_deer'] = fits
        # except:
        #     self.data['fits_deer'] = None
        #     self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []
        RF_pi_time_start = self.settings['RF_pulses']['RF_pi_pulse_time'] - self.settings['RF_pulses']['RF_pi_time_range'] / 2.0
        RF_pi_time_stop = self.settings['RF_pulses']['RF_pi_pulse_time'] + self.settings['RF_pulses']['RF_pi_time_range'] / 2.0
        RF_pi_time_list = np.linspace(RF_pi_time_start, RF_pi_time_stop, num=self.settings['RF_pulses']['RF_pi_time_points']).tolist()
        print('RF_pi_time_list:', RF_pi_time_list)
        # tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        # tau_list = [x for x in tau_list if x == 0 or x >= 15]


        if self.settings['mw_pulses']['microwave_channel'] == 'i':
            microwave_channel_1 = 'microwave_i'
            microwave_channel_2 = 'microwave_q'
        else:
            microwave_channel_1 = 'microwave_q'
            microwave_channel_2 = 'microwave_i'

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
        RF_pi_time = self.settings['RF_pulses']['RF_pi_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']
        number_of_pulse_blocks = self.settings['decoupling_seq']['num_of_pulse_blocks']
        tau_total = self.settings['tau_time']

        # if self.settings['decoupling_seq']['type'] == 'spin_echo':
        #     for RF_pi_time_current in RF_pi_time_list:
        #         RF_pi_time = RF_pi_time_current
        #         tau = tau_total / 2
        #         # ECHO SEQUENCE:
        #         pulse_sequence = \
        #             [
        #                 Pulse(microwave_channel_1, laser_off_time, pi_half_time),
        #                 Pulse(microwave_channel_1, laser_off_time + pi_half_time + tau - pi_time / 2., pi_time),
        #                 Pulse(microwave_channel_1, laser_off_time + pi_half_time + 2 * tau, pi_half_time)
        #             ]
        #
        #         end_of_first_HE = laser_off_time + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #         ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #             [
        #                 Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #                 Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time / 2., pi_time),
        #                 Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2 * tau, three_pi_half_time)
        #             ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2 * tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         # DEER SEQUENCE
        #
        #         start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #         pulse_sequence += \
        #             [
        #                 Pulse(microwave_channel_1, start_of_DEER, pi_half_time),
        #                 Pulse(microwave_channel_1, start_of_DEER + pi_half_time + tau - pi_time / 2., pi_time),
        #                 Pulse('RF_switch', start_of_DEER + pi_half_time + tau - RF_pi_time / 2., RF_pi_time),
        #                 Pulse(microwave_channel_1, start_of_DEER + pi_half_time + 2 * tau, pi_half_time)
        #             ]
        #
        #         end_of_first_HE = start_of_DEER + pi_half_time + 2 * tau + pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
        #         ]
        #
        #         start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
        #
        #         pulse_sequence += \
        #             [
        #                 Pulse(microwave_channel_1, start_of_second_HE, pi_half_time),
        #                 Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + tau - pi_time / 2., pi_time),
        #                 Pulse('RF_switch', start_of_second_HE + pi_half_time + tau - RF_pi_time / 2., RF_pi_time),
        #                 Pulse(microwave_channel_1, start_of_second_HE + pi_half_time + 2 * tau, three_pi_half_time)
        #             ]
        #
        #         end_of_second_HE = start_of_second_HE + pi_half_time + 2 * tau + three_pi_half_time
        #
        #         pulse_sequence += [
        #             Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
        #             Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
        #         ]
        #
        #         pulse_sequences.append(pulse_sequence)
        #
        #     print('number of sequences before validation ', len(pulse_sequences))
        #     return pulse_sequences, self.settings['num_averages'], RF_pi_time_list, meas_time

        if self.settings['decoupling_seq']['type'] == 'spin_echo':
            for RF_pi_time_current in RF_pi_time_list:
                RF_pi_time = RF_pi_time_current
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_1, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY4':
            for RF_pi_time_current in RF_pi_time_list:
                RF_pi_time = RF_pi_time_current
                tau = tau_total / (4 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 4 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'XY8':
            for RF_pi_time_current in RF_pi_time_list:
                RF_pi_time = RF_pi_time_current
                tau = tau_total / (8 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 2 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 2 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 3 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 3 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 4 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 4 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 5 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 5 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 6 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 6 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_1, section_begin_time + 7 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 7 * tau - RF_pi_time / 2, RF_pi_time),
                         Pulse(microwave_channel_2, section_begin_time + 8 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 8 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 8 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

        elif self.settings['decoupling_seq']['type'] == 'CPMG':
            for RF_pi_time_current in RF_pi_time_list:
                RF_pi_time = RF_pi_time_current
                tau = tau_total / (1 * self.settings['decoupling_seq']['num_of_pulse_blocks'])
                pulse_sequence = []
                # ECHO SEQUENCE:

                start_of_first_HE = laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_first_HE, pi_half_time)])
                section_begin_time = start_of_first_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_HE = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_HE, pi_half_time)])
                section_begin_time = start_of_second_HE + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time)])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                # DEER SEQUENCE

                start_of_DEER = section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                # the first pi/2 pulse
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_DEER, pi_half_time)])
                section_begin_time = start_of_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2
                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                start_of_second_DEER = section_begin_time + tau / 2 + pi_half_time + delay_mw_readout + nv_reset_time + laser_off_time
                pulse_sequence.extend([Pulse(microwave_channel_1, start_of_second_DEER, pi_half_time)])
                section_begin_time = start_of_second_DEER + pi_half_time - tau / 2  # for the first pulse, only wait tau/2

                for i in range(0, number_of_pulse_blocks):
                    pulse_sequence.extend(
                        [Pulse(microwave_channel_2, section_begin_time + 1 * tau - pi_time / 2, pi_time),
                         Pulse('RF_switch', section_begin_time + 1 * tau - RF_pi_time / 2, RF_pi_time)
                         ])
                    section_begin_time += 1 * tau

                # the second 3*pi/2 pulse and readout
                pulse_sequence.extend([Pulse(microwave_channel_1, section_begin_time + tau / 2, three_pi_half_time),
                                       Pulse('laser',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout,
                                             nv_reset_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 + three_pi_half_time + delay_mw_readout + delay_readout,
                                             meas_time)
                                       ])

                pulse_sequences.append(pulse_sequence)

            print('number of sequences before validation ', len(pulse_sequences))
            return pulse_sequences, self.settings['num_averages'], RF_freq_list, meas_time

    def _plot(self, axislist, data=None):
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
            tau = data['tau']
            print ('here we are')

        if data['norm_echo'] is not None and data['norm_deer'] is not None:

            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            # axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            # axislist[0].hold(True)
            # axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])
            #
            # tauinterp = np.linspace(np.min(tau), np.max(tau), 100)
            # axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]), 'b:')
            # axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            # axislist[0].set_title(
            #     'T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],
            #                                                                                             fits_deer[1]))
            # axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            axislist[0].legend(labels=('Echo', 'DEER'), fontsize=8)
            axislist[0].set_title(
                'DEER (RF pi pulse scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                    self.settings['tau_time'],
                    self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))

            # super(DEER_XYn, self)._plot(axislist)
            # axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            # axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

        else:
            super(DEER_XYn_RFpitimesw, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])),
                                       'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])),
                                       'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])),
                                       'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)
            axislist[0].set_title(
                'DEER (RF pi time scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                    self.settings['tau_time'],
                    self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                    self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                    self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))


    def _update_plot(self, axislist):
        # self._plot(axislist)
        if len(axislist[0].lines) == 0:
            self._plot(axislist)
            return
        super(DEER_XYn_RFpitimesw, self)._update_plot(axislist)

        axislist[0].legend(labels=('Echo +up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 1])),
                                   'Echo down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 0])),
                                   'DEER up {:.0f}kcps'.format(np.mean(self.data['counts'][:, 3])),
                                   'DEER down {:.0f}kcps'.format(np.mean(self.data['counts'][:, 2]))), fontsize=8)
        # axislist[0].set_title('DEER \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
        axislist[0].set_title(
            'DEER (RF pi time scan, tau = {:.2f} ns) {:s} {:d} block(s) \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(
                self.settings['tau_time'],
                self.settings['decoupling_seq']['type'], self.settings['decoupling_seq']['num_of_pulse_blocks'],
                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9,
                self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency'] * 1e-6))

class DEER_T1ref(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a Hahn echo on the NV to find the Hahn echo T2.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -45.0, float, 'microwave power in dB'),
            Parameter('RF_frequency', 250e6, float, 'microwave frequency in Hz')
            #Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 5, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 300, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 1000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 1000, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 30, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 10, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        self.data['fits_T1'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_T1ref, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])
        self.data['norm_T1'] = 2. * (- self.data['counts'][:, 5] + self.data['counts'][:, 4]) / (self.data['counts'][:, 5] + self.data['counts'][:, 4])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))
        self.data['T1_err'] = 2*(self.data['counts'][:,5]*self.data['counts'][:, 4])/np.square(self.data['counts'][:,5] + self.data['counts'][:, 4]) * np.sqrt(np.square(self.data['shot_noise'][:, 4]) + np.square(self.data['shot_noise'][:, 5]))

        tau = self.data['tau']
        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
            self.data['fits_deer'] = fits
        except:
            self.data['fits_deer'] = None
            self.log('DEER t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_T1'], offset=True, verbose=True)
            self.data['fits_T1'] = fits
        except:
            self.data['fits_T1'] = None
            self.log('T1 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']


        for tau in tau_list:
            #ECHO SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            #DEER SEQUENCE

            start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            # T1 REFERENCES:

            start_of_first_ref = end_of_second_HE + delay_mw_readout + nv_reset_time
            # ref1:
            pulse_sequence += [
                 Pulse('laser', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                 ]
            end_of_first_ref = start_of_first_ref + laser_off_time + tau + tau + pi_half_time  + delay_mw_readout + nv_reset_time

            # ref2:
            start_of_second_ref = end_of_first_ref + laser_off_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_ref + pi_half_time/2. + tau - pi_time/2., pi_time)
            ]

            end_of_flip = start_of_second_ref + pi_half_time + tau + tau

            pulse_sequence += \
            [
                Pulse('laser', end_of_flip + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_flip + delay_mw_readout + delay_readout, meas_time)
            ]
            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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
            tau = data['tau']

        if data['fits_echo'] is not None and data['fits_deer'] is not None:
            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']
            fits_T1 = data['fits_T1']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])
            axislist[0].errorbar(tau, data['norm_T1'], data['T1_err'])

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_T1[0], fits_T1[1], fits_T1[2]), 'r:')

            axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
            axislist[0].legend(labels=('Echo', 'DEER', 'T1', 'exp fit: echo', 'exp fit: deer', 'exp fit: T1'), fontsize=8)
        else:
            super(DEER_T1ref, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2])), 'REF up {:.0f}kcps'.format(np.mean(data['counts'][:, 4])), 'REF down {:.0f}kcps'.format(np.mean(data['counts'][:, 5]))), fontsize=8)

            axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER_T1ref, self)._update_plot(axislist)

# #this doesn't work
# class DEER_RFpitime(PulseBlasterBaseScript):
#     """
#     This script runs a DEER sequence scanning over RF pi duration
#     ==> last edited by Ziwei Qiu 8/17/2017
#         """
#     _DEFAULT_SETTINGS = [
#         Parameter('mw_pulses', [
#             Parameter('mw_power', -45.0, float, 'microwave power in dB'),
#             Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
#             Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
#             Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
#             Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
#             Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
#         ]),
#         Parameter('RF_pulses', [
#             Parameter('RF_power', -45.0, float, 'microwave power in dB'),
#             Parameter('RF_frequency', 250e6, float, 'microwave frequency in Hz')
#         ]),
#         # Parameter('tau_times', [
#         #     Parameter('min_time', 500, float, 'minimum time between pi pulses'),
#         #     Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
#         #     Parameter('time_step', 5, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
#         #               'time step increment of time between pi pulses (in ns)')
#         # ]),
#         Parameter('tau', 2000,float,'time between MW pi pulses (in ns)'),
#         Parameter('RF_pi_times', [
#             Parameter('min_time', 50, float, 'minimum RF pi pulse duration (in ns)'),
#             Parameter('max_time', 150, float, 'maximum RF pi pulse duration (in ns)'),
#             Parameter('time_step', 10, [2.5, 5, 10, 20, 50, 100],
#                       'time step increment(in ns)')
#         ]),
#         Parameter('read_out', [
#             Parameter('meas_time', 300, float, '[ns] APD window to count  photons during readout'),
#             Parameter('nv_reset_time', 1000, int, '[ns] time for optical polarization - typ. 1000 '),
#             Parameter('laser_off_time', 1000, int,
#                       '[ns] minimum laser off time before taking measurements'),
#             Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
#             Parameter('delay_readout', 30, int,
#                       '[ns] delay between laser on and readout (given by spontaneous decay rate)')
#         ]),
#         Parameter('num_averages', 100000, int, 'number of averages'),
#         Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
#         Parameter('mw_switch_extra_time', 10, [0, 10, 20, 30, 40],
#                   '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
#     ]
#
#     _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator,
#                     'RF_gen': R8SMicrowaveGenerator}
#
#     def _function(self):
#         # COMMENT_ME
#
#         self.data['fits_echo'] = None
#         self.data['fits_deer'] = None
#
#         ### MW generator amplitude and frequency settings:
#         self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         ### MW generator modulation settings:
#         self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
#         self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
#         self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
#         ### Turn on MW generator:
#         self.instruments['mw_gen']['instance'].update({'enable_output': True})
#
#         ### RF generator amplitude and frequency settings:
#         self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
#         self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
#         ### RF generator modulation settings:
#         self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
#         self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
#         ### Turn on RF generator:
#         self.instruments['RF_gen']['instance'].update({'enable_output': True})
#
#         ### Turn off green light (the pulse blaster will pulse it on when needed)
#         self.instruments['PB']['instance'].update({'laser': {'status': False}})
#
#         super(DEER_RFpitime, self)._function(self.data)
#
#         ### Turn off green, RF and MW at the end of DEER
#         self.instruments['PB']['instance'].update({'laser': {'status': False}})
#         self.instruments['RF_gen']['instance'].update({'enable_output': False})
#         self.instruments['mw_gen']['instance'].update({'enable_output': False})
#
#         self.data['norm_echo'] = 2. * (- self.data['counts'][:, 1] + self.data['counts'][:, 0]) / (
#         self.data['counts'][:, 1] + self.data['counts'][:, 0])
#         self.data['norm_deer'] = 2. * (- self.data['counts'][:, 3] + self.data['counts'][:, 2]) / (
#         self.data['counts'][:, 3] + self.data['counts'][:, 2])
#
#         # error propagation starting with shot noise for each trace:
#         self.data['echo_err'] = 2 * (self.data['counts'][:, 1] * self.data['counts'][:, 0]) / np.square(
#             self.data['counts'][:, 1] + self.data['counts'][:, 0]) * np.sqrt(
#             np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
#         self.data['deer_err'] = 2 * (self.data['counts'][:, 3] * self.data['counts'][:, 2]) / np.square(
#             self.data['counts'][:, 3] + self.data['counts'][:, 2]) * np.sqrt(
#             np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))
#
#         tau = self.data['tau']
#         # try:
#         #     fits = fit_exp_decay(tau, self.data['norm_echo'], offset=True, verbose=True)
#         #     self.data['fits_echo'] = fits
#         # except:
#         #     self.data['fits_echo'] = None
#         #     self.log('ECHO t2 fit failed')
#         #
#         # try:
#         #     fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
#         #     self.data['fits_deer'] = fits
#         # except:
#         #     self.data['fits_deer'] = None
#         #     self.log('DEER t2 fit failed')
#
#     def _create_pulse_sequences(self):
#         '''
#
#         Returns: pulse_sequences, num_averages, tau_list, meas_time
#             pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
#             scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
#             sequence must have the same number of daq read pulses
#             num_averages: the number of times to repeat each pulse sequence
#             tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
#             meas_time: the width (in ns) of the daq measurement
#
#         '''
#         pulse_sequences = []
#
#         # tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),
#         #                  self.settings['tau_times']['time_step'])
#         RF_pi_time_list = range(int(self.settings['RF_pi_times']['min_time']), int(self.settings['RF_pi_times']['max_time']),
#                          self.settings['RF_pi_times']['time_step'])
#
#         # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
#         # tau_list = [x for x in tau_list if x == 0 or x >= 15]
#         RF_pi_time_list = [x for x in RF_pi_time_list if x == 0 or x >= 15]
#         tau_list = RF_pi_time_list
#         print('RF_pi_time_list', RF_pi_time_list)
#
#         microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']
#
#         meas_time = self.settings['read_out']['meas_time']
#         nv_reset_time = self.settings['read_out']['nv_reset_time']
#         delay_readout = self.settings['read_out']['delay_readout']
#         laser_off_time = self.settings['read_out']['laser_off_time']
#         delay_mw_readout = self.settings['read_out']['delay_mw_readout']
#
#         pi_time = self.settings['mw_pulses']['pi_pulse_time']
#         pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
#         three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']
#
#         mw_sw_buffer = self.settings['mw_switch_extra_time']
#         tau_fixed = self.settings['tau']
#
#         # rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
#         # rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
#         # rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']
#
#
#         # for tau in tau_list:
#         for RF_pi_time in RF_pi_time_list:
#             # ECHO SEQUENCE:
#             pulse_sequence = \
#                 [
#                     Pulse(microwave_channel, laser_off_time, pi_half_time),
#                     Pulse(microwave_channel, laser_off_time + pi_half_time / 2. + tau_fixed - pi_time / 2., pi_time),
#                     Pulse(microwave_channel, laser_off_time + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2.,
#                           pi_half_time)
#                 ]
#
#             end_of_first_HE = laser_off_time + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2. + pi_half_time
#
#             pulse_sequence += [
#                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
#                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
#             ]
#
#             start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
#
#             pulse_sequence += \
#                 [
#                     Pulse(microwave_channel, start_of_second_HE, pi_half_time),
#                     Pulse(microwave_channel, start_of_second_HE + pi_half_time / 2. + tau_fixed - pi_time / 2., pi_time),
#                     Pulse(microwave_channel, start_of_second_HE + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2.,
#                           three_pi_half_time)
#                 ]
#
#             end_of_second_HE = start_of_second_HE + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2. + three_pi_half_time
#
#             pulse_sequence += [
#                 Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
#                 Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
#             ]
#
#             # DEER SEQUENCE
#
#             start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
#             pulse_sequence += \
#                 [
#                     Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
#                     Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time / 2. + tau_fixed - pi_time / 2.,
#                           pi_time),
#                     Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau_fixed - RF_pi_time / 2.,
#                           RF_pi_time),
#                     Pulse(microwave_channel,
#                           start_of_DEER + laser_off_time + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2.,
#                           pi_half_time)
#                 ]
#
#             end_of_first_HE = start_of_DEER + laser_off_time + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2. + pi_half_time
#
#             pulse_sequence += [
#                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
#                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
#             ]
#
#             start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time
#
#             pulse_sequence += \
#                 [
#                     Pulse(microwave_channel, start_of_second_HE, pi_half_time),
#                     Pulse(microwave_channel, start_of_second_HE + pi_half_time / 2. + tau_fixed - pi_time / 2., pi_time),
#                     Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau_fixed - RF_pi_time / 2., RF_pi_time),
#                     Pulse(microwave_channel, start_of_second_HE + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2.,
#                           three_pi_half_time)
#                 ]
#
#             end_of_second_HE = start_of_second_HE + pi_half_time / 2. + tau_fixed + tau_fixed - pi_half_time / 2. + three_pi_half_time
#
#             pulse_sequence += [
#                 Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
#                 Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
#             ]
#
#             pulse_sequences.append(pulse_sequence)
#
#         print('number of sequences before validation ', len(pulse_sequences))
#         return pulse_sequences, self.settings['num_averages'], RF_pi_time_list, meas_time
#
#     def _plot(self, axislist, data = None):
#         '''
#         Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
#         received for each time
#         Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
#         performed
#
#         Args:
#             axes_list: list of axes to write plots to (uses first 2)
#             data (optional) dataset to plot (dictionary that contains keys counts, tau, fits), if not provided use self.data
#         '''
#
#         if data is None:
#             data = self.data
#             tau = data['tau']
#
#         super(DEER_RFpitime, self)._plot(axislist)
#         axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2])), 'REF up {:.0f}kcps'.format(np.mean(data['counts'][:, 4])), 'REF down {:.0f}kcps'.format(np.mean(data['counts'][:, 5]))), fontsize=8)
#         axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
#
#
#         # if data['fits_echo'] is not None and data['fits_deer'] is not None:
#         #     fits_echo = data['fits_echo']
#         #     fits_deer = data['fits_deer']
#         #     fits_T1 = data['fits_T1']
#         #
#         #     # axislist[0].plot(tau, data['norm_echo'], 'b')
#         #     # axislist[0].hold(True)
#         #     # axislist[0].plot(tau, data['norm_deer'], 'r')
#         #
#         #     axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
#         #     axislist[0].hold(True)
#         #     axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])
#         #     axislist[0].errorbar(tau, data['norm_T1'], data['T1_err'])
#         #
#         #     tauinterp = np.linspace(np.min(tau),np.max(tau),100)
#         #     axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
#         #     axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
#         #     axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_T1[0], fits_T1[1], fits_T1[2]), 'r:')
#         #
#         #     axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
#         #     axislist[0].legend(labels=('Echo', 'DEER', 'T1', 'exp fit: echo', 'exp fit: deer', 'exp fit: T1'), fontsize=8)
#         # else:
#         #     super(DEER_RFpitime, self)._plot(axislist)
#         #     # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
#         #     # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
#         #     # axislist[0].hold(False)
#         #     # axislist[0].plot(tau, norm_echo, 'b')
#         #     # axislist[0].hold(True)
#         #     # axislist[0].plot(tau, norm_deer, 'r')
#         #     # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)
#         #
#         #     # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
#         #     # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
#         #     # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
#         #     # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
#         #     # axislist[0].hold(False)
#         #     # axislist[0].plot(tau, echo_up, 'b')
#         #     # axislist[0].hold(True)
#         #     # axislist[0].plot(tau, echo_down, 'k')
#         #     # axislist[0].plot(tau, deer_up, 'r')
#         #     # axislist[0].plot(tau, deer_down, 'm')
#         #     axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2])), 'REF up {:.0f}kcps'.format(np.mean(data['counts'][:, 4])), 'REF down {:.0f}kcps'.format(np.mean(data['counts'][:, 5]))), fontsize=8)
#         #
#         #     axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
#
#     def _update_plot(self, axislist):
#             # self._plot(axislist)
#             if len(axislist[0].lines) == 0:
#                 self._plot(axislist)
#                 return
#             super(DEER_RFpitime, self)._update_plot(axislist) #this one doesn't work # this doesn't work

class DEER_RF_pitime(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a DEER sequence on the NV scanning over RF pi duration while fixing tau.
==> last edited by Ziwei Qiu 8/18/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -3.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 100.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 50.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 150.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -15.0, float, 'microwave power in dB'),
            Parameter('RF_frequency', 224e6, float, 'microwave frequency in Hz')
            #Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau', 2000, float, 'time between MW pi pulses (in ns)'),
        Parameter('RF_pi_times', [
            Parameter('min_time', 50, float, 'minimum RF pi pulse duration (in ns)'),
            Parameter('max_time', 150, float, 'maximum RF pi pulse duration (in ns)'),
            Parameter('time_step', 10, [1, 2.5, 5, 10, 20, 50, 100],
                      'time step increment(in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEER_RF_pitime, self)._function(self.data)

        ### Turn off green, RF and MW at the end of DEER
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['deer_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))

        tau = self.data['tau']
        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
            self.data['fits_deer'] = fits
        except:
            self.data['fits_deer'] = None
            self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        RF_pi_time_list = range(int(self.settings['RF_pi_times']['min_time']), int(self.settings['RF_pi_times']['max_time']),self.settings['RF_pi_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        RF_pi_time_list = [x for x in RF_pi_time_list if x == 0 or x >= 15]
        print('RF_pi_time_list', RF_pi_time_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']

        tau = self.settings['tau']

        for RF_pi_time in RF_pi_time_list:
            #ECHO SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            #DEER SEQUENCE

            start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - RF_pi_time / 2., RF_pi_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau - RF_pi_time / 2., RF_pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], RF_pi_time_list, meas_time



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
            tau = data['tau']

        if data['fits_echo'] is not None and data['fits_deer'] is not None:
            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            # axislist[0].plot(tau, data['norm_echo'], 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, data['norm_deer'], 'r')

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_deer'], data['deer_err'])

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'g:')
            axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
            axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)
        else:
            super(DEER_RF_pitime, self)._plot(axislist)
            # norm_echo = 2. * (- data['counts'][:, 1] + data['counts'][:, 0]) / (data['counts'][:, 1] + data['counts'][:, 0])
            # norm_deer = 2. * (- data['counts'][:, 3] + data['counts'][:, 2]) / (data['counts'][:, 3] + data['counts'][:, 2])
            # axislist[0].hold(False)
            # axislist[0].plot(tau, norm_echo, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, norm_deer, 'r')
            # axislist[0].legend(labels=('Echo {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'DEER {:.0f}kcps'.format(np.mean(data['counts'][:, 3]))), fontsize=8)

            # echo_up = data['counts'][:, 1]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # echo_down = data['counts'][:, 0]/((data['counts'][:, 0] + data['counts'][:, 1])/2.)-1.
            # deer_up = data['counts'][:, 3]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # deer_down = data['counts'][:, 2]/((data['counts'][:, 2] + data['counts'][:, 3])/2.)-1.
            # axislist[0].hold(False)
            # axislist[0].plot(tau, echo_up, 'b')
            # axislist[0].hold(True)
            # axislist[0].plot(tau, echo_down, 'k')
            # axislist[0].plot(tau, deer_up, 'r')
            # axislist[0].plot(tau, deer_down, 'm')
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'DEER up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'DEER down {:.0f}kcps'.format(np.mean(data['counts'][:, 2]))), fontsize=8)

            axislist[0].set_title('DEER mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz, rf-power:{:.0f}dBm, rf_freq:{:.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(DEER_RF_pitime, self)._update_plot(axislist)


class T2echo_T2star_T1(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a Hahn echo on the NV to find the Hahn echo T2.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 5, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 300, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 1000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 1000, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 30, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 10, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None
        self.data['fits_T1'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### Turn off RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': False})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(T2echo_T2star_T1, self)._function(self.data)

        ### Turn off green, RF and MW at the end of sequence
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_T2star'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])
        self.data['norm_T1'] = 2. * (- self.data['counts'][:, 5] + self.data['counts'][:, 4]) / (self.data['counts'][:, 5] + self.data['counts'][:, 4])

        # error propagation starting with shot noise for each trace:
        self.data['echo_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['T2star_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))
        self.data['T1_err'] = 2*(self.data['counts'][:,5]*self.data['counts'][:, 4])/np.square(self.data['counts'][:,5] + self.data['counts'][:, 4]) * np.sqrt(np.square(self.data['shot_noise'][:, 4]) + np.square(self.data['shot_noise'][:, 5]))

        tau = self.data['tau']
        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_T2star'], offset=True, verbose=True)
            self.data['fits_T2star'] = fits
        except:
            self.data['fits_T2star'] = None
            self.log('T2star fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_T1'], offset=True, verbose=True)
            self.data['fits_T1'] = fits
        except:
            self.data['fits_T1'] = None
            self.log('T1 fit failed')

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']


        for tau in tau_list:
            #ECHO SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            #DEER SEQUENCE

            start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
                # Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                # Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                # Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                # Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            # T1 REFERENCES:

            start_of_first_ref = end_of_second_HE + delay_mw_readout + nv_reset_time
            # ref1:
            pulse_sequence += [
                 Pulse('laser', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                 ]
            end_of_first_ref = start_of_first_ref + laser_off_time + tau + tau + pi_half_time  + delay_mw_readout + nv_reset_time

            # ref2:
            start_of_second_ref = end_of_first_ref + laser_off_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_ref + pi_half_time/2. + tau - pi_time/2., pi_time)
            ]

            end_of_flip = start_of_second_ref + pi_half_time + tau + tau

            pulse_sequence += \
            [
                Pulse('laser', end_of_flip + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_flip + delay_mw_readout + delay_readout, meas_time)
            ]
            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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
            tau = data['tau']

        if data['fits_echo'] is not None and data['fits_T2star'] is not None:
            fits_echo = data['fits_echo']
            fits_T2star = data['fits_T2star']
            fits_T1 = data['fits_T1']

            axislist[0].errorbar(tau, data['norm_echo'], data['echo_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['norm_T2star'], data['T2star_err'])
            axislist[0].errorbar(tau, data['norm_T1'], data['T1_err'])

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_T2star[0], fits_T2star[1], fits_T2star[2]), 'g:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_T1[0], fits_T1[1], fits_T1[2]), 'r:')

            axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, T2star = {:2.1f} ns'.format(fits_echo[1],fits_T2star[1]))
            axislist[0].legend(labels=('Echo', 'T2star', 'T1', 'exp fit: echo', 'exp fit: T2star', 'exp fit: T1'), fontsize=8)
        else:
            super(T2echo_T2star_T1, self)._plot(axislist)
            axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'T2star up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'T2star down {:.0f}kcps'.format(np.mean(data['counts'][:, 2])), 'REF up {:.0f}kcps'.format(np.mean(data['counts'][:, 4])), 'REF down {:.0f}kcps'.format(np.mean(data['counts'][:, 5]))), fontsize=8)
            axislist[0].set_title('T2star_T2echo_T1 mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9))

    def _update_plot(self, axislist):
            # self._plot(axislist)
            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(T2echo_T2star_T1, self)._update_plot(axislist)

class IQ_calibration(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script calibrates the MW amplitude and phase between I and Q.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Ziwei Qiu on 08/24/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -3.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 120.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 60.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 180.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 500, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 500, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['XYX'] = None
        self.data['YXY'] = None


        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### Turn off RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': False})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(IQ_calibration, self)._function(self.data)

        ### Turn off green, RF and MW at the end of sequence
        self.instruments['PB']['instance'].update({'laser': {'status': False}})
        self.instruments['RF_gen']['instance'].update({'enable_output': False})
        self.instruments['mw_gen']['instance'].update({'enable_output': False})
        self.data['XYX'] = 2.*(self.data['counts'][:, 1] - self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['XXX'] = 2. * (self.data['counts'][:, 2] - self.data['counts'][:, 3]) / (self.data['counts'][:, 2] + self.data['counts'][:, 3])
        self.data['YXY'] = 2.*(self.data['counts'][:, 5] - self.data['counts'][:,4]) / (self.data['counts'][:,5] + self.data['counts'][:, 4])
        self.data['YYY'] = 2. * (self.data['counts'][:, 6] - self.data['counts'][:, 7]) / (self.data['counts'][:, 6] + self.data['counts'][:, 7])

        # error propagation starting with shot noise for each trace:
        self.data['XYX_err'] = 2*(self.data['counts'][:,1]*self.data['counts'][:, 0])/np.square(self.data['counts'][:,1] + self.data['counts'][:, 0])*np.sqrt(np.square(self.data['shot_noise'][:, 0]) + np.square(self.data['shot_noise'][:, 1]))
        self.data['XXX_err'] = 2*(self.data['counts'][:,3]*self.data['counts'][:, 2])/np.square(self.data['counts'][:,3] + self.data['counts'][:, 2])*np.sqrt(np.square(self.data['shot_noise'][:, 2]) + np.square(self.data['shot_noise'][:, 3]))
        self.data['YXY_err'] = 2*(self.data['counts'][:,5]*self.data['counts'][:, 4])/np.square(self.data['counts'][:,5] + self.data['counts'][:, 4]) * np.sqrt(np.square(self.data['shot_noise'][:, 4]) + np.square(self.data['shot_noise'][:, 5]))
        self.data['YYY_err'] = 2 * (self.data['counts'][:, 7] * self.data['counts'][:, 6]) / np.square(self.data['counts'][:, 7] + self.data['counts'][:, 6]) * np.sqrt(np.square(self.data['shot_noise'][:, 7]) + np.square(self.data['shot_noise'][:, 6]))

        tau = self.data['tau']

    def _create_pulse_sequences(self):
        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)


        if self.settings['mw_pulses']['microwave_channel'] == 'i':
            microwave_channel = 'microwave_i'
            microwave_channel_2 = 'microwave_q'
        else:
            microwave_channel = 'microwave_q'
            microwave_channel_2 = 'microwave_i'


        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']


        for tau_total in tau_list:
            # tau = tau_total / 2.0
            tau = tau_total
            pulse_sequence = []
            # Two XYX:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel_2, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_XYX = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_XYX + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_XYX + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_XYX = end_of_first_XYX + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_XYX, pi_half_time),
                Pulse(microwave_channel_2, start_of_second_XYX + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_XYX + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_XYX = start_of_second_XYX + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_XYX + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_XYX + delay_mw_readout + delay_readout, meas_time)
            ]

            #Two XXX:

            start_of_first_XXX = end_of_second_XYX + delay_mw_readout + nv_reset_time + laser_off_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_first_XXX, pi_half_time),
                Pulse(microwave_channel, start_of_first_XXX + pi_half_time/2. + tau  - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_first_XXX + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_XXX =  start_of_first_XXX + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_XXX + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_XXX + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_XXX = end_of_first_XXX + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
                [
                    Pulse(microwave_channel, start_of_second_XXX, pi_half_time),
                    Pulse(microwave_channel, start_of_second_XXX + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                    Pulse(microwave_channel, start_of_second_XXX + pi_half_time / 2. + tau + tau - pi_half_time / 2.,
                          three_pi_half_time)
                ]


            end_of_second_XXX = start_of_second_XXX + pi_half_time / 2. + tau + tau - pi_half_time / 2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_XXX + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_XXX + delay_mw_readout + delay_readout, meas_time)
            ]

            # Two YXY:
            start_of_first_YXY = end_of_second_XXX + delay_mw_readout + nv_reset_time + laser_off_time
            pulse_sequence += \
                [
                    Pulse(microwave_channel_2, start_of_first_YXY, pi_half_time),
                    Pulse(microwave_channel, start_of_first_YXY + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                    Pulse(microwave_channel_2, start_of_first_YXY + pi_half_time / 2. + tau + tau - pi_half_time / 2.,
                          pi_half_time)
                ]

            end_of_first_YXY = start_of_first_YXY + pi_half_time / 2. + tau + tau - pi_half_time / 2. + pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_first_YXY + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_first_YXY + delay_mw_readout + delay_readout, meas_time),
            ]

            start_of_second_YXY = end_of_first_YXY + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
                [
                    Pulse(microwave_channel_2, start_of_second_YXY, pi_half_time),
                    Pulse(microwave_channel, start_of_second_YXY + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                    Pulse(microwave_channel_2, start_of_second_YXY + pi_half_time / 2. + tau + tau - pi_half_time / 2.,
                          three_pi_half_time)
                ]

            end_of_second_YXY = start_of_second_YXY + pi_half_time / 2. + tau + tau - pi_half_time / 2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_YXY + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_YXY + delay_mw_readout + delay_readout, meas_time)
            ]

            # Two YYY:

            start_of_first_YYY = end_of_second_YXY + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
                [
                    Pulse(microwave_channel_2, start_of_first_YYY, pi_half_time),
                    Pulse(microwave_channel_2, start_of_first_YYY + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                    Pulse(microwave_channel_2, start_of_first_YYY + pi_half_time / 2. + tau + tau - pi_half_time / 2.,
                          pi_half_time)
                ]

            end_of_first_YYY = start_of_first_YYY + pi_half_time / 2. + tau + tau - pi_half_time / 2. + pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_first_YYY + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_first_YYY + delay_mw_readout + delay_readout, meas_time),
            ]

            start_of_second_YYY = end_of_first_YYY + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
                [
                    Pulse(microwave_channel_2, start_of_second_YYY, pi_half_time),
                    Pulse(microwave_channel_2, start_of_second_YYY + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                    Pulse(microwave_channel_2, start_of_second_YYY + pi_half_time / 2. + tau + tau - pi_half_time / 2.,
                          three_pi_half_time)
                ]

            end_of_second_YYY = start_of_second_YYY + pi_half_time / 2. + tau + tau - pi_half_time / 2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_YYY + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_YYY + delay_mw_readout + delay_readout, meas_time)
            ]

            # Append this singlge sequence to the total sequnce
            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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
            tau = data['tau']

        if data['XYX'] is not None and data['YXY'] is not None and data['XXX'] is not None and data['YYY'] is not None:
            print('plot normalized data')
            axislist[0].errorbar(tau, data['XYX'], data['XYX_err'])
            axislist[0].hold(True)
            axislist[0].errorbar(tau, data['XXX'], data['XXX_err'])
            axislist[0].errorbar(tau, data['YXY'], data['YXY_err'])
            axislist[0].errorbar(tau, data['YYY'], data['YYY_err'])

            axislist[0].set_title('IQ_calibration \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz'.format(
                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9))
            axislist[0].legend(labels=('XYX', 'XXX', 'YXY', 'YYY'), fontsize=8)
        else:
            print('plot raw data')
            super(IQ_calibration, self)._plot(axislist)
            # axislist[0].legend(labels=('Echo up {:.0f}kcps'.format(np.mean(data['counts'][:, 1])), 'Echo down {:.0f}kcps'.format(np.mean(data['counts'][:, 0])), 'T2star up {:.0f}kcps'.format(np.mean(data['counts'][:, 3])), 'T2star down {:.0f}kcps'.format(np.mean(data['counts'][:, 2])), 'REF up {:.0f}kcps'.format(np.mean(data['counts'][:, 4])), 'REF down {:.0f}kcps'.format(np.mean(data['counts'][:, 5]))), fontsize=8)
            axislist[0].legend(labels=('XYX_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 0])),
                                       'XYX_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 1])),
                                       'XXX_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 2])),
                                       'XXX_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 3])),
                                       'YXY_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 4])),
                                       'YXY_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 5])),
                                       'YYY_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 6])),
                                       'YYY_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 7]))), fontsize=8)
            axislist[0].set_title('IQ_calibration \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9))
            super(IQ_calibration, self)._plot(axislist)
            # axislist[0].legend(labels=('XYX_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 0])),
            #                            'XYX_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 1])),
            #                            'XX_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 2])),
            #                            'XX_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 3])),
            #                            'YXY_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 4])),
            #                            'YXY_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 5])),
            #                            'YY_pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 6])),
            #                            'YY_3pi/2 {:.0f}kcps'.format(np.mean(data['counts'][:, 7]))), fontsize=8)
            # axislist[0].set_title('IQ_calibration \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz'.format(
            #     self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9))

    def _update_plot(self, axislist):
            # self._plot(axislist)

            if len(axislist[0].lines) == 0:
                self._plot(axislist)
                return
            super(IQ_calibration, self)._update_plot(axislist)

            axislist[0].legend(labels=('XYX_pi/2', 'XYX_3pi/2', 'XXX_pi/2 ', 'XXX_3pi/2 ', 'YXY_pi/2 ', 'YXY_3pi/2 ', 'YYY_pi/2 ', 'YYY_3pi/2 '), fontsize = 8)

            axislist[0].set_title('IQ_calibration \n mw-power:{:.0f}dBm, mw_freq:{:.3f} GHz'.format(
                self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency'] * 1e-9))

class DEERmodified(PulseBlasterBaseScript): # ER 5.25.2017
    """
This script runs a Hahn echo on the NV to find the Hahn echo T2.
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 06/29/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('RF_pulses', [
            Parameter('RF_power', -45.0, float, 'microwave power in dB'),
            Parameter('RF_frequency', 250e6, float, 'microwave frequency in Hz')
            #Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)'),
            #Parameter('pi_half_pulse_time', 25.0, float, 'time duration of a pi/2 pulse (in ns)'),
            #Parameter('3pi_half_pulse_time', 75.0, float, 'time duration of a 3pi/2 pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 10000, float, 'maximum time between pi pulses'),
            Parameter('time_step', 5, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 300, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 1000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 1000, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 30, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 10, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fits_echo'] = None
        self.data['fits_deer'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### RF generator amplitude and frequency settings:
        self.instruments['RF_gen']['instance'].update({'power': self.settings['RF_pulses']['RF_power']})
        self.instruments['RF_gen']['instance'].update({'frequency': self.settings['RF_pulses']['RF_frequency']})
        ### RF generator modulation settings:
        self.instruments['RF_gen']['instance'].update({'freq_mode': 'CW'})
        self.instruments['RF_gen']['instance'].update({'power_mode': 'CW'})
        ### Turn on RF generator:
        self.instruments['RF_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(DEERmodified, self)._function(self.data)

        ### Turn off green light:
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        self.data['norm_echo'] = 2.*(- self.data['counts'][:, 1] + self.data['counts'][:,0]) / (self.data['counts'][:,1] + self.data['counts'][:, 0])
        self.data['norm_deer'] = 2.*(- self.data['counts'][:, 3] + self.data['counts'][:,2]) / (self.data['counts'][:,3] + self.data['counts'][:, 2])
        tau = self.data['tau']

        try:
            fits = fit_exp_decay(tau, self.data['norm_echo'], offset = True, verbose = True)
            self.data['fits_echo'] = fits
        except:
            self.data['fits_echo'] = None
            self.log('ECHO t2 fit failed')

        try:
            fits = fit_exp_decay(tau, self.data['norm_deer'], offset=True, verbose=True)
            self.data['fits_deer'] = fits
        except:
            self.data['fits_deer'] = None
            self.log('DEER t2 fit failed')

    def _create_pulse_sequences(self):
        '''
        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement
        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']

        #rf_pi_time = self.settings['RF_pulses']['pi_pulse_time']
        #rf_pi_half_time = self.settings['RF_pulses']['pi_half_pulse_time']
        #rf_three_pi_half_time = self.settings['RF_pulses']['3pi_half_pulse_time']


        for tau in tau_list:
            #ECHO SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_half_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE = laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            #DEER SEQUENCE

            start_of_DEER = end_of_second_HE + delay_mw_readout + nv_reset_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_DEER + laser_off_time, pi_half_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_DEER + laser_off_time + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2., pi_half_time)
            ]

            end_of_first_HE =  start_of_DEER + laser_off_time + pi_half_time/2. + tau + tau - pi_half_time/2. + pi_half_time

            pulse_sequence += [
                 Pulse('laser', end_of_first_HE + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', end_of_first_HE + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_HE = end_of_first_HE + delay_mw_readout + nv_reset_time + laser_off_time

            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_HE, pi_half_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau - pi_time/2., pi_time),
                Pulse('RF_switch', start_of_second_HE + pi_half_time / 2. + tau - pi_time / 2., pi_time),
                Pulse(microwave_channel, start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2., three_pi_half_time)
            ]

            end_of_second_HE = start_of_second_HE + pi_half_time/2. + tau + tau - pi_half_time/2. + three_pi_half_time

            pulse_sequence += [
                Pulse('laser', end_of_second_HE + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_second_HE + delay_mw_readout + delay_readout, meas_time)
            ]

            # T1 REFERENCES:

            start_of_first_ref = end_of_second_HE + delay_mw_readout + nv_reset_time
            # ref1:
            pulse_sequence += [
                 Pulse('laser', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', start_of_first_ref + laser_off_time + tau + tau + pi_half_time + delay_mw_readout + delay_readout, meas_time)
                 ]
            end_of_first_ref = start_of_first_ref + laser_off_time + tau + tau + pi_half_time  + delay_mw_readout + nv_reset_time

            # ref2:
            start_of_second_ref = end_of_first_ref + laser_off_time
            pulse_sequence += \
            [
                Pulse(microwave_channel, start_of_second_ref + pi_half_time/2. + tau - pi_time/2., pi_time)
            ]

            end_of_flip = start_of_second_ref + pi_half_time + tau + tau

            pulse_sequence += \
            [
                Pulse('laser', end_of_flip + delay_mw_readout, nv_reset_time),
                Pulse('apd_readout', end_of_flip + delay_mw_readout + delay_readout, meas_time)
            ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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

        if data['fits_echo'] is not None and data['fits_deer'] is not None:
            tau = data['tau']
            fits_echo = data['fits_echo']
            fits_deer = data['fits_deer']

            axislist[0].plot(tau, self.data['norm_echo'], 'b')
            axislist[0].hold(True)
            axislist[0].plot(tau, self.data['norm_deer'], 'r')

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_echo[0], fits_echo[1], fits_echo[2]),'b:')
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fits_deer[0], fits_deer[1], fits_deer[2]), 'r:')
            axislist[0].set_title('T2 decay times (simple exponential, p = 1): echo={:2.1f} ns, deer = {:2.1f} ns'.format(fits_echo[1],fits_deer[1]))
            axislist[0].legend(labels=('Echo', 'DEER', 'exp fit: echo', 'exp fit: deer'), fontsize=8)
        else:
            super(DEERmodified, self)._plot(axislist)
            axislist[0].set_title('DEER mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz, rf-power:{:0.1f}dBm, rf_freq:{:0.3f} MHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9,self.settings['RF_pulses']['RF_power'], self.settings['RF_pulses']['RF_frequency']*1e-6))
            axislist[0].legend(labels=('Echo up', 'Echo down', 'DEER up', 'DEER down', 'Ref1', 'Ref2'), fontsize=8)


class T1balanced(PulseBlasterBaseScript):
    """
This script runs a balanced T1 measurement (state 0 and state 1 population versus time after initialization).
To symmetrize the sequence between the 0 and +/-1 state we reinitialize every time.
==> last edited by Alexei Bylinskii on 08/01/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('pi_pulse_time', 50.0, float, 'time duration of a pi pulse (in ns)')
        ]),
        Parameter('tau_times', [
            Parameter('min_time', 500, float, 'minimum time between pi pulses'),
            Parameter('max_time', 5500, float, 'maximum time between pi pulses'),
            Parameter('time_step', 500, [2.5, 5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000, 500000],
                  'time step increment of time between pi pulses (in ns)')
        ]),
        Parameter('read_out', [
            Parameter('meas_time', 100, float, '[ns] APD window to count  photons during readout'),
            Parameter('nv_reset_time', 2000, int, '[ns] time for optical polarization - typ. 1000 '),
            Parameter('laser_off_time', 500, int,
                      '[ns] minimum laser off time before taking measurements'),
            Parameter('delay_mw_readout', 100, int, '[ns] delay between mw and readout'),
            Parameter('delay_readout', 100, int, '[ns] delay between laser on and readout (given by spontaneous decay rate)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('mw_switch_extra_time', 20, [0, 10, 20, 30, 40],
                  '[ns] buffer time of the MW switch window on both sides of MW_i or MW_q pulses')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator, 'RF_gen': R8SMicrowaveGenerator}

    def _function(self):
        #COMMENT_ME

        self.data['fit_T1'] = None

        ### MW generator amplitude and frequency settings:
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        ### MW generator modulation settings:
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'modulation_function': 'External'})
        self.instruments['mw_gen']['instance'].update({'enable_modulation': True})
        ### Turn on MW generator:
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        ### Turn off green light (the pulse blaster will pulse it on when needed)
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        super(T1balanced, self)._function(self.data)

        ### Turn off green light:
        self.instruments['PB']['instance'].update({'laser': {'status': False}})

        self.data['norm_T1'] = 2.*(self.data['counts'][:, 1] - self.data['counts'][:, 0])/(self.data['counts'][:, 1] + self.data['counts'][:, 0])
        tau = self.data['tau']

        try:
            fits = fit_exp_decay(tau, self.data['norm_T1'], offset = True, verbose = True)
            self.data['fit_T1'] = fits
        except:
            self.data['fit_T1'] = None
            self.log('T1 fit failed')

    def _create_pulse_sequences(self):
        '''
        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement
        '''
        pulse_sequences = []

        tau_list = range(int(self.settings['tau_times']['min_time']), int(self.settings['tau_times']['max_time']),self.settings['tau_times']['time_step'])

        # ignore the sequence if the mw-pulse is shorter than 15ns (0 is ok because there is no mw pulse!)
        tau_list = [x for x in tau_list if x == 0 or x >= 15]
        print('tau_list', tau_list)

        microwave_channel = 'microwave_' + self.settings['mw_pulses']['microwave_channel']

        meas_time = self.settings['read_out']['meas_time']
        nv_reset_time = self.settings['read_out']['nv_reset_time']
        delay_readout = self.settings['read_out']['delay_readout']
        laser_off_time = self.settings['read_out']['laser_off_time']
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        pi_time = self.settings['mw_pulses']['pi_pulse_time']

        mw_sw_buffer = self.settings['mw_switch_extra_time']


        for tau in tau_list:
            #T1 SEQUENCE:
            pulse_sequence = \
            [
                Pulse(microwave_channel, laser_off_time, pi_time)
            ]

            pulse_sequence += [
                 Pulse('laser', laser_off_time + pi_time + tau + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', laser_off_time + pi_time + tau + delay_mw_readout + delay_readout, meas_time),
                 ]

            start_of_second_ref = laser_off_time + pi_time + tau + delay_mw_readout + nv_reset_time

            pulse_sequence += [
                 Pulse('laser', start_of_second_ref + laser_off_time + pi_time + tau + delay_mw_readout, nv_reset_time),
                 Pulse('apd_readout', start_of_second_ref + laser_off_time + pi_time + tau + delay_mw_readout + delay_readout, meas_time),
                 ]

            pulse_sequences.append(pulse_sequence)

        print('number of sequences before validation ', len(pulse_sequences))
        return pulse_sequences, self.settings['num_averages'], tau_list, meas_time

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

        if data['fit_T1'] is not None:
            tau = data['tau']
            fit_T1 = data['fit_T1']

            axislist[0].plot(tau, self.data['norm_T1'], 'b')
            axislist[0].hold(True)

            tauinterp = np.linspace(np.min(tau),np.max(tau),100)
            axislist[0].plot(tauinterp, exp_offset(tauinterp, fit_T1[0], fit_T1[1], fit_T1[2]),'b:')
            axislist[0].set_title('T1 decay times (simple exponential, p = 1): {:2.1f} ns'.format(fit_T1[1]))
            axislist[0].legend(labels=('T1 data', 'T1 exp fit'), fontsize=8)
        else:
            super(T1balanced, self)._plot(axislist)
            axislist[0].set_title('T1 mw-power:{:0.1f}dBm, mw_freq:{:0.3f} GHz'.format(self.settings['mw_pulses']['mw_power'], self.settings['mw_pulses']['mw_frequency']*1e-9))
            axislist[0].legend(labels=('Down', 'Up'), fontsize=8)


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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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

    # def _calc_progress(self):
    #     #COMMENT_ME
    #     # todo: change to _calc_progress(self, index):
    #     progress = int(100. * (self._loop_count) / self.settings['freq_points'])
    #     return progress

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

# NOT FINSISHED - TESTING PHASE!!!
class ESRSingleFreqCont(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power and durations for varying frequencies.
This is the CW version, where we apply the MW only for short times but still much longer than a pi/2 pulse, ie. a few micro seconds to avoid heating of the sample.
This is different from the actual pulsed ESR, where we apply pi/2 pulses to get the max contrast.
    """


    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('frequency', 2.82e9, float, 'frequency (Hz)')
        ]),
        Parameter('read_out', [
            Parameter('integration_time', 4000, int, '1.) Time the MWs are off (us) and the measurement time. Laser is on and photons are counted during this time.'),
            Parameter('delay_mw_readout', 100, int, 'delay between laser on and readout (in ns)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
        Parameter('max_points', 100, int, 'number of points to display if 0 show all')
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        self.data = {'mw_frequencies': np.linspace(self.settings['mw_pulses']['freq_start'], self.settings['mw_pulses']['freq_stop'],
                                                   self.settings['mw_pulses']['freq_points']), 'esr_counts': []}

        self.instruments['mw_gen']['instance'].update({'frequency': float(self.settings['mw_pulses']['frequency'])})

        while self._abort is False:

            super(ESRSingleFreqCont, self)._function(self.data)

            self.data['esr_counts'].append(self.data['counts'][0])


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
        esr_counts = np.array(data['esr_counts'])

        # if there is two measurement per run, the second serves as a normalization measurement
        if len(np.shape(esr_counts))== 2:
            esr_counts  = esr_counts[:,0]/esr_counts[:,1]


        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
            # axis1.set_title('avrg count')
        axis2 = axes_list[1]
        plot_pulses(axis2, self.pulse_sequences[0])


    def _update_plot(self, axes_list):
        mw_frequencies = self.data['mw_frequencies']
        esr_counts = np.array(self.data['esr_counts'])


        # if there is two measurement per run, the second serves as a normalization measurement
        if len(np.shape(esr_counts)) == 2:
            esr_counts  = esr_counts[:,0]/esr_counts[:,1]

        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
            # axis2 = axes_list[1]
            # update_pulse_plot(axis2, self.pulse_sequences[0])

    def _create_pulse_sequences(self):

        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement(s)
        '''


        # on contrast to other script the following times are given in us and have to be converted to ns
        integration_time = self.settings['read_out']['integration_time']*1e3
        delay_mw_readout = self.settings['read_out']['delay_mw_readout']



        pulse_sequences = [[Pulse('laser', 0, 2*integration_time+2*delay_mw_readout),
                            Pulse('apd_readout', delay_mw_readout, integration_time), #read fluourescence
                            Pulse('microwave_i',2*delay_mw_readout + integration_time, integration_time),
                            Pulse('apd_readout',2*delay_mw_readout + integration_time, integration_time)
                            ]]

        tau_list = [integration_time]
        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, mw_on_time

class PulsedESRSlow(PulseBlasterBaseScript):
    """
This script applies a microwave pulse at fixed power and durations for varying frequencies.
This is the CW version, where we apply the MW only for short times but still much longer than a pi/2 pulse, ie. a few micro seconds to avoid heating of the sample.
This is different from the actual pulsed ESR, where we apply pi/2 pulses to get the max contrast.
    """


    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -45.0, float, 'microwave power in dB'),
            Parameter('microwave_channel', 'i', ['i', 'q'], 'Channel to use for mw pulses'),
            Parameter('freq_start', 2.82e9, float, 'start frequency of scan in Hz'),
            Parameter('freq_stop', 2.92e9, float, 'end frequency of scan in Hz'),
            Parameter('freq_points', 100, int, 'number of frequencies in scan in Hz'),
        ]),
        Parameter('read_out', [
            Parameter('mw_off_time', 4000, int, '1.) Time the MWs are off (us) and the measurement time. Laser is on and photons are counted during this time.'),
            Parameter('mw_on_time', 250, float, '2.) Time the MWs are on (us). If measure_ref is True Laser is on and photons are counted during time of duration mw_off_time (i.e. the measurement time)!!.'),
            Parameter('laser_off_time', 250, float, '3.) Laser is off during this time after mW_on and mw_off pulse (us). Set to zero to skip this.'),
            Parameter('measure_ref', True, bool, 'If true take reference measurement. In that case mw_on_time has to be larger than mw_off_time. If not mw_on_time is set equal to mw_off_time'),
            Parameter('delay_mw_readout', 100, int, 'delay between laser on and readout (in ns)')
        ]),
        Parameter('num_averages', 100000, int, 'number of averages'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'enable_output': True})

        assert self.settings['mw_pulses']['freq_start'] < self.settings['mw_pulses']['freq_stop']

        self.data = {'mw_frequencies': np.linspace(self.settings['mw_pulses']['freq_start'], self.settings['mw_pulses']['freq_stop'],
                                                   self.settings['mw_pulses']['freq_points']), 'esr_counts': []}

        for i, mw_frequency in enumerate(self.data['mw_frequencies']):
            self._loop_count = i
            self.instruments['mw_gen']['instance'].update({'frequency': float(mw_frequency)})
            super(PulsedESRSlow, self)._function(self.data)

            self.data['esr_counts'].append(self.data['counts'][0])

    # def _calc_progress(self):
    #     #COMMENT_ME
    #     # todo: change to _calc_progress(self, index):
    #     progress = int(100. * (self._loop_count) / self.settings['freq_points'])
    #     return progress

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
        esr_counts = np.array(data['esr_counts'])

        if len(np.shape(esr_counts))== 3:
            esr_counts  = esr_counts[:,0,0]/esr_counts[:,0,1]

        # print('sXXadsdasda', np.shape(esr_counts)[-1])
        #
        # # if there is two measurement per run, the second serves as a normalization measurement
        # if len(esr_counts.T) == 2:
        #     print('sadsdasda', np.shape(esr_counts))
        #     esr_counts = esr_counts[:,0] / esr_counts[:,1]
        #     print('ggggg', len(esr_counts))

        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
            # axis1.set_title('avrg count')
        axis2 = axes_list[1]
        plot_pulses(axis2, self.pulse_sequences[0])


    def _update_plot(self, axes_list):
        mw_frequencies = self.data['mw_frequencies']
        esr_counts = np.array(self.data['esr_counts'])


        # if there is two measurement per run, the second serves as a normalization measurement
        if len(np.shape(esr_counts))== 3:
            esr_counts  = esr_counts[:,0,0]/esr_counts[:,0,1]


        print(len(esr_counts))
        axis1 = axes_list[0]
        if not esr_counts == []:
            counts = esr_counts
            plot_esr(axis1, mw_frequencies[0:len(counts)], counts)
            axis1.hold(False)
            # axis2 = axes_list[1]
            # update_pulse_plot(axis2, self.pulse_sequences[0])

    def _create_pulse_sequences(self):

        '''

        Returns: pulse_sequences, num_averages, tau_list, meas_time
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement(s)
        '''


        # on contrast to other script the following times are given in us and have to be converted to ns
        mw_on_time = self.settings['read_out']['mw_on_time']*1e3
        mw_off_time = self.settings['read_out']['mw_off_time']*1e3
        laser_off_time = self.settings['read_out']['laser_off_time']*1e3

        delay_mw_readout = self.settings['read_out']['delay_mw_readout']

        measure_ref = self.settings['read_out']['measure_ref']

        # minimum pulse length is 15ns, if less set to zero, i.e. don't turn laser off
        if laser_off_time <= 15:
            laser_off_time = 0
        # enforce minimum pulse length (15ns)
        if delay_mw_readout <=15:
            delay_mw_readout = 15

        # if measuring the reference fluorescence duing the MW off time enforce that the MW are off at least as long as it takes to take the measurement
        if (measure_ref is True) and (mw_off_time < mw_on_time):
            mw_off_time = mw_on_time

        # todo: JG - this is a quick fix and should be handled by pulse_blaster_script.validate, which adds a pulse for the microwave switch, this has a hard coded delay of 40ns /
        # since here the mw pulse is the first pulse this results in negative times
        mw_offset_time = 40


        if laser_off_time == 0:
            if (measure_ref is True):
                pulse_sequences = [[Pulse('laser', mw_offset_time+ 0, mw_on_time+mw_off_time+2*delay_mw_readout),
                                    Pulse('microwave_i', mw_offset_time+ 0, mw_on_time+delay_mw_readout),
                                    Pulse('apd_readout', mw_offset_time+ delay_mw_readout, mw_on_time),
                                    Pulse('apd_readout', mw_offset_time+ 2*delay_mw_readout+mw_on_time, mw_on_time), # the readout is actually on mw_on_time long but the mw are mw_off_time off
                                    Pulse('off_channel', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + mw_off_time, 15)
                                    # Pulse('laser', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + mw_off_time,15) # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    ]]
            else:
                pulse_sequences = [[Pulse('laser', mw_offset_time+ 0, mw_on_time+2*delay_mw_readout),
                                    Pulse('microwave_i', mw_offset_time+ 0, mw_on_time+delay_mw_readout),
                                    Pulse('apd_readout', mw_offset_time+ delay_mw_readout, mw_on_time),
                                    Pulse('off_channel', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + mw_off_time, 15)
                                    # Pulse('laser', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + mw_off_time,15) # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    ]]
        else:
            if (measure_ref is True):
                pulse_sequences = [[Pulse('laser', mw_offset_time+ 0, mw_on_time+delay_mw_readout),
                                    Pulse('microwave_i',mw_offset_time+  0, mw_on_time+delay_mw_readout),
                                    Pulse('apd_readout',mw_offset_time+  delay_mw_readout, mw_on_time),
                                    Pulse('laser', mw_offset_time+ mw_on_time + delay_mw_readout + laser_off_time, mw_off_time+delay_mw_readout),
                                    Pulse('apd_readout', mw_offset_time+ 2*delay_mw_readout+mw_on_time+ laser_off_time, mw_on_time), # the readout is actually on mw_on_time long but the mw are mw_off_time off
                                    Pulse('off_channel', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + 2 * laser_off_time + mw_off_time,15)
                                    # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    # Pulse('laser', 2*delay_mw_readout+mw_on_time+ 2*laser_off_time +mw_off_time, 15) # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    ]]
            else:
                pulse_sequences = [[Pulse('laser', mw_offset_time+ 0, mw_on_time+delay_mw_readout),
                                    Pulse('microwave_i',mw_offset_time+  0, mw_on_time+delay_mw_readout),
                                    Pulse('apd_readout',mw_offset_time+  delay_mw_readout, mw_on_time),
                                    Pulse('off_channel', mw_offset_time+ 2 * delay_mw_readout + mw_on_time + 2 * laser_off_time + mw_off_time,15)
                                    # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    # Pulse('laser', 2*delay_mw_readout+mw_on_time+ 2*laser_off_time +mw_off_time, 15) # at the end we want mw and laser to be off, so we add this short laser pulse because 'off' doesn't exist
                                    ]]


        tau_list = [mw_on_time]
        # end_time_max = 0
        # for pulse_sequence in pulse_sequences:
        #     for pulse in pulse_sequence:
        #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
        # for pulse_sequence in pulse_sequences:
        #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))

        return pulse_sequences, self.settings['num_averages'], tau_list, mw_on_time





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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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
            pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15)) # Jan Feb 1st 2017: what is 1850??? Need to comment!

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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_frequency']})
        super(Pulsed_ESR_Pulsed_Laser, self)._function()

    def _create_pulse_sequences(self):
        """
        Returns:
            pulse_sequences, num_averages, tau_list
            pulse_sequences: a list of pulse sequences, each corresponding to a different time 'tau' that is to be
            scanned over. Each pulse sequence is a list of pulse objects containing the desired pulses. Each pulse
            sequence must have the same number of daq read pulses
            num_averages: the number of times to repeat each pulse sequence
            tau_list: the list of times tau, with each value corresponding to a pulse sequence in pulse_sequences
            meas_time: the width (in ns) of the daq measurement

        """

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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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


# class XY8(PulseBlasterBaseScript):
#     """
# This script runs a CPMG pulse sequence.
#     """
#     _DEFAULT_SETTINGS = [
#         Parameter('mw_pulses',[
#             Parameter('mw_power', -45.0, float, 'microwave power in dB'),
#             Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
#             # Parameter('mw_switch_extra_time', 15, int, 'Time to add before and after microwave switch is turned on'),
#             Parameter('pi_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
#             Parameter('number_of_pulse_blocks', 1, range(1, 17), 'number of alternating x-y-x-y-y-x-y-x pulses'),
#             Parameter('end_in_0', False, bool, 'end with 3pi/2 pulse so end state is |0> rather than |1>')
#         ]),
#         Parameter('tau_times',[
#             Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000],
#                       'time step increment of time between pulses (in ns)'),
#             Parameter('min_time', 100, float, 'minimum time between pulses (in ns)'),
#             Parameter('max_time', 1000, float, 'maximum time between pulses (in ns)'),
#         ]),
#         Parameter('read_out',[
#             Parameter('delay_mw_init', 1000, int, 'delay between initialization and mw (in ns)'),
#             Parameter('delay_mw_readout', 200, int, 'delay between mw and readout (in ns)'),
#             Parameter('meas_time', 250, float, 'measurement time after CPMG sequence (in ns)'),
#             Parameter('nv_reset_time', 3000, int, 'time with laser on at the beginning to reset state'),
#             Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)')
#         ]),
#         Parameter('num_averages', 1000, int, 'number of averages (should be less than a million)'),
#         Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
#     ]
#
#     _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}
#     _SCRIPTS = {}
#
#     def _function(self):
#         self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
#         self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
#         self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
#         super(XY8, self)._function()
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
#         # tau_list = range(int(max(15, self.settings['min_delay_time'])), int(self.settings['max_delay_time'] + 15),
#         #                  self.settings['delay_time_step'])
#
#         # JG: changed the previous because the 15ns is taken care of later
#         tau_list = range(int(self.settings['tau_times']['min_time']),
#                          int(self.settings['tau_times']['max_time']),
#                          self.settings['tau_times']['time_step']
#                          )
#
#         reset_time = self.settings['read_out']['nv_reset_time']
#         pi_time = self.settings['mw_pulses']['pi_pulse_time']
#         pi_half_time = pi_time/2.0
#
#         ref_meas_off_time = self.settings['read_out']['ref_meas_off_time']
#         meas_time = self.settings['read_out']['meas_time']
#         delay_mw_init = self.settings['read_out']['delay_mw_init']
#         delay_mw_readout = self.settings['read_out']['delay_mw_readout']
#
#         number_of_pulse_blocks = self.settings['mw_pulses']['number_of_pulse_blocks']
#
#
#         for tau in tau_list:
#
#             pulse_sequence = []
#
#             #initialize and pi/2 pulse
#             pulse_sequence.extend([Pulse('laser', 0, reset_time - ref_meas_off_time - 15 - meas_time),
#                                    Pulse('apd_readout', reset_time - 15 - meas_time, meas_time),
#                                    Pulse('laser', reset_time - 15 - meas_time, meas_time),
#                                    Pulse('microwave_i', reset_time + delay_mw_init-pi_half_time/2, pi_half_time)
#                                    ])
#
#             #CPMG xyxyyxyx loops added number_of_pulse_blocks times
#             section_begin_time = reset_time + delay_mw_init - tau/2 #for the first pulse, only wait tau/2
#             # JG 16-08-19 - begin changed to pi time instead of pi/2
#             # section_begin_time = reset_time + delay_mw_init + pi_time
#             # JG 16-08-19 - end
#
#             # for i in range(0, number_of_pulse_blocks):
#             #     pulse_sequence.extend([Pulse('microwave_i', section_begin_time + 1*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_q', section_begin_time + 2*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_i', section_begin_time + 3*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_q', section_begin_time + 4*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_q', section_begin_time + 5*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_i', section_begin_time + 6*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_q', section_begin_time + 7*tau - pi_half_time, pi_time),
#             #                            Pulse('microwave_i', section_begin_time + 8*tau - pi_half_time, pi_time)
#             #                           ])
#             #     section_begin_time += 8*tau
#
#             # AK 17-02-28 - switched to yx rather than xy since we saw echo was better with rephasing pulses
#             #               perpendicular to pi/2 pulses
#             for i in range(0, number_of_pulse_blocks):
#                 pulse_sequence.extend([Pulse('microwave_q', section_begin_time + 1*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_i', section_begin_time + 2*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_q', section_begin_time + 3*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_i', section_begin_time + 4*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_i', section_begin_time + 5*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_q', section_begin_time + 6*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_i', section_begin_time + 7*tau - pi_half_time, pi_time),
#                                        Pulse('microwave_q', section_begin_time + 8*tau - pi_half_time, pi_time)
#                                       ])
#                 section_begin_time += 8*tau
#
#
#             if self.settings['mw_pulses']['end_in_0']:
#                 # 3pi/2 and readout
#                 pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau / 2 - 3*pi_half_time/4, 3*pi_half_time),
#                                        Pulse('laser', section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
#                                              meas_time),
#                                        Pulse('apd_readout',
#                                              section_begin_time + tau / 2 + pi_half_time + delay_mw_readout,
#                                              meas_time)])
#             else:
#                 #pi/2 and readout
#                 pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau/2 - pi_half_time/2, pi_half_time),
#                                        Pulse('laser',       section_begin_time + tau/2 + pi_half_time + delay_mw_readout, meas_time),
#                                        Pulse('apd_readout', section_begin_time + tau/2 + pi_half_time + delay_mw_readout, meas_time)])
#
#             # JG 16-08-19 - begin changed to pi time instead of pi/2
#             # pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau, pi_half_time),
#             #                        Pulse('laser',       section_begin_time + tau + pi_half_time + delay_mw_readout, meas_time),
#             #                        Pulse('apd_readout', section_begin_time + tau + pi_half_time + delay_mw_readout, meas_time)])
#             # JG 16-08-19 - end
#
#
#             pulse_sequences.append(pulse_sequence)
#
#         # end_time_max = 0
#         # for pulse_sequence in pulse_sequences:
#         #     for pulse in pulse_sequence:
#         #         end_time_max = max(end_time_max, pulse.start_time + pulse.duration)
#         # for pulse_sequence in pulse_sequences:
#         #     pulse_sequence.append(Pulse('laser', end_time_max + 1850, 15))
#
#         return pulse_sequences, self.settings['num_averages'], tau_list, meas_time
#
#
#     def _plot(self, axislist, data = None):
#         """
#         Plot 1: self.data['tau'], the list of times specified for a given experiment, verses self.data['counts'], the data
#         received for each time
#         Plot 2: the pulse sequence performed at the current time (or if plotted statically, the last pulse sequence
#         performed
#
#         Args:
#             axes_list: list of axes to write plots to (uses first 2)
#             data (optional) dataset to plot (dictionary that contains keys counts, tau), if not provided use self.data
#         """
#
#         super(XY8, self)._plot(axislist, data)
#         axislist[0].set_title('XY8')
#         axislist[0].legend(labels=('Ref Fluorescence', 'XY8 data'), fontsize=8)

class XY8(PulseBlasterBaseScript):
    """
This script runs a CPMG pulse sequence.
modified by ZQ 8/21/2017
    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses',[
            Parameter('mw_power', -15.0, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            # Parameter('mw_switch_extra_time', 15, int, 'Time to add before and after microwave switch is turned on'),
            Parameter('pi_pulse_time', 100.0, float, 'time duration of a pi pulse (in ns)'),
            Parameter('pi_half_pulse_time', 50.0, float, 'time duration of a pi/2 pulse (in ns)'),
            Parameter('3pi_half_pulse_time', 150.0, float, 'time duration of a 3pi/2 pulse (in ns)'),
            Parameter('number_of_pulse_blocks', 1, range(1, 17), 'number of alternating x-y-x-y-y-x-y-x pulses'),
            Parameter('end_in_0', False, bool, 'end with 3pi/2 pulse so end state is |0> rather than |1>')
        ]),
        Parameter('tau_times',[
            Parameter('min_time', 100, float, 'minimum time between pulses (in ns)'),
            Parameter('max_time', 1000, float, 'maximum time between pulses (in ns)'),
            Parameter('time_step', 5, [5, 10, 20, 50, 100, 200, 500, 1000, 10000, 100000],
                      'time step increment of time between pulses (in ns)')
        ]),
        Parameter('read_out',[
            Parameter('delay_mw_init', 500, int, 'delay between initialization and mw (in ns)'),
            Parameter('delay_mw_readout', 100, int, 'delay between mw and readout (in ns)'),
            Parameter('meas_time', 500, float, 'measurement time after CPMG sequence (in ns)'),
            Parameter('nv_reset_time', 2000, int, 'time with laser on at the beginning to reset state'),
            Parameter('ref_meas_off_time', 1000, int,'laser off time before taking reference measurement at the end of init (ns)')
        ]),
        Parameter('num_averages', 1000, int, 'number of averages (should be less than a million)'),
        Parameter('skip_invalid_sequences', True, bool, 'Skips any sequences with <15ns commands'),
    ]

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}
    #_SCRIPTS = {}

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
        # pi_half_time = pi_time/2.0
        pi_half_time = self.settings['mw_pulses']['pi_half_pulse_time']
        three_pi_half_time = self.settings['mw_pulses']['3pi_half_pulse_time']


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
                                   Pulse('microwave_i', reset_time + delay_mw_init-pi_half_time/2, pi_half_time)
                                   ])

            #CPMG xyxyyxyx loops added number_of_pulse_blocks times
            section_begin_time = reset_time + delay_mw_init - tau/2 #for the first pulse, only wait tau/2
            # JG 16-08-19 - begin changed to pi time instead of pi/2
            # section_begin_time = reset_time + delay_mw_init + pi_time
            # JG 16-08-19 - end

            # for i in range(0, number_of_pulse_blocks):
            #     pulse_sequence.extend([Pulse('microwave_i', section_begin_time + 1*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_q', section_begin_time + 2*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_i', section_begin_time + 3*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_q', section_begin_time + 4*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_q', section_begin_time + 5*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_i', section_begin_time + 6*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_q', section_begin_time + 7*tau - pi_half_time, pi_time),
            #                            Pulse('microwave_i', section_begin_time + 8*tau - pi_half_time, pi_time)
            #                           ])
            #     section_begin_time += 8*tau

            # AK 17-02-28 - switched to yx rather than xy since we saw echo was better with rephasing pulses
            #               perpendicular to pi/2 pulses
            for i in range(0, number_of_pulse_blocks):
                pulse_sequence.extend([Pulse('microwave_q', section_begin_time + 1*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 2*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 3*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 4*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 5*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 6*tau - pi_half_time, pi_time),
                                       Pulse('microwave_i', section_begin_time + 7*tau - pi_half_time, pi_time),
                                       Pulse('microwave_q', section_begin_time + 8*tau - pi_half_time, pi_time)
                                      ])
                section_begin_time += 8*tau


            if self.settings['mw_pulses']['end_in_0']:
                # 3pi/2 and readout
                pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau / 2 - pi_half_time / 2, three_pi_half_time),
                                       Pulse('laser', section_begin_time + tau / 2 - pi_half_time / 2 + three_pi_half_time + delay_mw_readout,
                                             meas_time),
                                       Pulse('apd_readout',
                                             section_begin_time + tau / 2 - pi_half_time / 2 + three_pi_half_time + delay_mw_readout,
                                             meas_time)])
            else:
                #pi/2 and readout
                pulse_sequence.extend([Pulse('microwave_i', section_begin_time + tau/2 - pi_half_time/2, pi_half_time),
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

class XY4(PulseBlasterBaseScript):
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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}
    #_SCRIPTS = {}

    def _function(self):
        self.instruments['mw_gen']['instance'].update({'modulation_type': 'IQ'})
        self.instruments['mw_gen']['instance'].update({'amplitude': self.settings['mw_pulses']['mw_power']})
        self.instruments['mw_gen']['instance'].update({'frequency': self.settings['mw_pulses']['mw_frequency']})
        super(XY4, self)._function()

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
                                      ])
                section_begin_time += 4*tau

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

        super(XY4, self)._plot(axislist, data)
        axislist[0].set_title('XY4')
        axislist[0].legend(labels=('Ref Fluorescence', 'XY4 data'), fontsize=8)


class PDD(PulseBlasterBaseScript):
    """
This script runs a PDD ( Periodic Dynamical Decoupling) sequence for different number of pi pulses.
For a single pi-pulse this is a Hahn-echo sequence.
For zero pulses this is a Ramsey sequence.

The sequence is pi/2 - tau/4 - (tau/4 - pi  - tau/4)^n - tau/4 - pi/2

Tau/2 is the time between the center of the pulses!


    """
    _DEFAULT_SETTINGS = [
        Parameter('mw_pulses', [
            Parameter('mw_power', -2, float, 'microwave power in dB'),
            Parameter('mw_frequency', 2.87e9, float, 'microwave frequency in Hz'),
            Parameter('pi_pulse_time', 50, float, 'time duration of pi-pulse (in ns)'),
            Parameter('number_of_pi_pulses', 1, range(0, 17), 'number of pi pulses')
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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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
            next_pi_pulse_time = reset_time + delay_mw_init
            # # 16-08-25 JG: changed :
            # next_pi_pulse_time = reset_time + delay_mw_init - pi_half_time / 2 + tau / 2

            for n in range(1, number_of_pi_pulses + 1):
                next_pi_pulse_time += tau/2
                pulse_sequence.extend([Pulse('microwave_q', next_pi_pulse_time - pi_time/2, pi_time)])
                # next_pi_pulse_time += tau*2 + pi_time
                # 16-08-19 JG: changed:
                # next_pi_pulse_time += tau
                # 16 - 08 -24 JG: changed
                next_pi_pulse_time += tau/2

            if number_of_pi_pulses == 0:
                next_pi_pulse_time += tau

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
            # pulse_sequence.extend([Pulse('microwave_i', next_pi_pulse_time + pi_half_time, pi_half_time),
            #                        Pulse('laser', next_pi_pulse_time + pi_time + delay_mw_readout, meas_time),
            #                        Pulse('apd_readout', next_pi_pulse_time + pi_time + delay_mw_readout,
            #                              meas_time)
            #                        ])

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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster}

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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

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
                    Pulse('apd_readout',    reset_time - ref_meas_off_time - meas_time - 15 - microwave_duration, meas_time),
                    Pulse('laser',          reset_time - ref_meas_off_time - meas_time - 15 - microwave_duration, meas_time),
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

class TestingSeq(PulseBlasterBaseScript):
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

    _INSTRUMENTS = {'daq': NI6259, 'PB': CN041PulseBlaster, 'mw_gen': MicrowaveGenerator}

    def _function(self):
        #COMMENT_ME
        self.instruments['mw_gen']['instance'].update({
            'modulation_type': 'IQ',
            'amplitude': self.settings['mw-pulse']['mw_power'],
            'frequency': self.settings['mw-pulse']['mw_frequency']
        })
        super(TestingSeq, self)._function()

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
            starttime = 500.0

            if self.settings['apply mw-pulse']:
                sequence = [
                    Pulse('laser', 0,       reset_time - ref_meas_off_time - meas_time - 15 - ref_meas_off_time- microwave_duration),
                    Pulse('apd_readout',    reset_time - ref_meas_off_time - meas_time - 15 - microwave_duration, meas_time),
                    Pulse('laser',          reset_time - ref_meas_off_time - meas_time - 15 - microwave_duration, meas_time),
                    Pulse(microwave_channel,reset_time - 15 - ref_meas_off_time/2.- microwave_duration, microwave_duration)
                ]
            else:
                sequence = [
                    Pulse('laser', starttime, reset_time - ref_meas_off_time - meas_time - 15),
                    Pulse('apd_readout', starttime + reset_time - 15 - meas_time, meas_time),
                    Pulse('laser', starttime + reset_time - 15 - meas_time, meas_time)
                ]

            sequence += [
                Pulse('apd_readout', starttime + reset_time + tau, meas_time)
                # Pulse('laser',       reset_time + tau, meas_time)
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
        super(TestingSeq, self)._plot(axislist, data)
        axislist[0].set_title('T1')
        axislist[0].legend(labels=( 'Ref Fluorescence', 'T1 data'), fontsize=8)


if __name__ == '__main__':
    script = {}
    instr = {}
    script, failed, instr = Script.load_and_append({'CalibrateMeasurementWindow': 'CalibrateMeasurementWindow'}, script, instr)

    print(script)
    print('failed', failed)
    print(instr)