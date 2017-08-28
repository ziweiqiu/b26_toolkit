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

from PyLabControl.src.core import Script, Parameter
from b26_toolkit.src.scripts import ESR
from b26_toolkit.src.scripts.pulse_blaster_scripts_CN041 import Rabi, DEER
import numpy as np


class EsrRabiDeer(Script):
    """
    Does both an ESR experiment and a Rabi experiment on an NV, using the reference frequency from the esr data.
    """

    _DEFAULT_SETTINGS = [
        Parameter('DEER_spectrum', [
            Parameter('RF_center_freq', 250e6, float, 'RF carrier frequency for dark spin [Hz]'),
            Parameter('do_RF_freq_sweep', True, bool, 'check if taking a DEER spectrum by varying RF carrier frequency'),
            Parameter('RF_freq_sweep_range', 100e6, float, 'RF frequency sweep range [Hz]'),
            Parameter('RF_freq_sweep_npoints', 10, float, 'RF frequency sweep number of points'),
        ]),
        Parameter('DEER_power_sweep', [
            Parameter('RF_pwr', -45, float, 'RF pulse power for dark spin [dBm]'),
            Parameter('do_RF_pwr_sweep', True, bool, 'check if sweeping RF power'),
            Parameter('RF_pwr_sweep_range', 6, float, 'RF power sweep range [dBm]'),
            Parameter('RF_pwr_sweep_npoints', 3, float, 'RF power sweep number of points'),
        ])
    ]

    _INSTRUMENTS = {}

    _SCRIPTS = {'esr': ESR, 'rabi': Rabi, 'deer':DEER}

    def __init__(self, scripts, name = None, settings = None, log_function = None, timeout = 1000000000, data_path = None):

        Script.__init__(self, name, settings = settings, scripts = scripts, log_function= log_function, data_path = data_path)


    def _function(self):

        self.data = {'dummy': 'placeholder'}

        ####### run ESR script
        self.scripts['esr'].run()

        if self.scripts['esr'].data['fit_params'] is not None:
            if len(self.scripts['esr'].data['fit_params']) == 4:
                self.rabi_frequency = self.scripts['esr'].data['fit_params'][2]
            elif len(self.scripts['esr'].data['fit_params']) == 6:
                self.rabi_frequency = self.scripts['esr'].data['fit_params'][4]
            else:
                raise RuntimeError('Could not get fit parameters from esr script')

            centerfreq = self.scripts['esr'].settings['freq_start']
            freqrange = self.scripts['esr'].settings['freq_stop']
            if self.rabi_frequency < centerfreq-freqrange/3:
                self.log('Resonance frequency found ({:0.2e}) was below esr sweep range, aborting rabi attempt'.format(self.rabi_frequency))
            elif self.rabi_frequency > centerfreq+freqrange/3:
                self.log('Resonance frequency found ({:0.2e}) was above esr sweep range, aborting rabi attempt'.format(self.rabi_frequency))
            else:
                ####### run Rabi script
                self.log('Starting RABI with frequency {:.4e} Hz'.format(self.rabi_frequency))
                self.scripts['rabi'].settings['mw_pulses']['mw_frequency'] = float(self.rabi_frequency)
                self.scripts['rabi'].run()

                if self.scripts['rabi'].data['pi_time'] is not None and self.scripts['rabi'].data['pi_half_time'] is not None and self.scripts['rabi'].data['three_pi_half_time'] is not None:
                    self.scripts['deer'].settings['mw_pulses']['mw_frequency'] = float(self.rabi_frequency)
                    self.pi_time = self.scripts['rabi'].data['pi_time']
                    self.pi_half_time = self.scripts['rabi'].data['pi_half_time']
                    self.three_pi_half_time = self.scripts['rabi'].data['three_pi_half_time']

                    if not (self.pi_half_time>15 and self.pi_time>self.pi_half_time and self.three_pi_half_time>self.pi_time):
                        self.log('Pi/2=({:0.2e}), Pi=({:0.2e}), 3Pi/2=({:0.2e}) do not make sense, aborting DEER for this NV'.format(self.pi_half_time,self.pi_time,self.three_pi_half_time))
                    else:
                        ####### run DEER script
                        self.log('Starting DEER sweeps with Pi/2=({:0.2e}), Pi=({:0.2e}), 3Pi/2=({:0.2e})'.format(self.pi_half_time, self.pi_time, self.three_pi_half_time))
                        self.scripts['deer'].settings['mw_pulses']['pi_half_pulse_time'] = float(self.pi_half_time)
                        self.scripts['deer'].settings['mw_pulses']['pi_pulse_time'] = float(self.pi_time)
                        self.scripts['deer'].settings['mw_pulses']['3pi_half_pulse_time'] = float(self.three_pi_half_time)

                        # tag before staring deer sweeps:
                        base_tag_deer = self.scripts['deer'].settings['tag']

                        if self.settings['DEER_spectrum']['do_RF_freq_sweep']:
                            self.do_deer_freq_sweep()
                        elif self.settings['DEER_power_sweep']['do_RF_pwr_sweep']:
                            self.do_deer_pwr_sweep()
                        else:
                            self.scripts['deer'].run()

                        # return to original tag:
                        self.scripts['deer'].settings['tag'] = base_tag_deer

        else:
            self.log('No resonance frequency found skipping rabi attempt')

    def do_deer_freq_sweep(self):
        deerfldrlblb1 = self.scripts['deer'].settings['tag']
        for freq in np.linspace(self.settings['DEER_spectrum']['RF_center_freq'] - self.settings['DEER_spectrum']['RF_freq_sweep_range'] / 2,
                                self.settings['DEER_spectrum']['RF_center_freq'] + self.settings['DEER_spectrum']['RF_freq_sweep_range'] / 2,
                                self.settings['DEER_spectrum']['RF_freq_sweep_npoints']):
            self.scripts['deer'].settings['RF_pulses']['RF_frequency'] = freq.tolist()
            self.log('RF frequency set to ({:0.2e})MHz'.format(freq/1e6))
            self.scripts['deer'].settings['tag'] = deerfldrlblb1 + '_freq{:.0f}MHz'.format(freq/1e6)
            ### inner loop does power sweeps:
            if self.settings['DEER_power_sweep']['do_RF_pwr_sweep']:
                self.do_deer_pwr_sweep()
            else:
                self.scripts['deer'].run()

    def do_deer_pwr_sweep(self):
        deerfldrlblb2 = self.scripts['deer'].settings['tag']
        for pwr in np.linspace(self.settings['DEER_power_sweep']['RF_pwr'] - self.settings['DEER_power_sweep']['RF_pwr_sweep_range'] / 2,
                                self.settings['DEER_power_sweep']['RF_pwr'] + self.settings['DEER_power_sweep']['RF_pwr_sweep_range'] / 2,
                                self.settings['DEER_power_sweep']['RF_pwr_sweep_npoints']):
            self.scripts['deer'].settings['RF_pulses']['RF_power'] = pwr.tolist()
            self.log('RF power set to ({:0.2e})'.format(pwr))
            self.scripts['deer'].settings['tag'] = deerfldrlblb2 + '_pwr{:.0f}dBm'.format(pwr)
            self.scripts['deer'].run()


    def _plot(self, axes_list):
        """
        Args:
            axes_list: list of axes objects on which to plot plots the esr on the first axes object
            data: data (dictionary that contains keys image_data, extent, initial_point, maximum_point) if not provided use self.data
        """

        # if self.scripts['esr'].is_running:
        #     self.scripts['esr']._plot([axes_list[1]])
        # elif self.scripts['rabi'].is_running:
        #     self.scripts['rabi']._plot(axes_list)
        # elif self.scripts['deer'].is_running:
        #     self.scripts['deer']._plot(axes_list)

        if self._current_subscript_stage['current_subscript'] is self.scripts['esr'] and self.scripts['esr'].is_running:
            self.scripts['esr']._plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] is self.scripts['rabi'] and self.scripts['rabi'].is_running:
            self.scripts['rabi']._plot(axes_list)
        elif self.scripts['deer'].is_running:
            self.scripts['deer']._plot(axes_list)


    def _update_plot(self, axes_list):
        """
        Args:
            axes_list: list of axes objects on which to plot plots the esr on the first axes object
        """

        # if self.scripts['esr'].is_running:
        #     self.scripts['esr']._update_plot([axes_list[1]])
        # elif self.scripts['rabi'].is_running:
        #     self.scripts['rabi']._update_plot(axes_list)
        # elif self.scripts['deer'].is_running:
        #     self.scripts['deer']._update_plot(axes_list)

        if self._current_subscript_stage['current_subscript'] is self.scripts['esr'] and self.scripts['esr'].is_running:
            self.scripts['esr']._update_plot([axes_list[1]])
        elif self._current_subscript_stage['current_subscript'] is self.scripts['rabi'] and self.scripts['rabi'].is_running:
            self.scripts['rabi']._update_plot(axes_list)
        elif self.scripts['deer'].is_running:
            self.scripts['deer']._update_plot(axes_list)


    # def get_axes_layout(self, figure_list):
    #     """
    #     returns the axes objects the script needs to plot its data
    #     the default creates a single axes object on each figure
    #     This can/should be overwritten in a child script if more axes objects are needed
    #     Args:
    #         figure_list: a list of figure objects
    #     Returns:
    #         axes_list: a list of axes objects
    #
    #     """
    #
    #     # create a new figure list that contains only figure 1, this assures that the super.get_axes_layout doesn't
    #     # empty the plot contained on figure 2
    #     # return super(EsrRabiDeer, self).get_axes_layout([figure_list[0]])
    #
    #     return super(EsrRabiDeer, self).get_axes_layout(figure_list)

if __name__ == '__main__':
    script, failed, instr = Script.load_and_append({'EsrRabiDeer': EsrRabiDeer})

    print(script)
    print(failed)
    print(instr)