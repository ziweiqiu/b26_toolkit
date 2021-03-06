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

try:
    from test_script import ScriptTest
except Exception:
    pass

from galvo_scan_v2 import GalvoScan
from confocal_scan_CN041 import ConfocalScan
from set_laser import SetLaser
from set_confocal_CN041 import SetConfocal
from optimize_z_CN041 import optimizeZ
from daq_read_counter import Daq_Read_Counter
from daq_read_counter_single_CN041 import Daq_Read_Counter_Single
from esr import ESR
from esr_two_freq_continuous import ESRTwoFreqContinuous
from find_nv import FindNV
# from pulse_blaster_scripts import XY8, T1, Rabi, CalibrateMeasurementWindow, PDD, XY4, T1SpinFlip, PulsedESR,PulsedESRSlow, Rabi_double_init, HahnEcho_double_init
from pulse_blaster_scripts_CN041 import Rabi, HahnEcho, DEER, DEER_XYn, DEER_XYn_RFpwrsw, DEER_XYn_RFfreqsw, DEER_XYn_RFpitimesw, DEERmodified, GrDelayMeas, T1balanced, DEER_T1ref, T2echo_T2star_T1, IQ_calibration, TestingSeq
from esr_and_rabi import ESRAndRabi
from esr_rabi_deer import EsrRabiDeer
from esr_rabi_deerXYn import EsrRabiDeerXYn
from esr_rabi_deerXYn_spectrum import EsrRabiDeerXYnSpectrum
from pulse_delays import PulseDelays
from correlate_images import Track_Correlate_Images, Take_And_Correlate_Images
from autofocus_CN041 import AutoFocusDAQ, AutoFocusTwoPoints


# from keysight_get_spectrum import KeysightGetSpectrum
# from labview_fpga_get_timetrace import LabviewFpgaTimetrace
# from ni_fpga_polarization_controller import FPGA_BalancePolarization, FPGA_CalibrateDetector, FPGA_BalancePolarizationAndActivateFB
# from zi_sweeper import ZISweeper
# from zi_high_res_sweep import ZISweeperHighResolution
# from atto_scan import AttoStep
# from keysight_get_spectrum import KeysightGetSpectrum
# from light_control import ApplyLightControlSettings, CameraOn
# from ni_fpga_galvo_scan import FPGA_GalvoScan
# from autofocus import AutoFocusDAQ, AutoFocusTwoPoints, AutoFocusDaqSMC, AutoFocusTwoPointsFR
# from ni_fpga_galvo_scan import FPGA_GalvoScan
# from record_pressures import RecordPressures
# from set_magnetic_coils import SetMagneticCoils
# from align_magnetic_field_to_NV import AlignFieldToNV
# from Ni_9263_polarization_controller import Ni9263_BalancePolarization
# from stability_with_microwaves import Stability_With_Microwaves
#
# from ni_fpga_polarization_controller import FPGA_PolarizationSignalMap,\
#     FPGA_PolarizationSignalScan, FPGA_BalancePolarization
# from read_temperature_lakeshore import ReadTemperatureLakeshore



# old imports with try/except
# verbose = False

# ==== import NI DAQ scripts ==================================================================================
# =============================================================================================================

# try:
#     from galvo_scan import GalvoScan
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! GalvoScan did not load")
# try:
#     from autofocus import AutoFocusDAQ
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! AutoFocus did not load")
# try:
#     from set_laser import SetLaser
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! SetLaser did not load")
# try:
#     from b26_toolkit.src.scripts.daq_read_counter import Daq_Read_Counter
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! Daq_Read_Counter did not load")
# =============================================================================================================

# try:
#     from Find_Points import Find_Points
# except:
#     print("./src/scripts/__init__ warning! Find_Points did not load")

# ==== import NI FPGA scripts =================================================================================
# =============================================================================================================
# try:
#     from labview_fpga_get_timetrace import LabviewFpgaTimetrace
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! LabviewFpgaTimetrace did not load")
#
# try:
#     from FPGA_PolarizationController import FPGA_PolarizationController
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FPGA_PolarizationController did not load")
#
# try:
#     from FPGA_PolarizationController import FPGA_PolarizationSignalMap
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FPGA_PolarizationSignalMap did not load")
#
# try:
#     from FPGA_PolarizationController import FPGA_PolarizationSignalScan
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FPGA_PolarizationSignalScan did not load")
# try:
#     from FPGA_PolarizationController import FPGA_BalancePolarization
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FPGA_BalancePolarization did not load")
# try:
#     from autofocus import AutoFocusNIFPGA
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! AutoFocusNIFPGA did not load")
#
# try:
#     from galvo_scan_ni_fpga import FPGA_GalvoScan
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FPGA_GalvoScan did not load")
#
# # try:
# #     from legacy_src.old_scripts.galvo_scan_ni_fpga_loop import GalvoScanNIFPGALoop
# # except:
# #     print("./src/scripts/__init__ warning! GalvoScanNIFPGALoop did not load")
#
# # =============================================================================================================
#
# # ==== import some other scripts ==============================================================================
# # =============================================================================================================
# # try:
# #     from src.scripts.Select_NVs import Select_NVs
# # except:
# #     print("./src/scripts/__init__ warning! Select_NVs did not load")
# #
#
# # try:
# #     from Correlate_Images import Take_And_Correlate_Images
# # except:
# #     print("./src/scripts/__init__ warning! Take_And_Correlate_Images did not load")
#
# try:
#     from find_max_counts_point_2d import FindMaxCounts2D
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! FindMaxCounts2D did not load")
#
# try:
#     from atto_scan import AttoStep
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! Attostep did not load")
#
# # try:
# #     from legacy_src.old_scripts.refind_NVs import Refind_NVs
# # except:
# #     print("./src/scripts/__init__ warning! Refind_NVs did not load")
#
# # =============================================================================================================
#
# # ==== import Stanford instruments ESR scripts=================================================================
# # =============================================================================================================
# try:
#     from esr import ESR
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! ESR did not load")
#
# # try:
# #     from legacy_src.old_scripts.ESR_Selected_NVs import ESR_Selected_NVs
# # except:
# #     print("./src/scripts/__init__ warning! ESR_Selected_NVs did not load")
# # try:
# #     from legacy_src.old_scripts.ESR_and_push import ESR_And_Push
# # except:
# #     print("./src/scripts/__init__ warning! ESR_And_Push did not load")
#
# # =============================================================================================================
#
#
#
#
# # ==== import Pulse blaster scripts============================================================================
# # =============================================================================================================
# try:
#     from exec_pulse_blaster_sequence import ExecutePulseBlasterSequence
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! ExecutePulseBlasterSequence did not load")
#
# try:
#     from pulse_delays import PulseDelays
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! PulseDelays did not load")
# # try:
# #     from legacy_src.old_scripts.pulse_blaser_derived_scripts import Rabi_Power_Sweep
# # except:
# #     print("./src/scripts/__init__ warning! Rabi_Power_Sweep did not load")
# try:
#     from pulse_blaster_scripts import Rabi
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! Rabi did not load")
# try:
#     from pulse_blaster_scripts import PulsedESR
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! Pulsed_ESR did not load")
# # try:
# #     from legacy_src.old_scripts.MW_Power_Broadening import MWPowerBroadening
# # except:
# #     print("./src/scripts/__init__ warning! MWPowerBroadening did not load")
# try:
#     from pulse_blaster_scripts import CalibrateMeasurementWindow
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! CalibrateMeasurementWindow did not load")
#
# try:
#     from pulse_blaster_scripts import Rabi_Power_Sweep_Single_Tau
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! Rabi_Power_Sweep_Single_Tau did not load")
# # try:
# #     from legacy_src.old_scripts.pulse_blaser_derived_scripts import RoundPiPulseTime
# # except:
# #     print("./src/scripts/__init__ warning! RoundPiPulseTime did not load")
#
# # try:
# #     from legacy_src.old_scripts.pulse_blaser_derived_scripts import Rabi_Loop
# # except:
# #     print("./src/scripts/__init__ warning! Rabi_Loop did not load")
#
# try:
#     from pulse_blaster_scripts import T1
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! T1 did not load")
#
# try:
#     from pulse_blaster_scripts import CPMG
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! CPMG did not load")
#
#     # =============================================================================================================
#

# try:
#     from keysight_get_spectrum import KeysightGetSpectrum
# except:
#     if verbose:
#         print("./src/scripts/__init__ warning! KeysightGetSpectrum did not load")
#     # from pulse_blaser_derived_scripts import Rabi_Loop