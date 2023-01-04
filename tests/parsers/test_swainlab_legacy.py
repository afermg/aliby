#!/usr/bin/env jupyter
"""


Output of legacy logfile parser:

channels: {'channel': ['Brightfield', 'GFPFast', 'mCherry'], 'exposure': [30, 30, 100], 'skip': [1, 1, 1], 'zsect': [1, 1, 1], 'start_time': [1, 1, 1], 'camera_mode': [2, 2, 2], 'em_gain': [270, 270, 270], 'voltage': [1.0, 3.5, 2.5]}
zsectioning: {'nsections': [3], 'spacing': [0.8], 'pfson': [True], 'anyz': [True], 'drift': [0], 'zmethod': [2]}
time_settings: {'istimelapse': [True], 'timeinterval': [120], 'ntimepoints': [660], 'totaltime': [79200]}
positions: {'posname': ['pos001', 'pos002', 'pos003', 'pos004', 'pos005', 'pos006', 'pos007', 'pos008', 'pos009'], 'xpos': [568.0, 1267.0, 1026.0, 540.0, 510.0, -187.0, -731.0, -1003.0, -568.0], 'ypos': [1302.0, 1302.0, 977.0, -347.0, -687.0, -470.0, 916.0, 1178.0, 1157.0], 'zpos': [1876.5, 1880.125, 1877.575, 1868.725, 1867.15, 1864.05, 1867.05, 1866.425, 1868.45], 'pfsoffset': [122.45, 119.95, 120.1, 121.2, 122.9, 119.6, 117.05, 121.7, 119.35], 'group': [1, 1, 1, 2, 2, 2, 3, 3, 3], 'Brightfield': [30, 30, 30, 30, 30, 30, 30, 30, 30], 'GFPFast': [30, 30, 30, 30, 30, 30, 30, 30, 30], 'mCherry': [100, 100, 100, 100, 100, 100, 100, 100, 100]}
npumps: 2
pumpinit: {'pump_port': ['COM7', 'COM8'], 'syringe_diameter': [14.43, 14.43], 'flowrate': [0.0, 4.0], 'flowdirection': ['INF', 'INF'], 'isrunning': [True, True], 'contents': ['2% glucose in SC', '0.1% glucose in SC']}
nswitches: 1
switchvol: 50
switchrate: 100
switchtimes: [0]
switchtopump: [2]
switchfrompump: [1]
pumprate: [[0.0], [4.0]]
multiDGUI_commit: 05903fb3769ccf612e7801b46e2248644ce7ca28
date: 2020-02-29 00:00:00
microscope: Batman
acqfile: C:\path\to\example_multiDGUI_log.txt
details: Aim:   Strain:   Comments:
setup: Brightfield:
White LED
->(Polariser + Prism + condenser)]
->Filter block:[Dual GFP/mCherry exciter (59022x),Dual dichroic (89021bs),No emission filter]
->Emission filter wheel:[No filter in emission wheel]
GFPFast:
470nm LED
->Combiner cube:[480/40 exciter, 515LP dichroic->(455LP dichroic)]
->Filter block:[Dual GFP/mCherry exciter (59022x),Dual dichroic (89021bs),No emission filter]
->Emission filter wheel:[520/40 emission filter]
mCherry:
White LED
->Combiner cube:[No exciter, No reflecting dichroic->(515LP and 455LP dichroics)]
->Filter block:[Dual GFP/mCherry exciter (59022x),Dual dichroic (89021bs),No emission filter]
->Emission filter wheel:[632/60 emission filter]
Micromanager config file:C:\path\to\config_file.cfg
omero_project: SteadystateGlucose
omero_tags: ['29-Feb-2020', 'Batman', '3 chamber', 'GFP', 'mCherry', '1106.Mig2-GFP Mig1-mCherry', '900.Mig1-GFP Msn2-mCherry', '898.Msn2-GFP Mig1-mCherry', '0.1% glucose', '2% glucose', '']
expt_start: 2020-02-29 01:16:51
first_capture: 2020-02-29 01:17:01
omero_tags_stop: Time to next time point:-104.2112
"""


def test_essential_meta_fields(legacy_log_interface: dict):
    """
    We test the ability of the parser to find channel names and z-stacks
    """
    assert "channels" in legacy_log_interface, "Channels not found at root"
    assert len(
        legacy_log_interface["channels"]
    ), "Channels present but names not found"

    assert len(
        legacy_log_interface["channels"]
    ), "Channels present but names not found"
