# Plot the resolution between event pulse times and trigger time
# By default, only uses events with DeepCore SMT3 trigger

import os, sys
import glob
import numpy
import matplotlib.pyplot as plt
import argparse
from icecube import icetray, dataio, dataclasses

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files", type=str, default=None,
                    dest="input_files", help="name for input file")
parser.add_argument("-o", "--output_folder", type=str, default=None,
                    dest="output_folder", help="name for output folder")
parser.add_argument("-p", "--pulse_type", type=str, default=None,
                    dest="pulse_type", help="type of pulseseries to use")
parser.add_argument("-l", "--logscale", type=bool, default=False,
                    dest="logscale", help="whether or not to use y logscale for plotting")
parser.add_argument("-n", "--num_use", type=int, default=None,
                    dest="num_use", help="maximum number of events to use")
args = parser.parse_args()

input_files = args.input_files
output_folder = args.output_folder
pulse_type = str.lower(args.pulse_type)
logscale = args.logscale
num_use = args.num_use

if output_folder[-1] != '/':
    output_folder += '/'

if os.path.isdir(output_folder) != True:
    os.mkdir(output_folder)
print("Saving to %s"%output_folder)

def get_event_info(filename_list, pulse_type, num_use):

    all_pulse_times = []
    DC_pulse_times = []

    if type(filename_list) == str:
        filename_list = [filename_list]

    for event_file_name in filename_list:
        print("Reading {}...".format(event_file_name))
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics()
            except:
                continue

            # check correct P frame type
            if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
                continue
            else:
                SMT3_flag = False
                # Config 1011 is SMT3
                # dataclasses.TriggerKey(source, ttype, config_id)
                triggers = frame['I3TriggerHierarchy']
                for trig in triggers:
                    key_str = str(trig.key)
                    s = key_str.strip('[').strip(']').split(':')
                    if len(s) > 2:
                        config_id = int(s[2])
                        if config_id == 1011:
                            trigger_time = trig.time
                            SMT3_flag = True
                            break

                if SMT3_flag == True: # DeepCore SMT3 trigger
                    # get all pulses
                    pulseseriesmap = None
                    try:
                        if pulse_type == 'uncleaned':
                            pulseseriesmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulses')
                        elif pulse_type == 'cleaned':
                            pulseseriesmap = dataclasses.I3RecoPulseSeriesMap.from_frame(frame, 'SplitInIcePulsesSRT')
                        else:
                            raise RuntimeError("Unknown pulseseries type specified: %s"%pulse_type)
                    except:
                        pulseseriesmap = None
                    if pulseseriesmap is None:
                        print("Broken pulse_series_map - skipping event.")
                        continue

                    for omkey, pulseseries in pulseseriesmap: # Go through each event
                        all_pulses = []
                        DC_pulses = []
                        for pulse in pulseseries: # Grab pulse information
                            all_pulses.append(pulse.time - trigger_time)
                            if (omkey.string <= 86) and (omkey.string >= 79) and (omkey.om >= 11): # Select DeepCore DOMs in DeepCore fiducial volume
                                DC_pulses.append(pulse.time - trigger_time)
                            elif (omkey.string >= 87) and (omkey.string <= 93): # All Upgrade DOMs in DeepCore fiducial volume
                                DC_pulses.append(pulse.time - trigger_time)

                    all_pulse_times.append(all_pulses)
                    if len(DC_pulses) > 0:
                        DC_pulses_times.append(DC_pulses)
                    del pulseseriesmap

                else:
                    pass

        event_file.close()

    # Cut to maximum number of events, if applicable
    if num_use and len(all_pulse_times) > num_use:
        cut_all_pulses = all_pulse_times[:num_use]
        removed_events = len(all_pulse_times) - len(cut_all_pulses)
        print("Removed %i events -- using remaining %i events"%(removed_events, len(cut_all_pulses)))
    else:
        cut_all_pulses = all_pulse_times
        print("Keeping all %i events"%len(cut_all_pulses))

    if num_use and len(DC_pulse_times) > num_use:
        cut_DC_pulses = DC_pulse_times[:num_use]
        removed_events = len(DC_pulse_times) - len(cut_DC_pulses)
        print("Removed %i DeepCore events -- using remaining %i events"%(removed_events, len(cut_DC_pulses)))
    else:
        cut_DC_pulses = DC_pulse_times
        print("Keeping all %i DeepCore events"%len(cut_DC_pulses))

    return cut_all_pulses, cut_DC_pulses

def plot_trig_resolution(shifted_all_pulses, shifted_DC_pulses, output_folder, logscale, num_use):

    # Flatten pulse array for plotting
    flattened_all_pulses = [pulse for pulse_list in shifted_all_pulses for pulse in pulse_list]
    flattened_DC_pulses = [pulse for pulse_list in shifted_DC_pules for pulse in pulse_list]
    print("Number of flattened pulses:", len(flattened_all_pulses))
    print("Number of flattened DC pulses:", len(flattened_DC))

    # Plotting code
    plt.figure()
    plt.title("Total Pulses Shifted by Trigger Time")
    plt.xlabel("True Pulse Time - Trigger Time [ns]")
    plt.ylabel("Counts")
    if logscale == True:
        plt.yscale('log')
    plt.hist(flattened_all_pulses, range=(min(flattened_all_pulses), max(flattened_all_pulses)), bins=100, histtype='stepfilled', alpha=0.5, label="All, n=%i"%(len(shifted_all_pulses)))
    plt.hist(flattened_DC_pulses, range=(min(flattened_DC_pulses), max(flattened_DC_pulses)), bins=100, histtype='stepfilled', alpha=0.5, label="DC, n=%i"%(len(shifted_DC_pulses)))
    plt.axvline(x=0, color='green')
    plt.legend(loc="best")
    plt.tight_layout()
    if len(shifted_all_pulses) != num_use and logscale == True:
        imgname = output_folder+'trigger_resolution_ylog_all.png'
    elif len(shifted_all_pulses) != num_use:
        imgname = output_folder+'trigger_resolution_all.png'
    elif logscale == True:
        imgname = output_folder+'trigger_resolution_ylog_'+str(num_use)+'.png'
    else:
        imgname = output_folder+'trigger_resolution_'+str(num_use)+'.png'
    plt.savefig(imgname)
    print("Plot saved as %s"%imgname)

if '*' in input_files or '?' in input_files:
    input_files = sorted(glob.glob(input_files))

shifted_all_pulses, shifted_DC_pulses = get_event_info(input_files, pulse_type, num_use)
plot_trig_resolution(shifted_all_pulses, shifted_DC_pulses, output_folder, logscale, num_use)
