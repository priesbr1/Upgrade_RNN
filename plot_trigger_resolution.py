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
parser.add_argument("-n", "--num_use", type=int, default=None,
                    dest="num_use", help="maximum number of events to use")
args = parser.parse_args()

input_files = args.input_files
output_folder = args.output_folder
pulse_type = str.lower(args.pulse_type)
num_use = args.num_use

if output_folder[-1] != '/':
    output_folder += '/'

if os.path.isdir(output_folder) != True:
    os.mkdir(output_folder)
print("Saving to %s"%output_folder)

def get_event_info(filename_list, pulse_type, num_use):

    all_pulse_times = []

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
                        pulse_times = []
                        for pulse in pulseseries: # Grab pulse information
                            pulse_times.append(pulse.time - trigger_time) # Shift pulse by event trigger time

                    all_pulse_times.append(pulse_times)
                    del pulseseriesmap

                else:
                    pass

        event_file.close()

    # Cut to maximum number of events, if applicable
    if num_use and len(all_pulse_times) > num_use:
        cut_pulses = all_pulse_times[:num_use]
        removed_events = len(all_pulse_times) - len(cut_pulses)
        print("Removed %i events -- using remaining %i events"%(removed_events, len(cut_pulses)))
    else:
        cut_pulses = all_pulse_times
        print("Keeping all %i events"%len(cut_pulses))

    return cut_pulses

def plot_trig_resolution(shifted_pulses, output_folder):

    # Flatten pulse array for plotting
    flattened_pulses = [pulse for pulse_list in shifted_pulses for pulse in pulse_list]
    print("Number of flattened pulses:", len(flattened_pulses))

    # Plotting code
    plt.figure()
    plt.title("Total Pulses Shifted by Trigger Time, n=%i"%len(shifted_pulses))
    plt.xlabel("True Pulse Time - Trigger Time [ns]")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.hist(flattened_pulses, range=(min(flattened_pulses), max(flattened_pulses)), bins=100, histtype='stepfilled', alpha=0.5)
    plt.axvline(x=0, color='green')
    imgname = output_folder+'trigger_resolution.png'
    plt.savefig(imgname)
    print("Plot saved as %s"%imgname)

if '*' in input_files or '?' in input_files:
    input_files = sorted(glob.glob(input_files))

shifted_pulses = get_event_info(input_files, pulse_type, num_use)
plot_trig_resolution(shifted_pulses, output_folder)
