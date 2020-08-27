#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
#METAPROJECT: simulation/V06-01-00-RC4

# we need icesim because of MuonGun
# icerec could be (for example): #METAPROJECT: icerec/V05-02-02-RC2

import os, sys
import glob
import numpy
import h5py
import argparse
from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units
#from icecube import MuonGun, simclasses

#input_files = sorted(glob.glob("/mnt/research/IceCube/jpandre/Matt/level5p/numu/14640/Level5p_IC86.2013_genie_numu.014640.000???.i3.bz2"))
#input_files = sorted(glob.glob("/mnt/scratch/priesbr1/Simulation_Files/step4_output_mixed.1?57.000???.i3.gz"))

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, default=None,
                    dest="input_files", help="name for input file")
parser.add_argument("-o", "--overwrite", type=bool, default=False,
                    dest="overwrite", help="whether or not ot overwrite previous file")
parser.add_argument("-p", "--oulse_type", type=str, default='uncleaned',
                    dest="pulse_type", help="type of pulseseries to use")
args = parser.parse_args()

input_files = args.input_files
overwrite = args.overwrite
pulse_type = str.lower(args.pulse_type)

def load_geometry(geometry_filename): # Gets geometry from specific geometry file
    
    geo_file = dataio.I3File(geometry_filename)
    while geo_file.more():
        frame = geo_file.pop_frame()
        if not (frame.Stop==icetray.I3Frame.Geometry):
            continue
        geometry = frame["I3Geometry"]
        om_geo_map = frame["I3OMGeoMap"]
        global list_of_omkeys
        list_of_omkeys = sorted(list(om_geo_map.keys()))
        geo_file.close()
        del geo_file
        return geometry
    
    geo_file.close()
    del geo_file
    return None
# geometry = load_geometry("/opt/i3-data/GCD/GeoCalibDetectorStatus_IC86.55697_V2.i3.gz")

def read_files(filename_list):
    def track_get_pos(p, length):
        if (not numpy.isfinite(length)) or (length < 0.) or (length >= p.length):
            return dataclasses.I3Position(numpy.nan, numpy.nan, numpy.nan)
        return dataclasses.I3Position( p.pos.x + length*p.dir.x, p.pos.y + length*p.dir.y, p.pos.z + length*p.dir.z )

    def track_get_time(p, length):
        if (not numpy.isfinite(length)) or (length < 0.) or (length >= p.length):
            return numpy.nan
        return p.time + length/p.speed

    weights = []
    
    features = dict()
    features['pmt_index'] = [] # Use PMT indexing for Upgrade simulation
    features['pulse_time'] = []
    features['pulse_charge'] = []

    labels = dict()
    labels['energy'] = []
    labels['azimuth'] = []
    labels['zenith'] = []
    labels['dir_x'] = []
    labels['dir_y'] = []
    labels['dir_z'] = []
    labels['isTrack'] = []
    labels['isCascade'] = []
    labels['isNC'] = []
    labels['isCC'] = []
    labels['track_length'] = []
    
    reco = dict()
    reco['energy'] = []
    reco['zenith'] = []
    reco['azimuth'] = []

    for event_file_name in filename_list:
        event_file = dataio.I3File(event_file_name)

        while event_file.more():
            try:
                frame = event_file.pop_physics() # Get next P frame if it exists
            except:
                continue

            # check correct P frame type
            if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
                continue
            else:
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
                    #print("Broken pulse_series_map - skipping event.")
                    continue

                nu = frame['I3MCTree'][0]
                nu_energy = nu.energy,
                nu_zen    = nu.dir.zenith,
                nu_azi    = nu.dir.azimuth,
                nu_x      = nu.pos.x,
                nu_y      = nu.pos.y,
                nu_z      = nu.pos.z,

                if frame.Has('IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC'): # Check if event reconstruction exists
                    reco_frame = frame['IC86_Dunkman_L6_PegLeg_MultiNest8D_NumuCC']
                    reco_energy = reco_frame.energy
                    reco_zenith = reco_frame.dir.zenith
                    reco_azimuth = reco_frame.dir.azimuth
                else:
                    reco_energy = 0
                    reco_zenith = 0
                    reco_azimuth = 0

                isCC = frame['I3MCWeightDict']['InteractionType']==1.
                isNC = frame['I3MCWeightDict']['InteractionType']==2.
                isOther = not isCC and not isNC
 
                # set track classification for numu CC only
                if ((nu.type == dataclasses.I3Particle.NuMu or nu.type == dataclasses.I3Particle.NuMuBar) and isCC): # Save NuMu/NuMuBar CC as track
                    isTrack = True 
                    isCascade = False
                    track_length = nu.length
                elif (nu.type == dataclasses.I3Particle.NuE or nu.type == dataclasses.I3Particle.NuEBar or 
                      nu.type == dataclasses.I3Particle.NuTau or nu.type == dataclasses.I3Particle.NuTauBar): # Save other neutrino as cascade
                    isTrack = False
                    isCascade = True
                    track_length = 0
                else: # Skip non-neutrinos
                    continue

                # calculate the event weight
                weightdict = frame['I3MCWeightDict']
                weight = weightdict['OneWeight'] * (weightdict['PrimaryNeutrinoEnergy'])**(-2.) / weightdict['NEvents']

                # store labels
                weights.append(weight)
                labels['energy'].append(nu_energy[0]/I3Units.GeV)
                labels['azimuth'].append(nu_azi[0]/I3Units.rad)
                labels['zenith'].append(nu_zen[0]/I3Units.rad)
                labels['dir_x'].append(nu_x[0])
                labels['dir_y'].append(nu_y[0])
                labels['dir_z'].append(nu_z[0])
                labels['isTrack'].append(isTrack)
                labels['isCascade'].append(isCascade)
                labels['isCC'].append(isCC)
                labels['isNC'].append(isNC)
                labels['track_length'].append(track_length)

                reco['energy'].append(reco_energy/I3Units.GeV)
                reco['zenith'].append(reco_zenith/I3Units.rad)
                reco['azimuth'].append(reco_azimuth/I3Units.rad)

                pmt_index = []
                pulse_time = []
                pulse_charge = []

                for omkey, pulseseries in pulseseriesmap: # Go through each event

                    # convert string, om, and pmt into a single index based on sorted list of omkeys
                    for i in range(len(list_of_omkeys)):
                        if (omkey.string == list_of_omkeys[i].string and omkey.om == list_of_omkeys[i].om and omkey.pmt == list_of_omkeys[i].pmt):
                            pmt_ind = i

                    for pulse in pulseseries: # Grab pulse information
                        pmt_index.append(pmt_ind)
                        pulse_time.append(pulse.time)
                        pulse_charge.append(pulse.charge)

                pulse_time = numpy.asarray(pulse_time, dtype=numpy.float64)
                pulse_charge = numpy.asarray(pulse_charge, dtype=numpy.float32)
                pmt_index = numpy.asarray(pmt_index, dtype=numpy.uint16)

                # sort the arrays by time (second "feature", index 1)
                sorting = numpy.argsort(pulse_time)
                pulse_time = pulse_time[sorting]
                pulse_charge = pulse_charge[sorting]
                pmt_index = pmt_index[sorting]

                # convert absolute times to relative times
                #pulse_time[1:] -= pulse_time[:-1]
                #pulse_time[0] = 0.
                avg_time = numpy.mean(pulse_time)
                pulse_time -= avg_time
                pulse_time = numpy.asarray(pulse_time, dtype=numpy.float32)

                features['pmt_index'].append(pmt_index)
                features['pulse_time'].append(pulse_time)
                features['pulse_charge'].append(pulse_charge)

                del pulseseriesmap

        event_file.close()

    weights = numpy.asarray(weights, dtype=numpy.float64)
    for k in labels.keys():
        labels[k] = numpy.asarray(labels[k], dtype=numpy.float64)

    return (features, labels, reco, weights)

def write_hdf5_file(filename, features, labels, reco, weights):
    f = h5py.File(output_file, "w")
    grp_features = f.create_group("features")
    grp_labels   = f.create_group("labels")
    if reco != None: grp_reco = f.create_group("reco")

    f.create_dataset("weights", data=weights)
    for k in labels.keys():
        grp_labels.create_dataset(k, data=labels[k])

    for k in features.keys():
        features[k]
        dt = h5py.special_dtype(vlen=features[k][0].dtype)
        dset = grp_features.create_dataset(k, (len(features[k]), ), dtype=dt)
    
        for i in range(len(features[k])):
            dset[i] = features[k][i]

    if reco != None:
        for k in reco.keys():
            grp_reco.create_dataset(k, data=reco[k])

    f.close()   

def strip_i3_ext(filename, keep_path=True):
    path, name = os.path.split(filename)

    while True:
        basename, ext = os.path.splitext(os.path.basename(name))
        if (ext == "") or (ext == '.i3'):
            if keep_path:
                if pulse_type == 'cleaned':
                    return os.path.join(path, basename[:18]+"_cleaned"+basename[18:])
                else:
                    return os.path.join(path, basename)
            else:
                if pulse_type == 'cleaned':
                    return basename[:18]+"_cleaned"+basename[18:]
                else:
                    return basename
        name = basename

load_geometry("/mnt/scratch/priesbr1/Simulation_Files/GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2")

if input_files == None:
    input_files = sorted(glob.glob('/mnt/scratch/priesbr1/Simulation_Files/step4_output_mixed.1457.000???.i3.gz'))
    print("Defaulting to input files: /mnt/scratch/priesbr1/Simulation_Files/step4_output_mixed.1457.000???.i3.gz")

if '*' in input_files or '?' in input_files:
    input_files = sorted(glob.glob(input_files))

if isinstance(input_files, list):
    for input_file in input_files:
        output_file = "/mnt/scratch/priesbr1/Processed_Files/" + strip_i3_ext(input_file, keep_path=False) + ".hdf5"

        if os.path.isfile(output_file) == True and overwrite == False:
            print("Skipping file -- %s already exists"%output_file)

        else:
            print("Reading {}...".format(input_file))
            features, labels, reco, weights = read_files([input_file])

            if sum(reco['energy']) == 0:
                reco = None

            if len(weights) > 0 and os.path.isfile(output_file) == False:
                print("Writing {}...".format(output_file))
                write_hdf5_file(output_file, features, labels, reco, weights)
            elif len(weights) > 0 and os.path.isfile(output_file) == True and overwrite == True:
                print("Overwriting {}...".format(output_file))
                write_hdf5_file(output_file, features, labels, reco, weights)
            else:
                print("No output to write, file {} is empty".format(input_file))

elif isinstance(input_files, str):
    input_file = input_files
    output_file = "/mnt/scratch/priesbr1/Processed_Files/" + strip_i3_ext(input_file, keep_path=False) + ".hdf5"

    if os.path.isfile(output_file) == True and overwrite == False:
        print("Skipping file -- %s already exists"%output_file)

    else:
        print("Reading {}...".format(input_file))
        features, labels, reco, weights = read_files([input_file])

        if sum(reco['energy']) == 0:
            reco = None

        if len(weights) > 0 and os.path.isfile(output_file) == False:
            print("Writing {}...".format(output_file))
            write_hdf5_file(output_file, features, labels, reco, weights)
        elif len(weights) > 0 and os.path.isfile(output_file) == True and overwrite == True:
            print("Overwriting {}...".format(output_file))
            write_hdf5_file(output_file, features, labels, reco, weights)
        else:
            print("No output to write, file {} is empty".format(input_file))

else:
    print("Unknown data type for input file(s):", type(input_files))
