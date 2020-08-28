#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v3.0.1/icetray-start
#METAPROJECT: simulation/V06-01-00-RC4

import numpy
import h5py
import glob
import sys
import math
import argparse
numpy.set_printoptions(threshold=sys.maxsize)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default=None,
                    dest="input_file", help="name of input hdf5 file")
parser.add_argument("--base_name", type=str, default=None,
                    dest="base_name", help="base of output file name")
parser.add_argument("--energy_range", type=int, nargs='+', default=[5,200],
                    dest="energy_range", help="minimum and maximum energy values")
parser.add_argument("--keep_track", type=bool, default=True,
                    dest="keep_track", help="include (True) or exclude (False) track events from sample")
parser.add_argument("--keep_cascade", type=bool, default=True,
                    dest="keep_cascade", help="include (True) or exclude (False) cascade events from sample")
parser.add_argument("--keep_CC", type=bool, default=True,
                    dest="keep_CC", help="include (True) or exclude (False) CC events from sample")
parser.add_argument("--keep_NC", type=bool, default=False,
                    dest="keep_NC", help="include (True) or exclude (False) NC events from sample")
parser.add_argument("--vertex_cut", type=str, default='all_start',
                    dest="vertex_cut", help="type of vertex cut to apply")
parser.add_argument("--flat_energy", type=bool, default=False,
                    dest="flatten_energy", help="flatten energy sample in 1-GeV bins")
parser.add_argument("--flat_tc", type=bool, default=False,
                    dest="flatten_tc", help="flatten the number of tracks and cascades in sample")
parser.add_argument("--min_PMT", type=int, default=0,
                    dest="min_PMT", help="minimum number of PMTs required for event to pass")
parser.add_argument("--max_events", type=int, default=0,
                    dest="max_events", help="maximum number of events to use in sample")
parser.add_argument("--only_reco", type=bool, default=False,
                    dest="only_reco", help="include only reco events in sample")
args = parser.parse_args()

energy_range = args.energy_range
minenergy = energy_range[0]
maxenergy = energy_range[1]
include_tracks = args.keep_track
include_cascades = args.keep_cascade
include_CC = args.keep_CC
include_NC = args.keep_NC
min_PMT = args.min_PMT # makes optional cut of events that only hit a certain number of PMTs
maxevents = args.max_events
only_reco = args.only_reco
vertex_cut = args.vertex_cut # makes optional cut of events that fits specified starting/ending positions (after PMT cut, if applicable)
flatten_energy = args.flatten_energy # if true, script flattens energy spectrum in the set range (after PMT cut and/or  vertex cut, if applicable)
adjust_tc = args.flatten_tc # if true, number of tracks and cascades are evened out (after PMT cut and/or vertex cut and/or energy flattening, if applicable)

print("Printing cuts...")
if min_PMT > 0: print("Minimum number of PMTs hit: %i"%min_PMT)
else: print("No cut on minimum number of PMTs hit")
print("Vertex cut: %s"%vertex_cut)
print("Energy range: %i-%i GeV"%(minenergy,maxenergy))
print("Tracks/cascades: %s/%s"%(include_tracks,include_cascades))
print("CC/NC: %s/%s"%(include_CC,include_NC))
if only_reco == True: print("Only including reco events")
else: print("Keeping events with and without PegLeg reconstructions")
print("Flat energy distribution: %s"%flatten_energy)
print("Equal number of tracks/cascades: %s"%adjust_tc)
if maxevents != 0: print("Only keeping up to %i events"%maxevents)
else: print("No cut on maximum number of events")

fin = h5py.File(args.input_file, "r")

label_keys = list(fin['labels'].keys())
feature_keys = list(fin['features'].keys())
if 'reco' in fin: reco_keys = list(fin['reco'].keys())

features = dict()
labels = dict()
if 'reco' in fin: reco = dict()

for k in label_keys:
    labels[k] = numpy.array(fin['labels'][k])
for k in feature_keys:
    features[k] = numpy.array(fin['features'][k])
if 'reco' in fin:
    for k in reco_keys:
        reco[k] = numpy.array(fin['reco'][k])
weights = numpy.array(fin['weights'])

total_entries = len(weights)

exist_CC = len(labels["isCC"][labels["isCC"] == 1.0]) > 0
exist_NC = len(labels["isNC"][labels["isNC"] == 1.0]) > 0
exist_track = len(labels["isTrack"][labels["isTrack"] == 1.0]) > 0
exist_cascade = len(labels["isCascade"][labels["isCascade"] == 1.0]) > 0

output_file = args.base_name
output_file += '_%ito%i'%(minenergy,maxenergy)

if flatten_energy == True:
    output_file += '_flat_energy'

if adjust_tc == True and include_tracks == True and include_cascades == True:
    output_file += '_flat_track_cascade'

if include_CC == True and include_NC == True and exist_CC == True and exist_NC == True:
    output_file += '_CCNC'
elif include_CC == True and exist_CC == True and exist_NC == True:
    output_file += '_CConly'
elif include_CC == True and exist_CC == True:
    output_file += '_CC'
elif include_NC == True and exist_CC == True and exist_NC == True:
    output_file += '_NConly'
elif include_NC == True and exist_NC == True:
    output_file += '_NC'

elif include_CC == True and exist_CC == False:
    raise RuntimeError("No CC events to keep")
elif include_NC == True and exist_NC == False:
    raise RuntimeError("No NC events to keep")
elif include_CC == False and include_NC == False:
    raise RuntimeError("Must keep CC or NC or both events")
elif exist_CC == False and exist_NC == False:
    raise RuntimeError("No CC/NC events in input file")

if include_tracks == True and include_cascades == False and exist_track == True:
    output_file += '_trackonly'
elif include_tracks == False and include_cascades == True and exist_cascade == True:
    output_file += '_cascadeonly'

elif include_tracks == False and include_cascades == False:
    raise RuntimeError("Must keep track or cascade ot both events")
elif exist_track == False and exist_cascade == False:
    raise RuntimeError("No track/cascade events in input file")

if min_PMT:
    output_file += '_%iminPMT'%min_PMT

if vertex_cut != None and vertex_cut != 'all_start' and vertex_cut != 'all_end':
    output_file += '_vertex_%s'%vertex_cut
else:
    output_file += '_allvertex'

if only_reco == True:
    output_file += '_onlyreco'

if maxevents > 0:
    num_digits = len(str(maxevents))
    if num_digits <= 3:
        num_prefix = maxevents
        output_file += '%i_max'%num_prefix
    elif num_digits <= 6:
        num_prefix = str(maxevents)[:-3]
        output_file += '%iK_max'%num_prefix
    elif num_digits <= 9:
        num_prefix = str(max_events)[:-6]
        output_file += '%iM_max'%num_prefix
    elif num_digits <= 12:
        num_prefix = str(max_events)[:-9]
        output_file += '%iB_max'%num_prefix
    else:
        raise Exception("Maxevents is greater than 1 trillion -- too big!")
        
output_file += '.hdf5'

fout = h5py.File("/mnt/scratch/priesbr1/Data_Files/"+output_file, "w")

#shuffle entries
order = numpy.arange(total_entries)
numpy.random.shuffle(order)
for k in label_keys:
    labels[k] = labels[k][order]
for k in feature_keys:
    features[k] = features[k][order]
if 'reco' in fin:
    for k in reco_keys:
        reco[k] = reco[k][order]
weights = weights[order]

print('Finished reading file')

#------------------------------------------------------------

if min_PMT != 0:
    print("Masking for events that have at least %i hit PMTs"%min_PMT)
    pmt_mask = []
    for i in range(len(features['pmt_index'])):
        pmt_ids = features['pmt_index'][i]
        unique_pmts = numpy.unique(pmt_ids)
        if len(unique_pmts) >= min_PMT:
            pmt_mask.append(True)
        else:
            pmt_mask.append(False)

    for k in label_keys:
        labels[k] = labels[k][pmt_mask]
    for k in feature_keys:
        if k != 'pmt_index': features[k] = features[k][pmt_mask]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][pmt_mask]
    weights = weights[pmt_mask]
    features['pmt_index'] = features['pmt_index'][pmt_mask]

if vertex_cut != None:
    # Position of String 36 for origin
    x_origin = 46.290000915527344
    y_origin = -34.880001068115234

    # Load true labels
    theta = labels['zenith']
    phi = labels['azimuth']
    x_start = labels['dir_x']
    y_start = labels['dir_y']
    z_start = labels['dir_z']
    track_length = labels['track_length']
    n_x = numpy.sin(theta)*numpy.cos(phi)
    n_y = numpy.sin(theta)*numpy.sin(phi)
    n_z = numpy.cos(theta)
    x_end = x_start + track_length*n_x
    y_end = y_start + track_length*n_y
    z_end = z_start + track_length*n_z

    # Set up Boundary conditions
    start_boundary = 50
    z_min_start = -505 - start_boundary
    z_max_start = -155 + start_boundary
    end_boundary = 50
    z_min_end = -505 - end_boundary
    z_max_end = 505 + end_boundary
    radius_IC7 = 150
    radius_DC = 90
    radius_IC19 = 260

    old_z_mask_start = numpy.logical_and(z_start > -505, z_start < 192)
    z_mask_start = numpy.logical_and(z_start > z_min_start, z_start < z_max_start)
    r_start = numpy.sqrt( (x_start - x_origin)**2 + (y_start - y_origin)**2 )
    z_mask_end = numpy.logical_and(z_end > z_min_end, z_end < z_max_end)
    r_end = numpy.sqrt((x_end - x_origin)**2 + (y_end - y_origin)**2)

    vertex_mask = {}
    vertex_mask['all_start'] = numpy.ones((len(theta)),dtype=bool)
    vertex_mask['old_start_DC'] = numpy.logical_and(old_z_mask_start, r_start < radius_DC)
    vertex_mask['start_DC'] = numpy.logical_and(z_mask_start, r_start < radius_DC)
    vertex_mask['start_IC7'] = numpy.logical_and(z_mask_start, r_start < radius_IC7)
    vertex_mask['start_IC19'] = numpy.logical_and(z_mask_start, r_start < radius_IC19)
    vertex_mask['all_end'] = numpy.ones((len(theta)),dtype=bool)
    vertex_mask['end_IC7'] = numpy.logical_and(z_mask_end, r_end < radius_IC7)
    vertex_mask['end_IC19'] = numpy.logical_and(z_mask_end, r_end < radius_IC19)

    vertex_cut_options = list(vertex_mask.keys())

    if vertex_cut not in vertex_cut_options:
        raise Exception("Unrecognized option for vertex cut: %s. Please use one of the specified options: %s."%(vertex_cut, vertex_cut_options))
    print("Masking for events with %s vertex constraint"%vertex_cut)

    for k in label_keys:
        labels[k] = labels[k][vertex_mask[vertex_cut]]
    for k in feature_keys:
        features[k] = features[k][vertex_mask[vertex_cut]]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][vertex_mask[vertex_cut]]
    weights = weights[vertex_mask[vertex_cut]]

#----------

if 'energy' in label_keys:
    print("Masking for energies between %i and %i GeV"%(minenergy, maxenergy))
    boolarray = numpy.logical_and(numpy.array(labels['energy']) > minenergy, numpy.array(labels['energy']) < maxenergy)
    for k in label_keys:
        if k != 'energy': labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    labels['energy'] = labels['energy'][boolarray]

#----------

if 'isTrack' in label_keys and include_tracks == False:
    print("Masking out track events")
    boolarray = numpy.array(labels['isTrack']) == False
    for k in label_keys:
        if k != 'isTrack': labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    labels['isTrack'] = labels['isTrack'][boolarray]

if 'isCascade' in label_keys and include_cascades == False:
    print("Masking out cascade events")
    boolarray = numpy.array(labels['isCascade']) == False
    for k in label_keys:
        if k != 'isCascade': labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    labels['isCascade'] = labels['isCascade'][boolarray]

#----------

if 'isCC' in label_keys and include_CC == False:
    print("Masking out CC events")
    boolarray = numpy.array(labels['isCC']) == False
    for k in label_keys:
        if k != 'isCC': labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    labels['isCC'] = labels['isCC'][boolarray]

if 'isNC' in label_keys and include_NC == False:
    print("Masking out NC events")
    boolarray = numpy.array(labels['isNC']) == False
    for k in label_keys:
        if k != 'isNC': labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    labels['isNC'] = labels['isNC'][boolarray]

#----------

if only_reco == True and 'reco' in fin:
    print("Only keeping events with reconstructions")
    boolarray = numpy.array(reco['energy']) != 0
    for k in label_keys:
        labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    for k in reco_keys:
        if k != 'energy': reco[k] = reco[k][boolarray]
    weights = weights[boolarray]
    reco['energy'] = reco['energy'][boolarray]

#----------

binwidth = 1.0 #units are GeV
if flatten_energy == True:
    print("Flattening energy distribution")
    energymax = math.ceil(max(labels['energy']))
    energymin = math.floor(min(labels['energy']))
    no_bins = int(math.ceil((energymax - energymin)/binwidth))
    ranges = numpy.linspace(energymin, energymax, num=no_bins+1)

    #find energy bin with lowest number of entries
    energy_bin_length = []
    for i in range(no_bins):
        boolarray = numpy.logical_and(numpy.array(labels['energy']) >= ranges[i], numpy.array(labels['energy']) < ranges[i+1])
        energy_bin_length.append(len(labels['energy'][boolarray]))
    print("Energy bin counts:", energy_bin_length)
    non_zero = [(element>0) for element in energy_bin_length]
    energy_bin_length = numpy.array(energy_bin_length)
    nonzero_energy_lengths = energy_bin_length[non_zero].tolist()
    min_entries = min(nonzero_energy_lengths)
    print("Minimum non-zero counts in a bin:", min_entries)

    #cut out extra entries in energy bins
    for i in range(no_bins):
        boolarray = numpy.logical_and(numpy.array(labels['energy']) >= ranges[i], numpy.array(labels['energy']) < ranges[i+1])
        energy_indices = numpy.argwhere(boolarray).flatten()
        energy_indices = energy_indices[numpy.array(range(len(energy_indices))) > min_entries]

        for k in label_keys:
            labels[k] = numpy.delete(labels[k], energy_indices)
        for k in feature_keys:
            features[k] = numpy.delete(features[k], energy_indices)
        if 'reco' in fin:
            for k in reco_keys:
                reco[k] = numpy.delete(reco[k], energy_indices)
        weights = numpy.delete(weights, energy_indices)

#----------

if adjust_tc == True and include_tracks == True and include_cascades == True:
    print("Masking for equal numbers of tracks and cascades")
    no_tracks = len(labels['isTrack'][numpy.array(labels['isTrack'] == True)])
    no_cascades = len(labels['isCascade'][numpy.array(labels['isCascade'] == True)])

    if no_tracks > no_cascades:
        min_entries = no_cascades
        cut_label = 'isTrack'
    elif no_tracks < no_cascades:
        min_entries = no_tracks
        cut_label = 'isCascade'

    indices = numpy.argwhere(numpy.array(labels[cut_label]))
    indices = indices[numpy.array(range(len(indices))) > min_entries]

    for k in label_keys:
        labels[k] = numpy.delete(labels[k], indices)
    for k in feature_keys:
        features[k] = numpy.delete(features[k], indices)
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = numpy.delete(reco[k], indices)
    weights = numpy.delete(weights, indices)

#----------

if maxevents != 0:
    print("Only keeping %i events"%maxevents)
    boolarray = numpy.array(range(len(weights))) < maxevents
    for k in label_keys:
        labels[k] = labels[k][boolarray]
    for k in feature_keys:
        features[k] = features[k][boolarray]
    if 'reco' in fin:
        for k in reco_keys:
            reco[k] = reco[k][boolarray]
    weights = weights[boolarray]

#------------------------------------------------------------

#shuffle entries
remaining_entries = len(weights)
order = numpy.arange(remaining_entries)
numpy.random.shuffle(order)
for k in label_keys:
    labels[k] = labels[k][order]
for k in feature_keys:
    features[k] = features[k][order]
if 'reco' in fin:
    for k in reco_keys:
        reco[k] = reco[k][order]
weights = weights[order]

print('Finished making cuts')

print('Total events before cuts:', total_entries)
print('Total events after cuts:', remaining_entries)

grp_features = fout.create_group("features")
grp_labels   = fout.create_group("labels")
if 'reco' in fin: grp_reco = fout.create_group("reco")
grp_weights = fout.create_dataset("weights", data=weights)

for k in label_keys:
    grp_labels.create_dataset(k, data=labels[k])
for k in feature_keys:
    grp_features.create_dataset(k, data=features[k])
if 'reco' in fin:
    for k in reco_keys:
        grp_reco.create_dataset(k, data=reco[k])

fin.close()
fout.close()
