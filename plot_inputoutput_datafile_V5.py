import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy
import numpy as np
import h5py
import scipy.stats
import glob
import os
import sys
import math
import argparse
import time

from Generators_V5 import DataGenerator, SplitGenerator
from Attention import AttentionWithContext
from Plots import plot_uncertainty, plot_2dhist, plot_1dhist, plot_error, plot_loss, plot_error_vs_reco, plot_inputs, plot_outputs, plot_outputs_classify, plot_vertex, plot_hit_info

import keras
import tensorflow as tf
from keras.utils.generic_utils import Progbar
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Embedding, BatchNormalization
from keras.optimizers import Adam

from keras.layers import Lambda, Flatten, Reshape, CuDNNLSTM, LSTM, Bidirectional, Activation, Dropout
from keras.layers import Conv1D, SpatialDropout1D, MaxPooling1D

numpy.set_printoptions(threshold=sys.maxsize)

def normalize(input_file, input_labels, use_log_energy):
    label_keys = [k for k in input_file['labels'].keys()]
    total_entries = len(input_file['weights'])
    normalization = dict()
    for k in input_labels:
        normalization[k] = numpy.zeros(2)
        for i in range(total_entries):
            if k == 'energy' and use_log_energy: normalization[k][0] += numpy.log10(input_file['labels'][k][i])/total_entries
            elif k == 'dx': normalization[k][0] += numpy.sin(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))*numpy.cos(numpy.radians(input_file['labels']['azimuth'][i])-numpy.pi)/total_entries
            elif k == 'dy': normalization[k][0] += numpy.sin(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))*numpy.sin(numpy.radians(input_file['labels']['azimuth'][i])-numpy.pi)/total_entries
            elif k == 'dz': normalization[k][0] += numpy.cos(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))/total_entries
            elif k in label_keys: normalization[k][0] += input_file['labels'][k][i]/total_entries

        for i in range(total_entries):
            if k == 'energy' and use_log_energy: normalization[k][1] += ((numpy.log10(input_file['labels'][k][i])-normalization[k][0])**2)/total_entries
            elif k == 'dx': normalization[k][1] += ((numpy.sin(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))*numpy.cos(numpy.radians(input_file['labels']['azimuth'][i])-numpy.pi)-normalization[k][0])**2)/total_entries
            elif k == 'dy': normalization[k][1] += ((numpy.sin(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))*numpy.sin(numpy.radians(input_file['labels']['azimuth'][i])-numpy.pi)-normalization[k][0])**2)/total_entries
            elif k == 'dz': normalization[k][1] += ((numpy.cos(numpy.pi-numpy.radians(input_file['labels']['zenith'][i]))-normalization[k][0])**2)/total_entries
            elif k in label_keys: normalization[k][1] += ((input_file['labels'][k][i]-normalization[k][0])**2)/total_entries
        normalization[k][1] = math.sqrt(normalization[k][1])
        print(k,normalization[k])
    return normalization

def energy_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true[:,0], y_pred[:,0])

def direction_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true[:,1], y_pred[:,2]) + keras.losses.mean_squared_error(y_true[:,2], y_pred[:,3]) + keras.losses.mean_squared_error(y_true[:,3], y_pred[:,4])

#def classification_loss(y_true, y_pred):
#    return keras.losses.binary_crossentropy(y_true[:,4], y_pred[:,4]) + keras.losses.binary_crossentropy(y_true[:,5], y_pred[:,5])

def energy_uncertainty_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_pred[:,1], tf.stop_gradient(tf.math.abs(y_true[:,0]-y_pred[:,0])))

def direction_uncertainty_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_pred[:,5], tf.stop_gradient(tf.math.abs(y_true[:,1]-y_pred[:,2]))) + keras.losses.mean_squared_error(y_pred[:,6], tf.stop_gradient(tf.math.abs(y_true[:,2]-y_pred[:,3]))) + keras.losses.mean_squared_error(y_pred[:,7], tf.stop_gradient(tf.math.abs(y_true[:,3]-y_pred[:,4])))

def customLoss(y_true, y_pred):
    e_loss = energy_loss(y_true, y_pred) + energy_uncertainty_loss(y_true, y_pred)
    d_loss = direction_loss(y_true, y_pred) + direction_uncertainty_loss(y_true, y_pred)
    loss = e_loss/700.0 + d_loss*8.0

def to_xyz(zenith, azimuth):
    theta = numpy.pi-zenith
    phi = azimuth-numpy.pi
    rho = numpy.sin(theta)
    return rho*numpy.cos(phi), rho*numpy.sin(phi), numpy.cos(theta)
    
def to_zenazi(x,y,z):
    r = numpy.sqrt(x*x+y*y+z*z)
    theta = numpy.zeros(len(r))
        
    normal_bins = (r>0.) & (numpy.abs(numpy.asarray(z)/r)<=1.)
    theta[normal_bins] = numpy.arccos(numpy.asarray(z)/r)
    theta[numpy.logical_not(normal_bins) & (numpy.asarray(z) < 0.)] = numpy.pi
    theta[theta<0.] += 2.*numpy.pi
    
    phi = numpy.zeros(len(r))
    phi[ (numpy.asarray(x)!=0.) & (numpy.asarray(y)!=0.) ] = numpy.arctan2(y,x)
    phi[phi < 0.] += 2.*numpy.pi

    zenith = numpy.pi - theta
    azimuth = phi + numpy.pi
   
    zenith[zenith > numpy.pi] -= numpy.pi-(zenith[zenith > numpy.pi]-numpy.pi)
    azimuth -= (azimuth/(2.*numpy.pi)).astype(numpy.int).astype(numpy.float) * 2.*numpy.pi
    
    return zenith, azimuth

def forward_generators(gen_train, gen_val, last_checkpoint_epoch):

    print("fast-forwarding generators...")
    initial_epoch = 0
    while initial_epoch < last_checkpoint_epoch:
        # request at least one item, just to make sure
        print("  forwarding one epoch...")

        dummy = gen_train[0]
        dummy = gen_val[0]
        del dummy

        gen_train.on_epoch_end()
        gen_val.on_epoch_end()

        initial_epoch += 1

    return gen_train, gen_val 
    
def main(config=1):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-h", "--hits",type=int,default=150, dest="hits", help="number of pmt hits used for training")
    parser.add_argument("-e", "--epochs",type=int,default=30, dest="epochs", help="number of training epochs")
    parser.add_argument("-d", "--decay",type=float,default=0.0, dest="decay", help="learning rate decay parameter")
    parser.add_argument("-r", "--lr", type=float,default=0.001, dest="lr", help="learning rate")
    parser.add_argument("-o", "--dropout", type=float,default=0.1, dest="dropout", help="change network dropout for each layer")
    parser.add_argument("-l", "--log_energy", type=int,default=0, dest="log_energy", help="use log energy rather than absolute for training")
    parser.add_argument("-f", "--file", type=str, default="outfile_l5p_le.hdf5", dest="file_name", help="file to use for training")
    parser.add_argument("-p", "--path", type=str, default="/mnt/scratch/priesbr1/Data_Files/", dest="path", help="path to input file")
    parser.add_argument("-u", "--output", type=str, default="/mnt/scratch/priesbr1/Upgrade_RNN/Reference_Plots/", dest="output", help="output folder destination")
    parser.add_argument("-s", "--standardize", type=int,default=0, dest="standardize", help="perform data standardization")
    parser.add_argument("-c", "--checkpoints", type=int,default=0, dest="checkpoints", help="use training checkpoints from previous run")
    parser.add_argument("-w", "--weights", type=int,default=1, dest="weights", help="use sample weights for training")
    parser.add_argument("-n", "--num_use", type=int,default=None, dest="num_use", help="number of samples to use in plotting")
    args = parser.parse_args()

    no_hits = args.hits
    no_epochs = args.epochs
    decay = args.decay
    learning_rate = args.lr
    dropout = args.dropout
    use_log_energy = bool(args.log_energy)
    ff_name = args.path + args.file_name
    use_standardization = bool(args.standardize)
    use_checkpoints = bool(args.checkpoints)
    use_weights = bool(args.weights)
    num_use = args.num_use

    ff = h5py.File(ff_name, 'r')
    global save_folder_name
    save_folder_name = args.output + args.file_name.replace('.hdf5', '/')
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
    print("Saving to:", save_folder_name)

    reco = False
    if "reco" in ff.keys(): reco = True

    pulse_time = numpy.array(ff["features/pulse_time"][:])
    pulse_charge = numpy.array(ff["features/pulse_charge"][:])
    pmt_index = numpy.array(ff["features/pmt_index"][:])

    energy_true = numpy.array(ff["labels/energy"][:])
    x_true = numpy.array(ff["labels/dir_x"][:])
    y_true = numpy.array(ff["labels/dir_y"][:])
    z_true = numpy.array(ff["labels/dir_z"][:])
    zenith_true = numpy.array(ff["labels/zenith"][:])
    azimuth_true = numpy.array(ff["labels/azimuth"][:])
    track_true = numpy.array(ff["labels/isTrack"][:])
    cascade_true = numpy.array(ff["labels/isCascade"][:])
    CC_true = numpy.array(ff["labels/isCC"][:])
    NC_true = numpy.array(ff["labels/isNC"][:])
    tracklength_true = numpy.array(ff["labels/track_length"][:])
    
    weights = numpy.array(ff["weights"])

    dx_true, dy_true, dz_true = to_xyz(zenith_true, azimuth_true)

    print("Plotting input distributions")
    t_input_start = time.time()
    plot_inputs(pulse_time, pulse_charge, num_use=num_use, log_charge=True, gen_filename=save_folder_name)
    t_input_end = time.time()
    if (t_input_end-t_input_start)/60. > 1:
        print((t_input_end-t_input_start)/60., "minutes to plot inputs")
    else:
        print((t_input_end-t_input_start)/3600., "seconds to plot inputs")

    total_entries = len(weights)

    from scipy.stats import norm

    #Make plots
    print("Plotting regression output distributions")
    t_regression_start = time.time()
    if use_log_energy:
        plot_outputs(numpy.log10(energy_true), min(numpy.log10(energy_true)), max(numpy.log10(energy_true)), 'Energy [log10(E/GeV)]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    else:
        plot_outputs(energy_true, min(energy_true), max(energy_true), 'Energy [GeV]', weights, num_use=num_use, logscale=True, gen_filename=save_folder_name)
    plot_outputs(dx_true, -1.0, 1.0, 'dx [m]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    plot_outputs(dy_true, -1.0, 1.0, 'dy [m]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    plot_outputs(dz_true, -1.0, 1.0, 'dz [m]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    plot_outputs(azimuth_true*180/numpy.pi, 0, 360, 'Azimuth [degrees]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    plot_outputs(zenith_true*180/numpy.pi, 0, 180, 'Zenith [degrees]', weights, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    t_regression_end = time.time()
    if (t_regression_end-t_regression_start)/60. > 1:
        print((t_regression_end-t_regression_start)/60., "minutes to plot regression outputs")
    else:
        print((t_regression_end-t_regression_start)/3600., "seconds to plot regression outputs")

    print("Plotting classification output distributions")
    t_classification_start = time.time()
    plot_outputs_classify(track_true, cascade_true, energy_true, min(energy_true), max(energy_true), 'track', 'cascade', 'Energy [GeV]', ['Track','Cascade'], num_use=num_use, logscale=False, gen_filename=save_folder_name)
    plot_outputs_classify(CC_true, NC_true, energy_true, min(energy_true), max(energy_true), 'CC', 'NC', 'Energy [GeV]', ['CC','NC'], num_use=num_use, logscale=False, gen_filename=save_folder_name)
    t_classification_end = time.time()
    if (t_classification_end-t_classification_start)/60. > 1:
        print((t_classification_end-t_classification_start)/60., "minutes to plot classification outputs")
    else:
        print((t_classification_end-t_classification_start)/3600., "seconds to plot classification outputs")

    print("Plotting vertex locations")
    t_vertex_start = time.time()
    plot_vertex(x_true, y_true, z_true, num_use=num_use, gen_filename=save_folder_name)
    t_vertex_end = time.time()
    if (t_vertex_end-t_vertex_start)/60. > 1:
        print((t_vertex_end-t_vertex_start)/60., "minutes to plot vertex locations")
    else:
        print((t_vertex_end-t_vertex_start)/60., "seconds to plot vertex locations")

    print("Plotting hit information")
    t_hit_start = time.time()
    plot_hit_info(pulse_charge, pmt_index, energy_true, num_use=num_use, logscale=False, gen_filename=save_folder_name)
    t_hit_end = time.time()
    if (t_hit_end-t_hit_start)/60. > 1:
        print((t_hit_end-t_hit_start)/60., "minutes to plot hit information")
    else:
        print((t_hit_end-t_hit_start)/3600., "seconds to plot hit information")

    return 0#network_history.history['val_loss']

if __name__ == "__main__":
    main()
