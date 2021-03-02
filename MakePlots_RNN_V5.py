import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import numpy
import h5py
import scipy.stats
import glob
import os
import sys
import math
import argparse

from Generators_V5 import DataGenerator, SplitGenerator
from Attention import AttentionWithContext
from Plots import plot_uncertainty, plot_uncertainty_2d, plot_2dhist, plot_2dhist_contours, plot_1dhist, plot_error, plot_error_contours, plot_loss, plot_error_vs_reco

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
    return keras.losses.mean_absolute_percentage_error(y_true[:,0], y_pred[:,0])

def direction_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_true[:,1], y_pred[:,2]) + keras.losses.mean_squared_error(y_true[:,2], y_pred[:,3]) + keras.losses.mean_squared_error(y_true[:,3], y_pred[:,4])

#def classification_loss(y_true, y_pred):
#    return keras.losses.binary_crossentropy(y_true[:,4], y_pred[:,4]) + keras.losses.binary_crossentropy(y_true[:,5], y_pred[:,5])

def energy_uncertainty_loss(y_true, y_pred):
    return keras.losses.mean_absolute_percentage_error(y_pred[:,1], tf.stop_gradient(tf.math.abs(y_true[:,0]-y_pred[:,0])))

def direction_uncertainty_loss(y_true, y_pred):
    return keras.losses.mean_squared_error(y_pred[:,5], tf.stop_gradient(tf.math.abs(y_true[:,1]-y_pred[:,2]))) + keras.losses.mean_squared_error(y_pred[:,6], tf.stop_gradient(tf.math.abs(y_true[:,2]-y_pred[:,3]))) + keras.losses.mean_squared_error(y_pred[:,7], tf.stop_gradient(tf.math.abs(y_true[:,3]-y_pred[:,4])))

def customLoss(y_true, y_pred):
    e_loss = energy_loss(y_true, y_pred) + energy_uncertainty_loss(y_true, y_pred)
    d_loss = direction_loss(y_true, y_pred) + direction_uncertainty_loss(y_true, y_pred)
    loss = e_loss/700.0 + d_loss*8.0

    #loss = 0
    #for i in range(K.int_shape(y_pred)[1]):
    #    energy_dist = tf.distributions.Normal(loc=y_pred[i,0], scale=y_pred[i,1])
    #    dx_dist = tf.distributions.Normal(loc=y_pred[i,2], scale=y_pred[i,5])
    #    dy_dist = tf.distributions.Normal(loc=y_pred[i,3], scale=y_pred[i,6])
    #    dz_dist = tf.distributions.Normal(loc=y_pred[i,4], scale=y_pred[i,7])
    #    loss += tf.reduce_mean(-dx_dist.log_prob(y_true[i,1])) + tf.reduce_mean(-dy_dist.log_prob(y_true[i,2])) + tf.reduce_mean(-dz_dist.log_prob(y_true[i,3])) #+ tf.reduce_mean(-energy_dist.log_prob(y_true[i,0])) 
    return loss

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
    parser.add_argument("-h", "--hits",type=int,default=150, dest="hits", help="number of dom hits used for training")
    parser.add_argument("-e", "--epochs",type=int,default=30, dest="epochs", help="number of training epochs")
    parser.add_argument("-d", "--decay",type=float,default=0.0, dest="decay", help="learning rate decay parameter")
    parser.add_argument("-r", "--lr", type=float,default=0.001, dest="lr", help="learning rate")
    parser.add_argument("-o", "--dropout", type=float,default=0.1, dest="dropout", help="change network dropout for each layer")
    parser.add_argument("-l", "--log_energy", type=int,default=0, dest="log_energy", help="use log energy rather than absolute for training")
    parser.add_argument("-f", "--file", type=str, default="outfile_l5p_le.hdf5", dest="file_name", help="file to use for training")
    parser.add_argument("-p", "--path", type=str, default="/mnt/scratch/priesbr1/Data_Files/", dest="path", help="path to input file")
    parser.add_argument("-u", "--output", type=str, default="/mnt/scratch/priesbr1/Upgrade_RNN/", dest="output", help="output folder destination")
    parser.add_argument("-s", "--standardize", type=int,default=0, dest="standardize", help="perform data standardization")
    parser.add_argument("-c", "--checkpoints", type=int,default=0, dest="checkpoints", help="use training checkpoints from previous run")
    parser.add_argument("-w", "--weights", type=int,default=1, dest="weights", help="use sample weights for training")
    parser.add_argument("-t", "--data_type", type=str,default=None, dest="data_type", help="type of data used to train RNN")
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

    ff = h5py.File(ff_name, 'r')
    global gen_filename
    global save_folder_name
    gen_filename = "run_"+str(no_epochs)+"_epochs_"+args.data_type+"_energyMAPE_lr"+str(int(numpy.log10(learning_rate)))+"_"+str(int(no_hits))+"hits"
    save_folder_name = args.output + gen_filename + '/'
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
    print("Saving to:", save_folder_name)

    reco = False
    if "reco" in ff.keys(): reco = True

    network_labels = ['energy', 'dx', 'dy', 'dz']#, 'isTrack', 'isCascade']
    if use_standardization: normalization = normalize(ff, network_labels, use_log_energy)
    else: normalization = []
    gen = DataGenerator(ff, labels=network_labels, maxlen=no_hits, use_log_energy=use_log_energy,use_weights=use_weights,normal=normalization)
    gen_train = SplitGenerator(gen, fraction=0.70, offset=0.00)
    gen_val = SplitGenerator(gen, fraction=0.10, offset=0.70)
    gen_test = SplitGenerator(gen, fraction=0.20, offset=0.80)

    vocab_size = 15700
    time_samples = no_hits

    # Instantiate the base model (or "template" model).
    # We recommend doing this with under a CPU device scope,
    # so that the model's weights are hosted on CPU memory.
    # Otherwise they may end up hosted on a GPU, which would
    # complicate weight sharing.
    input_data = Input(shape=(time_samples,3), name="input_data") # variable length
    input_pmt_index = Lambda( lambda x: x[:,:,0], name="input_pmt_index" )(input_data) # slice out the dom index
    input_rel_time  = Reshape( (-1,1), name="reshaped_rel_time" )(Lambda( lambda x: x[:,:,1], name="input_rel_time" )(input_data))  # slice out the relative time
    input_charge    = Reshape( (-1,1), name="reshaped_charge" )(Lambda( lambda x: x[:,:,2], name="input_charge" )(input_data))    # slice out the charge

    geometry_file = h5py.File("/mnt/scratch/priesbr1/Processed_Files/geometry_upgrade.hdf5",'r') ########
    pmt_positions = numpy.array(geometry_file['positions'][:]) ########
    pmt_directions = numpy.array(geometry_file['directions'][:]) ########
    
    embedding_pmt_index = Embedding(input_dim=vocab_size,
                                    output_dim=5,
                                    input_length=time_samples,
                                    #weights=numpy.concatenate((pmt_positions,pmt_directions),axis=0),trainable=False,
                                    # mask_zero=True,
                                    name="embedding_pmt_index")(input_pmt_index)
    x = Concatenate(axis=-1, name="concatenated_features")([embedding_pmt_index, input_rel_time, input_charge])
    
    x = CuDNNLSTM(128, return_sequences=True, name='lstm1')(x)
    x = Dropout(dropout)(x)
    x = CuDNNLSTM(128, return_sequences=True, name='lstm2')(x)
    x = Dropout(dropout)(x)
    x = CuDNNLSTM(128, return_sequences=True, name='lstm3')(x)
    x = Dropout(dropout)(x)
 
    x = AttentionWithContext(name="attention")(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    dense_regression = Dense(64, activation='relu')(x)
    dense_regression = Dropout(dropout)(dense_regression)
    #dense_classification = Dense(64, activation='tanh')(x)
    #dense_classification = Dropout(dropout)(dense_classification)
    
    dense_energy = Dense( 1, activation='linear', name="dense_energy")(dense_regression) # range -inf..inf
    dense_energy_sig = Dense(1, activation=lambda x: tf.nn.elu(x)+1, name="dense_energy_sig")(dense_regression)
    dense_dxdydz = Dense( 3, activation='linear',   name="dense_dxdydz")(dense_regression) # range   -1..1
    dense_dxdydz_sig = Dense(3, activation=lambda x: tf.nn.elu(x)+1, name="dense_dxdydz_sig")(dense_regression)
    #dense_tc = Dense(2, activation='sigmoid', name='dense_tc')(dense_classification)
    outputs = Concatenate(axis=-1, name="output")([dense_energy, dense_energy_sig, dense_dxdydz, dense_dxdydz_sig])#dense_tc])
    
    model = Model(inputs=input_data, outputs=outputs)

    opt = keras.optimizers.Adamax(lr=learning_rate,decay=decay)#keras.optimizers.SGD(lr=0.01,momentum=0.8)

    model.compile(optimizer=opt, loss=customLoss, metrics=[energy_loss, direction_loss, energy_uncertainty_loss, direction_uncertainty_loss])
    
    model.summary()

    # get all files
    checkpoint_files = glob.glob("%sweights.?????.hdf5"%save_folder_name)
    checkpoint_files.sort()
 
    if len(checkpoint_files) == 0:
        print("no checkpoints available, starting from scratch.")
        initial_epoch = 0
    elif not use_checkpoints:
        print("checkpoints not used, starting from scratch.")
        initial_epoch = 0
    else:
        indices = []
        for i in range(len(checkpoint_files)):
            # strip the path
            _, filename = os.path.split(checkpoint_files[i])
            if int(filename[8:8+5]) <= no_epochs:
                print(filename)
                indices.append( int(filename[8:8+5]) )
    
        indices = numpy.array(indices)
        sorting = numpy.argsort(indices)
        last_checkpoint = checkpoint_files[ sorting[-1] ]
        last_checkpoint_epoch = indices[ sorting[-1] ]
        initial_epoch = last_checkpoint_epoch
 
        print("Loading epoch {} from checkpoint file {}".format( last_checkpoint_epoch, last_checkpoint ))
        model.load_weights(last_checkpoint)
    
        gen_train, gen_val = forward_generators(gen_train, gen_val, last_checkpoint_epoch)

    if initial_epoch == no_epochs:
        train = False
    else:
        train = True
 
    print("Initial epoch index is {}".format(initial_epoch))
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        "weights.{epoch:05d}.hdf5", 
        monitor='val_loss', 
        save_weights_only=True)
    
    def schedule_function(epoch,lr):
        U = 4
        L = -4
        lrmin = 0.005
        lrmax = 0.05
        #return lrmin+((lrmax-lrmin)/scipy.stats.norm(25,5).pdf(25))*scipy.stats.norm(25,5).pdf(epoch)
        return lrmin+0.5*(lrmax-lrmin)*(1-numpy.tanh(L*(1-epoch/50.0)+U*epoch/50.0))
    
    lr_schedule = keras.callbacks.LearningRateScheduler(schedule_function)
 
    if train:
        network_history = model.fit_generator(
            generator=gen_train,
            steps_per_epoch=len(gen_train),
            validation_data=gen_val,
            validation_steps=len(gen_val),
            epochs=no_epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            shuffle=True,
            workers=1,
            use_multiprocessing=False,
            callbacks=[model_checkpoint])#, lr_schedule])
    
        weightfile_name = save_folder_name+'weightfile.hdf5'
        model.save_weights(weightfile_name)
        model.load_weights(weightfile_name)
    
    labels_raw = None
    labels_predicted_raw = None
    if reco: labels_reco = None
    weights_raw = None
   
    if train: 
        test_metrics = model.evaluate_generator(gen_test)
        train_metrics = model.evaluate_generator(gen_train)
        val_metrics = model.evaluate_generator(gen_val)

    print("Testing model")

    for i in range(len(gen_test)-1):
        batch_features, batch_labels, batch_weights = gen_test[i]
        if reco: batch_reco = gen_test.get_reco(i)
        batch_labels_predicted = model.predict(batch_features)

        if labels_raw is None:
            labels_raw = batch_labels
            labels_predicted_raw = batch_labels_predicted
            if reco: labels_reco = batch_reco
            weights_raw = batch_weights
        else:
            labels_raw           = numpy.append(labels_raw,           batch_labels,           axis=0)
            labels_predicted_raw = numpy.append(labels_predicted_raw, batch_labels_predicted, axis=0)
            if reco: labels_reco = numpy.append(labels_reco, batch_reco, axis=-1)
            weights_raw          = numpy.append(weights_raw,          batch_weights,          axis=0)
        del batch_labels_predicted
        del batch_features
        del batch_labels
        if reco: del batch_reco
        del batch_weights
 
    labels_predicted = labels_predicted_raw#gen.untransform_labels(labels_predicted_raw)
    labels = labels_raw#gen.untransform_labels(labels_raw)
    weights = weights_raw

    energy_predicted = labels_predicted[:,0]
    energy_true = labels[:,0]
    energy_sigma = labels_predicted[:,1]
    dx_predicted = labels_predicted[:,2]
    dx_true = labels[:,1]
    dx_sigma = labels_predicted[:,5]
    dy_predicted = labels_predicted[:,3]
    dy_true = labels[:,2]
    dy_sigma = labels_predicted[:,6]
    dz_predicted = labels_predicted[:,4]
    dz_true = labels[:,3]
    dz_sigma = labels_predicted[:,7]

    if reco:
        energy_reco = labels_reco[0]
        azimuth_reco = (180.0/numpy.pi)*numpy.array(labels_reco[1])
        zenith_reco = (180.0/numpy.pi)*numpy.array(labels_reco[2])

    from scipy.stats import norm

    if train:
        plot_loss(network_history.history, test_metrics[0], 'loss', "Loss", no_epochs, gen_filename=save_folder_name, unc=False)
        plot_loss(network_history.history, [test_metrics[1],test_metrics[3]], ['energy_loss','energy_uncertainty_loss'], "Energy", no_epochs, gen_filename=save_folder_name, unc=True)
        plot_loss(network_history.history, [test_metrics[2],test_metrics[4]], ['direction_loss','direction_uncertainty_loss'], "Direction", no_epochs, gen_filename=save_folder_name, unc=True)
 
    #isTrack_predicted = labels_predicted[:,4]
    #isCascade_predicted = labels_predicted[:,5]
    #isTrack_true = labels[:,4]
    #isCascade_true = labels[:,5]
    
    #isTrack_predicted = [isTrack_predicted > isCascade_predicted]
    #isCascade_predicted = [isCascade_predicted > isTrack_predicted]
    
    #trueTracks = numpy.sum(numpy.logical_and(isTrack_true, isTrack_predicted))
    #falseTracks = numpy.sum(numpy.logical_and(numpy.logical_not(isTrack_true), isTrack_predicted))
    #trueCascades = numpy.sum(numpy.logical_and(isCascade_true, isCascade_predicted))
    #falseCascades = numpy.sum(numpy.logical_and(numpy.logical_not(isCascade_true), isCascade_predicted))
    
    #fig, ax = plt.subplots()
    #bars1 = ax.bar(numpy.arange(2), [trueTracks, trueCascades], 0.25, color='SkyBlue')
    #bars2 = ax.bar(numpy.arange(2)+0.5*numpy.ones(2), [falseTracks, falseCascades], 0.25, color='IndianRed')
    #ax.set_title('Track vs. Cascade classification results')
    #ax.set_xticks(numpy.arange(4)/2)
    #ax.set_xticklabels(('True Tracks', 'False Tracks', 'True Cascades', 'False Cascades'))
    #imgname = save_folder_name+'class.png'
    #plt.savefig(imgname)
    
    zenith_predicted, azimuth_predicted = numpy.degrees(to_zenazi(dx_predicted, dy_predicted, dz_predicted))
    zenith_true, azimuth_true = numpy.degrees(to_zenazi(dx_true, dy_true, dz_true))

    r_predicted = numpy.sqrt(dx_predicted**2+dy_predicted**2+dz_predicted**2)
    r_sigma = numpy.sqrt(numpy.divide((dx_predicted*dx_sigma)**2+(dy_predicted*dy_sigma)**2+(dz_predicted*dz_sigma)**2,r_predicted**2))
    zenith_sigma = numpy.degrees(numpy.sqrt(numpy.divide((dz_predicted*r_sigma)**2+(r_predicted*dz_sigma)**2,r_predicted**2*(r_predicted**2-dz_predicted**2))))
    azimuth_sigma = numpy.degrees(numpy.sqrt(numpy.divide((dx_sigma*dy_predicted)**2+(dy_sigma*dx_predicted)**2,(dx_predicted**2+dy_predicted**2)**2)))

    #Make plots
    if use_log_energy:
        plot_2dhist(numpy.log10(energy_true), numpy.log10(energy_predicted), min(numpy.log10(energy_true)), max(numpy.log10(energy_true)), 'Energy [log10(E/GeV)]', weights, gen_filename=save_folder_name)
        plot_2dhist_contours(numpy.log10(energy_true), numpy.log10(energy_predicted), min(numpy.log10(energy_true)), max(numpy.log10(energy_true)), 'Energy [log10(E/GeV)]', weights, gen_filename=save_folder_name)
        plot_1dhist(numpy.log10(energy_true), numpy.log10(energy_predicted), min(numpy.log10(energy_true)), max(numpy.log10(energy_true)), 'Energy [log10(E/GeV)]', weights, gen_filename=save_folder_name)
    else:
        plot_2dhist(energy_true, energy_predicted, min(energy_true), max(energy_true), 'Energy [GeV]', weights, gen_filename=save_folder_name)
        plot_2dhist_contours(energy_true, energy_predicted, min(energy_true), max(energy_true), 'Energy [GeV]', weights, gen_filename=save_folder_name)
        plot_1dhist(energy_true, energy_predicted, min(energy_true), max(energy_true), 'Energy [GeV]', weights, gen_filename=save_folder_name)
    
    if reco:
        plot_error_vs_reco(energy_true, energy_predicted, energy_reco, min(energy_true), max(energy_true), 'Energy [GeV]', gen_filename=save_folder_name)
        plot_error_vs_reco(azimuth_true, azimuth_predicted, azimuth_reco, 0, 360, 'Azimuth [degrees]', gen_filename=save_folder_name)
        plot_error_vs_reco(zenith_true, zenith_predicted, zenith_reco, 0, 180, 'Zenith [degrees]', gen_filename=save_folder_name)

        plot_error_vs_reco(azimuth_true, azimuth_predicted, azimuth_reco, min(energy_true), max(energy_true), 'Azimuth [degrees]', quantity2='Energy [GeV]', x=labels[:,0], gen_filename=save_folder_name)
        plot_error_vs_reco(zenith_true, zenith_predicted, zenith_reco, min(energy_true), max(energy_true), 'Zenith [degrees]', quantity2='Energy [GeV]', x=labels[:,0], gen_filename=save_folder_name)
    else:
        plot_error(energy_true, energy_predicted, min(energy_true), max(energy_true), 'Energy [GeV]', gen_filename=save_folder_name)
        plot_error(azimuth_true, azimuth_predicted, 0, 360, 'Azimuth [degrees]', gen_filename=save_folder_name)
        plot_error(zenith_true, zenith_predicted, 0, 180, 'Zenith [degrees]', gen_filename=save_folder_name)
        plot_error(azimuth_true, azimuth_predicted, min(energy_true), max(energy_true), 'Azimuth [degrees]', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)
        plot_error(zenith_true, zenith_predicted, min(energy_true), max(energy_true), 'Zenith [degrees]', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)

    plot_2dhist(dx_true, dx_predicted, -1.0, 1.0, 'dx [m]', weights, gen_filename=save_folder_name)
    plot_2dhist(dy_true, dy_predicted, -1.0, 1.0, 'dy [m]', weights, gen_filename=save_folder_name)
    plot_2dhist(dz_true, dz_predicted, -1.0, 1.0, 'dz [m]', weights, gen_filename=save_folder_name)
    plot_2dhist(azimuth_true, azimuth_predicted, 0, 360, 'Azimuth [degrees]', weights, gen_filename=save_folder_name)
    plot_2dhist(zenith_true, zenith_predicted, 0, 180, 'Zenith [degrees]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dx_true, dx_predicted, -1.0, 1.0, 'dx [m]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dy_true, dy_predicted, -1.0, 1.0, 'dy [m]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(dz_true, dz_predicted, -1.0, 1.0, 'dz [m]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(azimuth_true, azimuth_predicted, 0, 360, 'Azimuth [degrees]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(zenith_true, zenith_predicted, 0, 180, 'Zenith [degrees]', weights, gen_filename=save_folder_name)
    plot_2dhist_contours(numpy.cos(zenith_true*numpy.pi/180), numpy.cos(zenith_predicted*numpy.pi/180), -1, 1, 'Cos(Zenith)', weights, gen_filename=save_folder_name)
    plot_1dhist(dx_true, dx_predicted, -1.0, 1.0, 'dx [m]', weights, gen_filename=save_folder_name)
    plot_1dhist(dy_true, dy_predicted, -1.0, 1.0, 'dy [m]', weights, gen_filename=save_folder_name)
    plot_1dhist(dz_true, dz_predicted, -1.0, 1.0, 'dz [m]', weights, gen_filename=save_folder_name)
    plot_1dhist(azimuth_true, azimuth_predicted, 0, 360, 'Azimuth [degrees]', weights, gen_filename=save_folder_name)
    plot_1dhist(zenith_true, zenith_predicted, 0, 180, 'Zenith [degrees]', weights, gen_filename=save_folder_name)
    plot_error(dx_true, dx_predicted, -1.0, 1.0, 'dx [m]', gen_filename=save_folder_name)
    plot_error(dy_true, dy_predicted, -1.0, 1.0, 'dy [m]', gen_filename=save_folder_name)
    plot_error(dz_true, dz_predicted, -1.0, 1.0, 'dz [m]', gen_filename=save_folder_name)
    plot_error(dx_true, dx_predicted, min(energy_true), max(energy_true), 'dx [m]', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)
    plot_error(dy_true, dy_predicted, min(energy_true), max(energy_true), 'dy [m]', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)
    plot_error(dz_true, dz_predicted, min(energy_true), max(energy_true), 'dz [m]', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)
    plot_error_contours(energy_true, energy_predicted, 0, 100, 'Energy [GeV]', gen_filename=save_folder_name)
    plot_error_contours(numpy.cos(zenith_true*numpy.pi/180), numpy.cos(zenith_predicted*numpy.pi/180), -1, 1, 'Cos(Zenith)', gen_filename=save_folder_name)
    plot_error_contours(numpy.cos(zenith_true*numpy.pi/180), numpy.cos(zenith_predicted*numpy.pi/180), 0, 100, 'Cos(Zenith)', 'Energy [GeV]', energy_true, gen_filename=save_folder_name)
    plot_uncertainty(energy_true, energy_predicted, energy_sigma, 'Energy [GeV]', weights, gen_filename=save_folder_name)
    plot_uncertainty(dx_true, dx_predicted, dx_sigma, 'dx [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty(dy_true, dy_predicted, dy_sigma, 'dy [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty(dz_true, dz_predicted, dz_sigma, 'dz [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty(azimuth_true, azimuth_predicted, azimuth_sigma, 'Azimuth [degrees]', weights, gen_filename=save_folder_name)
    plot_uncertainty(zenith_true, zenith_predicted, zenith_sigma, 'Zenith [degrees]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(energy_true, energy_predicted, energy_sigma, 'Energy [GeV]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(dx_true, dx_predicted, dx_sigma, 'dx [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(dy_true, dy_predicted, dy_sigma, 'dy [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(dz_true, dz_predicted, dz_sigma, 'dz [m]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(azimuth_true, azimuth_predicted, azimuth_sigma, 'Azimuth [degrees]', weights, gen_filename=save_folder_name)
    plot_uncertainty_2d(zenith_true, zenith_predicted, zenith_sigma, 'Zenith [degrees]', weights, gen_filename=save_folder_name)

    #output results
    print("DIAGNOSTICS")
    if reco:
        zen_RNN_err = numpy.absolute(zenith_true[zenith_reco > 0] - zenith_predicted[zenith_reco > 0])
        zen_PL_err = numpy.absolute(zenith_true[zenith_reco > 0] - zenith_reco[zenith_reco > 0])
        eng_RNN_err = numpy.absolute(numpy.divide(energy_true[energy_reco > 0] - energy_predicted[energy_reco > 0],energy_true[energy_reco > 0]))
        eng_PL_err = numpy.absolute(numpy.divide(energy_true[energy_reco > 0] - energy_reco[energy_reco > 0],energy_true[energy_reco > 0]))
        azi_RNN_err = numpy.absolute(azimuth_true[azimuth_reco > 0] - azimuth_predicted[azimuth_reco > 0])
        azi_PL_err = numpy.absolute(azimuth_true[azimuth_reco > 0] - azimuth_reco[azimuth_reco > 0])
        azi_PL_err = numpy.array([azi_PL_err[i] if (azi_PL_err[i] < 180) else (360-azi_PL_err[i]) for i in range(len(azi_PL_err))])
        azi_PL_err = numpy.array([azi_PL_err[i] if (azi_PL_err[i] > -180) else (360+azi_PL_err[i]) for i in range(len(azi_PL_err))])
        avg_zen_PL_err = numpy.mean(zen_PL_err)
        avg_eng_PL_err = numpy.mean(eng_PL_err)
        avg_azi_PL_err = numpy.mean(azi_PL_err)
        std_zen_PL_err = numpy.std(zen_PL_err)
        std_eng_PL_err = numpy.std(eng_PL_err)
        std_azi_PL_err = numpy.std(azi_PL_err)
        print("PegLeg")
        print("Energy: average fractional error = "+str(avg_eng_PL_err)+", sigma = "+str(std_eng_PL_err))
        print("Azimuth: average absolute error = "+str(avg_azi_PL_err)+", sigma = "+str(std_azi_PL_err))
        print("Zenith: average absolute error = "+str(avg_zen_PL_err)+", sigma = "+str(std_zen_PL_err))
    else:
        zen_RNN_err = numpy.absolute(zenith_true - zenith_predicted)
        eng_RNN_err = numpy.absolute(energy_true - energy_predicted)
        azi_RNN_err = numpy.absolute(azimuth_true - azimuth_predicted)
    azi_RNN_err = numpy.array([azi_RNN_err[i] if (azi_RNN_err[i] < 180) else (360-azi_RNN_err[i]) for i in range(len(azi_RNN_err))])
    azi_RNN_err = numpy.array([azi_RNN_err[i] if (azi_RNN_err[i] > -180) else (360+azi_RNN_err[i]) for i in range(len(azi_RNN_err))])
    avg_zen_RNN_err = numpy.mean(zen_RNN_err)
    avg_eng_RNN_err = numpy.mean(eng_RNN_err)
    avg_azi_RNN_err = numpy.mean(azi_RNN_err)
    std_zen_RNN_err = numpy.std(zen_RNN_err)
    std_eng_RNN_err = numpy.std(eng_RNN_err)
    std_azi_RNN_err = numpy.std(azi_RNN_err)
    print("RNN")
    print("Energy: average fractional error = "+str(avg_eng_RNN_err)+", sigma = "+str(std_eng_RNN_err))
    print("Azimuth: average absolute error = "+str(avg_azi_RNN_err)+", sigma = "+str(std_azi_RNN_err))
    print("Zenith: average absolute error = "+str(avg_zen_RNN_err)+", sigma = "+str(std_zen_RNN_err))

    return 0#network_history.history['val_loss']

if __name__ == "__main__":
    main()
