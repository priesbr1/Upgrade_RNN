import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import h5py
import scipy

import keras
import tensorflow as tf
import keras.backend as K
from keras.utils.generic_utils import Progbar

class DataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self, hdf5_file, labels=None, maxlen=None, use_log_energy=False, use_weights=False, normal=[], batch_size=256):

        if normal == []:
            normal = dict()
            for k in labels:
                normal[k] = np.zeros(2)
                normal[k][1] = 1.0

        self.f = hdf5_file
        self.norm = normal
        self.num_total_entries = len(self.f["weights"])
        self.max_hits = maxlen
        self.use_log_energy = use_log_energy
        self.use_weights = use_weights

        self.num_batches = int(np.floor(self.num_total_entries / batch_size))
        self.batch_info = np.zeros(shape=(self.num_batches,2), dtype=np.int32)
        for i in range(self.num_batches):
            self.batch_info[i][0] = i*batch_size
            self.batch_info[i][1] = batch_size
        
        # the order in which we will look at the batches
        self.batch_order = np.array(range(self.num_batches))

        # shuffle but use the same seed every time (i.e. make this deterministic)
        #rng_state = np.random.get_state()
        #np.random.seed(0)
        #np.random.shuffle(self.batch_order)
        #np.random.set_state(rng_state) # re-set RNG state

        # create a sorted list of label names as they will appear in the output array
        self.labels = dict()
        if labels is None:
            self.label_keys = [k for k in self.f["labels"].keys()]
            for k in self.label_keys:
                self.labels[k] = self.f["labels"][k]
        else:
            available_keys = [k for k in self.f["labels"].keys()]
            
            self.label_keys = labels
            gen_dx = False
            gen_dy = False
            gen_dz = False
            for k in self.label_keys:
                if (k not in available_keys):
                    if   k == "dx": gen_dx = True
                    elif k == "dy": gen_dy = True
                    elif k == "dz": gen_dz = True
                    else:
                        raise RuntimeError("label {} not in hdf5 file".format(k))
                else: # k *is* in available_keys
                    self.labels[k] = self.f["labels"][k]
            
            if gen_dx or gen_dy or gen_dz:
                azi = None
                zen = None
                for k in available_keys:
                    if k == "azimuth": azi = self.f["labels"][k]
                    if k == "zenith":  zen = self.f["labels"][k]

                if (azi is None) or (zen is None):
                    raise RuntimeError("need to generate dx/dy/dz but azimuth and/or zenith labels are not in hdf5 file")
                
                dx, dy, dz = self._to_dxdydz(zen[:], azi[:])
                if gen_dx: self.labels["dx"] = dx
                if gen_dy: self.labels["dy"] = dy
                if gen_dz: self.labels["dz"] = dz

                del zen, azi
            
    def _to_dxdydz(self, zenith, azimuth):
        theta = np.pi-zenith
        phi = azimuth-np.pi
        rho = np.sin(theta)
        return rho*np.cos(phi), rho*np.sin(phi), np.cos(theta)
    
    def get_label_keys(self):
        return self.label_keys

    def untransform_labels(self, labels): #label_names=["azimuth", "entry_energy", "zenith"], label_scalers=label_scalers):
        ret_labels = np.copy(labels)
        for i, k in enumerate(self.label_keys):
            ret_labels[:,i] = ret_labels[:,i]*self.norm[k][1]+self.norm[k][0]
            if k == "energy" and self.use_log_energy:
                ret_labels[:,i] = 10**ret_labels[:,i]
        return ret_labels

    def get_reco(self, index):
        real_index = self.batch_order[index]
        batch_info = self.batch_info[real_index]

        batch_index = batch_info[0]
        batch_size = batch_info[1]

        # load reconstructed data
        reco_energy = self.f["reco/energy"][batch_index:batch_index+batch_size]
        reco_azimuth = self.f["reco/azimuth"][batch_index:batch_index+batch_size]
        reco_zenith = self.f["reco/zenith"][batch_index:batch_index+batch_size]

        # stack reco into one array
        reco = np.stack((reco_energy, reco_azimuth, reco_zenith), axis=0)

        return reco

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return self.num_batches

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        
        # get the pre-shuffled index corresponding to the requested index
        real_index = self.batch_order[index]
        batch_info = self.batch_info[real_index]

        batch_index = batch_info[0]
        batch_size = batch_info[1]
        
        # load weights
        if self.use_weights:
            weights = self.f["weights"][batch_index:batch_index+batch_size]
        else:
            weights = np.ones(batch_size)	    
    
        # load features and stack them into one array
        dom_index    = self.f["features/dom_index"][batch_index:batch_index+batch_size]
        pulse_time   = self.f["features/pulse_time"][batch_index:batch_index+batch_size]
        pulse_charge = self.f["features/pulse_charge"][batch_index:batch_index+batch_size]
        
        # zero-pad arrays to same length
        dom_index    = keras.preprocessing.sequence.pad_sequences(dom_index,    dtype=dom_index[0].dtype,    maxlen=self.max_hits)
        pulse_time   = keras.preprocessing.sequence.pad_sequences(pulse_time,   dtype=pulse_time[0].dtype,   maxlen=self.max_hits)
        pulse_charge = keras.preprocessing.sequence.pad_sequences(pulse_charge, dtype=pulse_charge[0].dtype, maxlen=self.max_hits)

        # re-scale features
        pulse_time   = pulse_time/20000.
        #pulse_charge = np.arctan(pulse_charge)/(np.pi/2.)
        
        # stack features into one array
        features = np.stack((dom_index, pulse_time, pulse_charge), axis=-1)

        # clean up
        del dom_index
        del pulse_time
        del pulse_charge
        
        # load labels and stack them
        labels = np.zeros(shape=(batch_size, len(self.label_keys)), dtype=np.float32)
        for i, k in enumerate(self.label_keys):
            if k == "energy" and self.use_log_energy:
                labels[:,i] = np.log10(self.labels[k][batch_index:batch_index+batch_size])
            else:
                labels[:,i] = self.labels[k][batch_index:batch_index+batch_size]
            labels[:,i] = (labels[:,i]-self.norm[k][0])/self.norm[k][1] 
        return features, labels, weights
    
class SplitGenerator(keras.utils.Sequence):
    def __init__(self, input_generator, fraction=1.0, offset=0.0):
        self.input_generator = input_generator
        
        self.num_batches = len(self.input_generator)
        
        # the order in which we will look at the batches
        self.batch_order = np.array(range(self.num_batches))

        self.start_offset = int(np.floor(self.num_batches * offset))
        self.num_batches_fraction = int(np.floor(self.num_batches * fraction))
        
        print("total batches:    {}".format(self.num_batches))
        print("starting batch:   {}".format(self.start_offset))
        print("num batches used: {}".format(self.num_batches_fraction))

    def get_reco(self, index):
        return self.input_generator.get_reco(index + self.start_offset)

    def __len__(self):
        return self.num_batches_fraction

    def __getitem__(self, index):
        real_index = self.batch_order[index + self.start_offset]
        
        return self.input_generator[real_index]
