# Upgrade_RNN

## Repository for IceCube-Upgrade RNN

### Background

This repository is for a Recurrent Neural Network (RNN) applied to simulated `i3` data from the IceCube detector at the South Pole. The detector currently consists of 5,160 Digital Optical Modules (DOMs), each with one photomultiplier tube (PMT), situated on 86 strings between 1.5-2.5 km below the surface. The DOMs collect light emitted when a neutrino interacts with a nucleus in the ice, called an event. Depending on when, where, and how much light is detected, we can reconstruct properties of the neutrino like energy, zenith (vertical angle), and azimuth (horizontal angle), among other things. Each of these detections is called a hit.

### Problem

When neutrinos have a very high energy (usually at least 1 TeV), they emit a lot of light in the detector, making it much easier to reconstruct its energy, direction, event type, etc. However, this becomes much more difficult to do at lower energies because those events produce less light, so we have less information to work with. The central area of IceCube, known as DeepCore, has a higher instrument density than the rest of the detector. DeepCore is used to probe the low-energy regime of neutrino physics, typically on the scale of 10 GeV, but even that is not enough to consistently create accurate reconstructions of the lowest-energy events. This is where the IceCube-Upgrade comes in. The Upgrade is deploying 7 new stings, each with approximately 100 DOMs, into the DeepCore area. The Upgrade will also make use of two new DOM designs: the D-Egg with two PMTs (one on top, one on bottom), and the mDOM with 24 PMTs scattered over its surface. In total, Gen-1 and the Upgrade will have 15,700 PMTs. With these new DOMs, they hope to improve energy reconstruction of low-energy events to the scale of 1 GeV, as well as improve directional reconstruction.

### i3 Data

IceCube uses a specific data type referred to as `i3`. This data type is specific to IceCube and can be accessed using a module called `I3Tray`/`IceTray`. This module stores information about the event, as well as detector status and detector geometry.

### RNN

The RNN is neural network designed to handle data with a sequential (e.g. temporal) relationship, which is great for IceCube. The RNN takes in three input variables per event: a list of times when light was detected, a list of charges (proprotional to how much light was detected), and a list of generated IDs that describe which PMTs were triggered. Each of the corresponding entries in these lists (e.g. t_1, q_1, p_1) would comprise one hit, and all three lists comprise one event. The RNN outputs energy, dx/dy/dz direction, and error estimates for all four.

### Files

`Attention.py`: Defines attention layer for RNN. No parsed inputs.

`combine_hdf5.py`: Takes data in .hdf5 form and combines multiple files into a single file. Parsed inputs:
* input_files: path/name of the input file(s) to be combined (can use ? and * as wildcard fillers)
* output_files: path/name of the output file after combination
* export_geometry.py: Initializes and outputs Gen-1 detector geometry. No parsed inputs.

`export_geometry_V5.py`: Initialzes and outputs Upgrade detector geometry. No parsed inputs.

`filter_hdf5.py`: Applies cuts to .hdf5 data files and shuffles entries. Parsed inputs:
* input_file: path/name of the input file that cuts are being applied to
* base_name: starting point for the name of the output file (should be simple description of data, e.g. 'UpgradeNuMu')
* energy_range: two-element list containing the minimum and maximum energy values for the output data
* keep_track: whether or not to keep track events
* keep_cascade: whether or not to keep cascade events
* keep_CC: whether or not to keep CC (charged current) events
* keep_NC: whether or not to keep NC (neutral current) events
* vertex_cut: the type of vertex cut to apply
* flat_energy: whether or not to flatten the energy distribution
* flat_tc: whether or not to equalize the number of tracks/cascades
* min_PMT: minimum number of triggered PMTs required for an event to pass
* max_events: after all cuts, limit output file to specific number of events
* only_reco: only keep events with PegLeg (likelihood-based) reconstructions

`Generators.py`: Sets up generators used to process large amounts of Gen-1 data for RNN. No parsed inputs.

`Generators_V5.py`: Sets up generators used to process large amounts of Upgrade data for RNN. No parsed inputs.

`i3_to_hdf5.py`: Processes information from Gen-1 .i3 files (or variants) into features/labels for the RNN, stored in .hdf5 format. Parsed inputs:
* file: path/name of the input file(s) to be processed (can use ? and * as wildcard fillers)
* overwrite: whether or not to overwrite previous output files
* pulse_type: type of pulseseries to use (cleaned or uncleaned)

`i3_to_hdf5_V5.py`: Processes information from Upgrade .i3 files (or variants) into features/labels for the RNN, stored in .hdf5 format. Parsed inputs:
* file: path/name of the input file(s) to be processed (can use ? and * as wildcard fillers)
* overwrite: whether or not to overwrite previous output files
* pulse_type: type of pulseseries to use (cleaned or uncleaned)

`MakePlots_RNN_V4.py`: Plots results from finished RNNs trained on Gen-1 data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training

`MakePlots_RNN_V5.py`: Plots results from finished RNNs trained on Upgrade data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* data_type: description of data used (should describe sample and cuts)

`plot_inputoutput_datafile.py`: Plots distributions of input/output variables from Gen-1 data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* num_use: maximum number of events to use for plotting

`plot_inputoutput_datafile_V5.py`: Plots distributions of input/output variables from Upgrade data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* num_use: maximum number of events to use for plotting

`plot_inputoutput_RNN.py`: Plots distributions of input/output variables for an RNN trained on Gen-1 data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* num_use: maximum number of events to use for plotting

`plot_inputoutput_RNN_V5.py`: Plots distributions of input/output variables for an RNN trained on Upgrade data. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* num_use: maximum number of events to use for plotting

`Plots.py`: Contains plotting functions used to plot variable distibutions and RNN results. No parsed inputs.

`plot_trigger_resolution.py`: Plots distributions of pulses for events with DeepCore SMT3 trigger relative to the event trigger time. Parsed inputs:
* input_files: path/name of input file(s) to process for plotting
* output_folder: path/name of the output folder where the plot will be saved
* pulse_type: type of pulseseries to use (cleaned or uncleaned)
* logscale: whether or not to use logscale for plotting
* num_use: maximum number of events to use for plotting

`RNN_V4_2.py`: Initializes, trains, and tests an RNN on Gen-1 data, and plots results. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training

`RNN_V5.py`: Initializes, trains, and tests an RNN on Upgrade data, and plots results. Parsed inputs:
* hits: maximum number of hits per event that the RNN utilizes
* epochs: total number of trained epochs
* decay: rate of learning rate decay
* lr: inital learning rate
* dropout: fraction of nodes whose weights will be dropped per epoch to prevent overtraining
* log_energy: whether or not to use log_10 of the energy for training/testing
* file: name of the input file used for training
* path: path of the input file used for training
* output: path of the output file for results
* standardize: whether or not to standardize data
* checkpoints: whether or not to use checkpoints from a previous training
* weights: whether or not to use sample weights for training
* data_type: description of data used (should describe sample and cuts)

### Data Processing Pipeline

`i3_to_hdf5.py` (Gen-1 data):
It first loads in an `.i3` file (or variant), then sets up dictionaries for features, labels, and reco (LLH-based reco to compare to RNN results). It goes through every file in the list provided (or the single file if only one), then gets the information from every Physics frame in the file. It then grabs a cleaned/uncleaned pulseseries depending on the object provided. It saves the energy, zenith, azimuth, and x/y/z vertex position, as well as the LLH reco energy, zenith, and azimuth. It also saves the interaction type (CC/NC), event type (track/cascade), and the event weight. Then it creates lists for the dom_index, pulse_time, and pulse_charge. After that, it goes through every event in the pulseseries and saves a manually-generated DOM ID, and goes through every pulse in the event and saves the time and charge. It then shifts the time so that the average time for the entire dataset is 0. Lastly, it writes the ouput `.hdf5` file using the same name as the input `.i3` file.

`i3_to_hdf5_V5.py` (Upgrade data):
It first loads in an `.i3` file (or variant), then sets up dictionaries for features, labels, and reco (LLH-based reco to compare to RNN results). It goes through every file in the list provided (or the single file if only one), then gets the information from every Physics frame in the file. It then grabs a cleaned/uncleaned pulseseries depending on the object provided. It saves the energy, zenith, azimuth, and x/y/z vertex position, as well as the LLH reco energy, zenith, and azimuth. It also saves the interaction type (CC/NC), event type (track/cascade, checking that the event is a neutrino interaction), and the event weight. Then it creates lists for the pmt_index, pulse_time, and pulse_charge. After that, it goes through every event in the pulseseries and saves a manually-generated PMT ID, and goes through every pulse in the event and saves the time and charge. It then shifts the time so that the average time for the entire dataset is 0. Lastly, it writes the ouput `.hdf5` file using the same name as the input `.i3` file.

`combine_hdf5.py`:
It first loads the input files and checks that they all exist and contain information. Next, it creates empty features/labels/reco dictionaries to match the data format of the input files. It then goes through all the non-empty input files and concatenates the information to the corresponding places in the output file dictionaries. As this happens, it also saves random selctions of some of the variables to check that they were properly stored. Lastly, it checks that these random portions of data are both correctly loaded from the input files and correctly saved in the output file.

`filter_hdf5.py`:
It first prints out the selected cuts as a double-check that the right ones are being applied. It loads the information from the input file as NumPy arrays into separate dictionaries for easier manipulation. It then automatically names the output file based on the selected cuts, checking that certain cuts make sense, and shuffles all the data before beginning cuts. It applies cuts in the following order:
* PMT, vertex, energy range, track/cascade, CC/NC, reco, energy flattening, track/cascade flattening, max events
After cuts, the data is shuffled again and it prints out the number of events before/after cuts. Lastly, it loads the data back into an `.hdf5` file and saves it.

### Recommended Processing Order

If final file will be small (roughly less than 30GB):
* `i3_to_hdf5`/`i3_to_hdf5_V5` all individual files
* `combine_hdf5` all individual files into full file
* `filter_hdf5` to apply cuts to full file

If final file will be large (roughly at least 30GB):
* `i3_to_hdf5`/`i3_to_hdf5_V5` all individual files
* `combine_hdf5` all individual files into multiple files/sets
* `filter_hdf5` each set to limit the file size earlier (do not run flattening)
* `combine_hdf5` all filtered sets into full file
* `filter_hdf5` full file with same cuts (and flattening, if desired) for thorough shuffling

### Running the RNN

Before loading in the data, the RNN code imports:
* Common Python imports like `pyplot`, `NumPy`, `SciPy`, and `math`
* Modules like `glob` and `h5py` to handle files
* Modules like `os` and `argparse` to handle running on the HPCC (MSU's Linux-based supercomputer)
* The generators from `Generators.py` (or `Generators_V5.py` for Upgrade simulation), and the attention layer from `Attention.py`
* Plotting functions from `Plots.py`
* `Keras`/`TensorFlow` modules for creating/running the RNN

The RNN also:
* defines a normaliztion function for data standardization
* defines the loss functions for energy/direction and their uncertainties
* defines functions for converting between zenith/azimuth and x/y/z
* a function to "fast forward" the generators to a certain epoch

It parses all arguments and loads in the datafile, creating a folder to save the results to (or checking that it already exists). It then splits the data into three different sets (70% train, 10% validate, 20% test) using the generators. After that, it begins constructing the model with the input layer. The embedding layer can be used to pass the DOMs' x/y/z position (or PMTs' x/y/z position and zenith/azimuth angle, in the case of Upgrade simulation) as weights to the RNN. These can also be initalized randomly, and random/non-random initialization showed little difference. It finishes building the model with three LSTM layers, the attention layer, two dense layers, and output layers. It also sets the model optimizer with the learning rate and decay, and then compiles the model with the custom loss functions and metrics. It then checks for (and loads, if available and specified) weights from previous trainings. It also defines an optional leraning rate scheduler modeled with a hyperbolic tangent function. It then trains and tests the RNN, predicting on the test data and loading the predictions into `NumPy` arrays. It proceeds to plot the network history and a variety of results, including histograms, 2D histograms, binned resolutions, and pull plots. It will also plot comparisons to the LLH reconstruction (if avaiable). It then runs some diagnostics on its final performance, and includes similar diagnostics for the LLH performance if available.
