# Upgrade_RNN

## Repository for IceCube-Upgrade RNN

### Background

This repository is for a Recurrent Neural Network (RNN) applied to simulated `i3` data from the IceCube detector at the South Pole. The detector currently consists of 5,160 Digital Optical Modules (DOMs), each with one photomultiplier tube (PMT), situated on 86 strings between 1.5-2.5 km below the surface. The DOMs collect light emitted when a neutrino interacts with a nucleus in the ice, called an event. Depending on when, where, and how much light is detected, we can reconstruct properties of the neutrino like energy, zenith (vertical angle), and azimuth (horizontal angle), among other things. Each of these detections is called a hit.

### Problem

When neutrinos have a very high energy (usually at least 1 TeV), they emit a lot of light in the detector, making it much easier to reconstruct its energy, direction, event type, etc. However, this becomes much more difficult at lower energies because those events produce less light, so we have less information to work with. The central area of IceCube, known as DeepCore, has a higher instrument density than the rest of the detector. DeepCore is used to probe the low-energy regime of neutrino physics, typically on the scale of 10 GeV, but even that is not enough to consistently create accurate reconstructions of the lowest-energy events. This is where the IceCube-Upgrade comes in. The Upgrade is deploying 7 new stings, each with approximately 100 DOMs, into the DeepCore area. The Upgrade will also make use of two new DOM designs: the D-Egg with two PMTs (one on top, one on bottom), and the mDOM with 24 PMTs scattered over its surface. In total, Gen-1 and the Upgrade will have 15,700 PMTs. With these new DOMs, we hope to improve energy reconstruction of low-energy events to the scale of 1 GeV, as well as improve directional reconstruction.

### i3 Data

IceCube uses a data type referred to as `i3`. This data type is specific to IceCube and can be accessed using a module called `I3Tray`/`IceTray`. This module stores information about the event, as well as detector status and detector geometry.

### RNN

The RNN is neural network designed to handle data with a sequential (e.g. temporal) relationship, which is great for IceCube. The RNN takes in three input variables per event: a list of times when light was detected, a list of charges (proportional to how much light was detected), and a list of generated IDs that describe which PMTs were triggered. Each of the corresponding entries in these lists (e.g. `t_1`, `q_1`, `p_1`) would comprise one hit, and all three lists comprise one event. The RNN outputs energy, dx/dy/dz direction, and error estimates for all four. There is also the capability to perform classification by evet type (track/cascade, CC/NC), but this is not currently used.

### Files

A breakdown of all files currently included in the project (excluding this README).

`Attention.py`: Defines attention layer for RNN. No parsed inputs.

`combine_hdf5.py`: Takes data in `.hdf5` form and combines multiple files into a single file. Parsed inputs:
* input_files: path/name of the input file(s) to be combined (can use `?` and `*` as wildcard fillers)
* output_files: path/name of the output file after combination

`export_geometry.py`: Initializes and outputs Gen-1 detector geometry. No parsed inputs.

`export_geometry_V5.py`: Initialzes and outputs Upgrade detector geometry. No parsed inputs.

`filter_hdf5.py`: Applies cuts to `.hdf5` data files and shuffles entries. Parsed inputs:
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

`i3_to_hdf5.py`: Processes information from Gen-1 `.i3` files (or variants) into features/labels for the RNN, stored in `.hdf5` format. Parsed inputs:
* file: path/name of the input file(s) to be processed (can use `?` and `*` as wildcard fillers)
* overwrite: whether or not to overwrite previous output files
* pulse_type: type of pulseseries to use (cleaned or uncleaned)

`i3_to_hdf5_V5.py`: Processes information from Upgrade `.i3` files (or variants) into features/labels for the RNN, stored in `.hdf5` format. Parsed inputs:
* file: path/name of the input file(s) to be processed (can use `?` and `*` as wildcard fillers)
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
It first prints out the selected cuts as a double-check that the right ones are being applied. It loads the information from the input file as `NumPy` arrays into separate dictionaries for easier manipulation. It then automatically names the output file based on the selected cuts, checking that certain cuts make sense, and shuffles all the data before beginning cuts. It applies cuts in the following order:
* PMT, vertex, energy range, track/cascade, CC/NC, reco, energy flattening, track/cascade flattening, max events
After cuts, the data is shuffled again and it prints out the number of events before/after cuts. Lastly, it loads the data back into an `.hdf5` file and saves it.

### Plotting Functions

A breakdown of all the functions currently included in `Plots.py`. In all of the following:

1. "var"/"var1"/"var2"/"var3" is used to represent a variable (energy, zenith, azimuth, etc.)
2. The y-axis is listed first in cases of "var1 vs. var2"
3. "args" is used to represent additional information in filenames that may depend on plotting specifications
4. "True prediction error" refers to the quantity `predicted_var - true_var`
5. "Predicted uncertainty" refers to the RNN estimate of the true prediction error

`plot_uncertainty`: 1D histograms using the true prediction error and predicted uncertainty. Produces:
* var_pull.png (pull plot of `var` uncertainties)
* var_uncertainty.png (histogram of `var` predicted uncertainties)
* var_error.png (histogram of `var` true prediction error)
* var_devetrue.png (histogram of `var` fractional true prediction error)

`plot_uncertainty_2d`: 2D histograms using the true prediction error and predicted uncertainty. Produces:
* var_true_unc_2D.png (2D histogram of `var` true prediction error vs. true `var`)
* predvar_true_unc_2D.png (2D histogram of `var` true prediction error vs. predicted `var`)
* var_pred_unc_2D.png (2D histogram of `var` predicted uncertainty vs. true `var`)
* predvar_pred_unc_2D.png (2D histogram of `var` predicted uncertainty vs. predicted `var`)
* varunc_2D.png (2D histogram of `var` predicted uncertainty vs. `var` true prediction error)

`plot_loss`: Loss during RNN training. Produces:
* loss.png (summarized loss vs. epochs curve across all variables)
* var_loss.png (loss vs. epochs curve specific to `var`)

`plot_2dhist_contours`: 2D histogram of RNN results with contours. Produces:
* var_contours_2D.png (2D histogram of predicted `var` vs. true `var` with median and 1-sigma contours)

`plot_2dhist`: 2D histogram of RNN results without contours. Produces:
* var_2D.png (2D histogram of predicted `var` vs. true `var`)

`plot_1dhist`: 1D histogram of RNN results. Produces:
* var_1D.png (histogram of true and predicted `var`)

`plot_inputs`: Distributions of input variables. Produces:
* time_firstpulse_dist_args.png (histogram of the times of first pulses)
* time_lastpulse_dist_args.png (histogram of the times of last pulses)
* sumcharge_dist_args.png (histogram of the sum of event charge)

`plot_outputs`: Distributions of ouput regression variables. Produces:
* true_var_args.png (histogram of true `var`)

`plot_outputs_classify`: Distributions of output classification variables. Produces:
* true_var1var2_var3_args.png (histogram of `case1` and `case2`, plotted with respect to true `var3`)

`plot_hit_info`: Histograms using the hit and PMT information for each event. Produces:
* dist_hits_energybins_args.png (histogram of average number of hits per event vs. energy)
* dist_hitsperenergy_energybins_args.png (histogram of average number of hits per event per energy vs. energy)
* dist_hits_args.png (histogram of hits per event)
* dist2D_hits_energy_args.png (2D histogram of hits per event vs. energy)
* dist_PMTs_args.png (histogram of unique PMTs triggered per event)
* hits_PMTs_fractions.txt (text file containing statistics on what fractions of events have more than `X` hits and trigger more than `Y` unique PMTs)

`plot_vertex`: Plots event vertices relative to IceCube String 36. Produces:
* vertex_positions_args.png (vertical position and radius from String 36, with `DeepCore` and `IC7` areas outlined)

`plot_error`: Errorbar plot of true prediction error. Produces:
* var1_var2_err.png (1-sigma error bars on `var1` true prediction error in bins of `var2`)

`plot_error_contours`: 2D histogram of true prediction error with contours. Produces:
* var1_var2_err_contours.png (2D histogram of `var1` true prediction error vs. `var2` with median and 1-sigma contours)

`plot_error_vs_reco`: Errorbar plot of true prediction error with LLH reconstruction for comparison. Produces:
* var1_var2_err_comp.png (1-sigma error bars on `var1` and `reco` true prediction error in bins of `var2`)

The above plotting functions make use of the following utility functions:

* `strip_units`: Removes the unit information from a variable like `Energy [GeV]`
* `get_units`: Gets the unit information from a variable like `Energy [GeV]`
* `file_abbrev`: Creates a file abbreviation for a variable like `Energy [GeV]` with special cases
* `bound_uncertainties`: Masks out events with RNN predicted uncertainties that are too large
* `find_contours_2D`: Calculates and returns the contours used in plots with 1-sigma contours

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

### State of the Project

Trained/tested on Upgrade NuMu CC data, the RNN currently performs comparably to RetroReco trained/tested on higher-quality Gen-1 data (see [Spring 2021 IceCube Collaboration Meeting presentation](https://drive.google.com/file/d/1a7s-12JyQ8WBkNC3UerCPal7R_5ioeie/view?usp=sharing) and [Spring 2021 MSU UURAF poster](https://drive.google.com/file/d/1bp8xCFlxWifmlOKAFH88pJ0fEj_qUN_Y/view?usp=sharing)). The primary concern we see now is that a number of events are consistently reconstructed around 5-10 GeV regardless of true energy. We think that some of these events are noise-only/noise-dominated, leading them to "non-reconstructable". In this case, a useful next step would be to develop a cut to mask out these kinds of events. One possible approach is to determine if there is a correlation between the predicted energy and preducted energy uncertainty. If some events at low predicted energies have a high predicted uncertainty, these are likely the events causing this issue. The plotting code to determine this would be `plot_uncertainty_2d` used in `RNN_V5.py`.
