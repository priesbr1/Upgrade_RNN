import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import math
import numpy
import numpy as np
import scipy
import itertools

def strip_units(variable):
    # Assumes format like "Base_variable [units]", "Base Variable [units]", or "Base_variable" (unitless)
    if variable.find('[') == -1:
        return variable # already unitless
    else:
        return variable[:variable.find('[')-1]

def get_units(name_with_units):
    # Assumes format like "Base_variable [units]", "Base Variable [units]", or "Base_variable" (unitless)
    if variable.find('[') == -1:
        return "" # already unitless
    else:
        return variable[variable.find('[')+1:variable.find(']')]

def file_abbrev(variable):
    abbrev = ""
    no_units = strip_units(variable)
    if "Cos(Zenith)" in no_units: # special abbrev for cosz
        abbrev += "cosz"
    elif "Uncertainty" in no_units: # uncertainty --> base + unc
        base_var = no_units[:no_units.find("Uncertainty")-1]
        abbrev += str.lower(base_var) + "unc"
    else: # just lowercase
        abbrev += str.lower(no_units)
    abbrev = abbrev.replace(' ','') # remove any spaces ("track length --> tracklength")

    return abbrev

def find_contours_2D(x_values,y_values,xbins,weights=None,c1=16,c2=84):
    """
    --From Jessie Micallef--
    Find upper and lower contours and median
    x_values = array, input for hist2d for x axis (typically truth)
    y_values = array, input for hist2d for y axis (typically reconstruction)
    xbins = values for the starting edge of the x bins (output from hist2d)
    c1 = percentage for lower contour bound (16% - 84% means a 68% band, so c1 = 16)
    c2 = percentage for upper contour bound (16% - 84% means a 68% band, so c2 = 84)
    Returns:
        x = values for xbins, repeated for plotting (i.e. [0,0,1,1,2,2,...]
        y_median = values for y value medians per bin, repeated for plotting (i.e. [40,40,20,20,50,50,...]
        y_lower = values for y value lower limits per bin, repeated for plotting (i.e. [30,30,10,10,20,20,...]
        y_upper = values for y value upper limits per bin, repeated for plotting (i.e. [50,50,40,40,60,60,...]
    """
    if weights is not None:
        import wquantiles as wq
    y_values = numpy.array(y_values)
    indices = numpy.digitize(x_values,xbins)
    r1_save = []
    r2_save = []
    median_save = []
    for i in range(1,len(xbins)):
        mask = indices==i
        if len(y_values[mask])>0:
            if weights is None:
                r1, m, r2 = numpy.percentile(y_values[mask],[c1,50,c2])
            else:
                r1 = wq.quantile(y_values[mask],weights[mask],c1/100.)
                r2 = wq.quantile(y_values[mask],weights[mask],c2/100.)
                m = wq.median(y_values[mask],weights[mask])
        else:
            print(i,"empty bin")
            r1 = 0
            m = 0
            r2 = 0
        median_save.append(m)
        r1_save.append(r1)
        r2_save.append(r2)
    median = numpy.array(median_save)
    lower = numpy.array(r1_save)
    upper = numpy.array(r2_save)

    x = list(itertools.chain(*zip(xbins[:-1],xbins[1:])))
    y_median = list(itertools.chain(*zip(median,median)))
    y_lower = list(itertools.chain(*zip(lower,lower)))
    y_upper = list(itertools.chain(*zip(upper,upper)))

    return x, y_median, y_lower, y_upper

def plot_uncertainty(true, predicted, sigma, quantity, weights, gen_filename="path/save_folder/"):

    errors = predicted-true

    if quantity == "Azimuth [degrees]":
        errors = numpy.array([errors[i] if (errors[i] < 180) else (360-errors[i]) for i in range(len(errors))])
        errors = numpy.array([errors[i] if (errors[i] > -180) else (360+errors[i]) for i in range(len(errors))])
    
    plt.figure()
    plt.title(strip_units(quantity) + " Pull Plot")
    plt.xlabel("Calculated/Predicted Uncertainty " + get_units(quantity))
    plt.ylabel("Normalized Counts")
    pull = numpy.divide(errors,sigma)
    plt.hist(pull, bins=60, range=(-8.0,8.0), histtype="step", density=True, weights=weights)
    x = numpy.linspace(-8.0, 8.0, 100)
    y = scipy.stats.norm.pdf(x,0,1)
    plt.plot(x,y)
    imgname = gen_filename + file_abbrev(quantity) + "_pull.png"
    plt.savefig(imgname)

    # Set uncertainty cutoff to be maximum of range for energy, zenith, azimuth
    if quantity == "Energy [GeV]":
        sigma_cutoff = math.ceil(float(numpy.max(true))/100)*100 # Rounds up to nearest hundred GeV
    elif quantity == "Zenith [deg]":
        sigma_cutoff = 180 # degrees
    elif quantity == "Azimuth [degrees]":
        sigma_cutoff = 360 # degrees
    else:
        sigma_cutoff = numpy.inf

    if np.max(sigma) >= sigma_cutoff:
        non_inf = sigma <= sigma_cutoff
        sigma_bounded = sigma[non_inf]
        sigma_overflow = len(sigma)-len(sigma_bounded)
        if sigma_overflow == 1:
            print(sigma_overflow, " large uncertainty for ", strip_units(quantity))
        else:
            print(sigma_overflow, " large uncertainties for ", strip_units(quantity))
        if len(sigma_bounded) > 0:
            print("New maximum uncertainty:", np.max(sigma_bounded))
            abort_sigma_plot = False
        else:
            print("Aborting sigma histogram -- no small uncertainties for ", strip_units(quantity))
            abort_sigma_plot = True
    else:
        sigma_overflow = None
        abort_sigma_plot = False

    if abort_sigma_plot == False:
        plt.figure()
        plt.title("Predicted " + strip_units(quantity) + " Uncertainty")
        plt.xlabel("Uncertainty " + get_units(quantity))
        plt.ylabel("Normalized Counts")
        if sigma_overflow:
            plt.hist(sigma_bounded, bins=100, histtype="step", density=True, weights=weights[non_inf])
        else:
            plt.hist(sigma, bins=100, histtype="step", density=True, weights=weights)
        imgname = gen_filename + file_abbrev(quantity) + "_uncertainty.png"
        plt.savefig(imgname)

    plt.figure()
    plt.title(strip_units(quantity) + " Prediction Error")
    plt.xlabel(strip_units(quantity) + " Error " + get_units(quantity))
    plt.ylabel("Normalized Counts")
    plt.hist(errors, bins=100, histtype="step", density=True, weights=weights)
    imgname = gen_filename + file_abbrev(quantity) + "_error.png"
    plt.savefig(imgname)

    plt.figure()
    plt.title("Error vs. True " + strip_units(quantity))
    plt.xlabel(strip_units(quantity) + " Error / True " + strip_units(quantity))
    plt.ylabel("Normalized Counts")
    plt.hist(numpy.divide(errors,true), bins=30, range=(-3.0,3.0), histtype="step", density=True, weights=weights)
    imgname = gen_filename + file_abbrev(quantity) + "_devetrue.png"
    plt.savefig(imgname)

    del sigma_overflow

def plot_uncertainty_2d(true, predicted, sigma, quantity, weights, gen_filename="path/save_folder/"):

   errors = predicted-true

    if quantity == "Azimuth [degrees]":
        errors = numpy.array([errors[i] if (errors[i] < 180) else (360-errors[i]) for i in range(len(errors))])
        errors = numpy.array([errors[i] if (errors[i] > -180) else (360+errors[i]) for i in range(len(errors))])

    plt.figure()
    plt.title("True " + strip_units(quantity) + " Uncertainty vs. True " + strip_units(quantity))
    plt.xlabel("True " + quantity)
    plt.ylabel("True " + strip_units(quantity) + " Uncertainty " + get_units(quantity))
    cnts, xbins, ybins, img = plt.hist2d(true, errors, weights=weights, bins=100, range=[[min(true),max(true)],[min(errors),max(errors)]], norm=matplotlib.colors.LogNorm())
    x, y_med, y_lower, y_upper = find_contours_2D(true, errors, xbins, weights=weights)
    plt.plot(x, y_med, color='r', label="Median")
    plt.plot(x, y_lower, color='r', linestyle="dashed", label="68% band")
    plt.plot(x, y_upper, color='r', linestyle="dashed")
    plt.legend(loc="best")
    plt.grid()
    bar = plt.colorbar()
    bar.set_label("Counts")
    plt.plot([min(true),max(true)], [min(errors),max(errors)], color="black", linestyle="dashed")
    imgname = gen_filename + "true_" + file_abbrev(quantity) + "unc_2D.png"
    plt.savefig(imgname)

    # Set uncertainty cutoff to be maximum of range for energy, zenith, azimuth
    if quantity == "Energy [GeV]":
        sigma_cutoff = math.ceil(float(numpy.max(true))/100)*100 # Rounds up to nearest hundred GeV
    elif quantity == "Zenith [deg]":
        sigma_cutoff = 180 # degrees
    elif quantity == "Azimuth [degrees]":
        sigma_cutoff = 360 # degrees
    else:
        sigma_cutoff = numpy.inf

    if np.max(sigma) >= sigma_cutoff:
        non_inf = sigma <= sigma_cutoff
        sigma_bounded = sigma[non_inf]
        sigma_overflow = len(sigma)-len(sigma_bounded)
        if sigma_overflow == 1:
            print(sigma_overflow, " large uncertainty for ", strip_units(quantity))
        else:
            print(sigma_overflow, " large uncertainties for ", strip_units(quantity))
        if len(sigma_bounded) > 0:
            print("New maximum uncertainty:", np.max(sigma_bounded))
            abort_sigma_plot = False
        else:
            print("Aborting sigma histogram -- no small uncertainties for ", strip_units(quantity))
            abort_sigma_plot = True
    else:
        sigma_overflow = None
        abort_sigma_plot = False

    if abort_sigma_plot == False:
        plt.figure()
        plt.title("Predicted " + strip_units(quantity) + " Uncertainty vs. True " + strip_units(quantity))
        plt.xlabel("True " + quantity)
        plt.ylabel("Predicted " + strip_units(quantity) + " Uncertainty " + get_units(quantity))
        cnts, xbins, ybins, img = plt.hist2d(true, errors, weights=weights, bins=100, range=[[min(true),max(true)],[min(sigma_bounded),max(sigma_bounded)]], norm=matplotlib.colors.LogNorm())
        x, y_med, y_lower, y_upper = find_contours_2D(true, sigma_bounded, xbins, weights=weights)
        plt.plot(x, y_med, color='r', label="Median")
        plt.plot(x, y_lower, color='r', linestyle="dashed", label="68% band")
        plt.plot(x, y_upper, color='r', linestyle="dashed")
        plt.legend(loc="best")
        plt.grid()
        bar = plt.colorbar()
        bar.set_label("Counts")
        plt.plot([min(true),max(true)], [min(sigma_bounded),max(sigma_bounded)], color="black", linestyle="dashed")
        imgname = gen_filename + "pred_" + file_abbrev(quantity) + "unc_2D.png"
        plt.savefig(imgname)

        plt.figure()
        plt.title("Predicted " + strip_units(quantity) + " Uncertainty vs. True " + strip_units(quantity) + "Uncertainty")
        plt.xlabel("True " + strip_units(quantity) + " Uncertainty " + get_units(quantity))
        plt.ylabel("Predicted " + strip_units(quantity) + " Uncertainty " + get_units(quantity))
        cnts, xbins, ybins, img = plt.hist2d(errors, sigma_bounded, weights=weights, bins=100, range=[[min(errors),max(errors)],[min(sigma_bounded),max(sigma_bounded)]], norm=matplotlib.colors.LogNorm())
        x, y_med, y_lower, y_upper = find_contours_2D(errors, sigma_bounded, xbins, weights=weights)
        plt.plot(x, y_med, color='r', label="Median")
        plt.plot(x, y_lower, color='r', linestyle="dashed", label="68% band")
        plt.plot(x, y_upper, color='r', linestyle="dashed")
        plt.legend(loc="best")
        plt.grid()
        bar = plt.colorbar()
        bar.set_label("Counts")
        plt.plot([min(errors),max(errors)], [min(sigma_bounded),max(sigma_bounded)], color="black", linestyle="dashed")
        imgname = gen_filename +  file_abbrev(quantity) + "_unc_2D.png"
        plt.savefig(imgname)

def plot_loss(history, test, metric, variable, no_epochs, gen_filename="path/save_folder/", unc=False):
    plt.figure()
    if unc == False:
        plt.title("Loss over %i Epochs"%no_epochs)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(history[metric], label="Training")
        plt.plot(history["val_"+metric], label="Validation")
        plt.plot(no_epochs-1, test, "ro", label="Test")
    else:
        plt.title("%s Loss over %i Epochs"%(variable, no_epochs))
        plt.xlabel("Epochs")
        plt.ylabel("%s Loss"%variable)
        plt.plot(history[metric[0]], label=variable+" Training")
        plt.plot(history["val_"+metric[0]], label=variable+" Validation")
        plt.plot(no_epochs-1, test[0], "ro", label=variable+" Test")
        plt.plot(history[metric[1]], label=variable+" Uncertainty Training")
        plt.plot(history["val_"+metric[1]], label=variable+" Uncertainty Validation")
        plt.plot(no_epochs-1, test[1], "go", label=variable+" Uncertainty Test")
    plt.legend(loc="best")
    if variable == "Loss":
        imgname = gen_filename + file_abbrev(variable) + ".png"
    else:
        imgname = gen_filename + file_abbrev(variable) + "_loss.png"
    plt.savefig(imgname)

def plot_2dhist_contours(true, predicted, xymin, xymax, quantity, weights, gen_filename="path/save_folder/"):
    plt.figure()
    plt.title("Predicted vs. True " + strip_units(quantity))
    plt.xlabel("True " + quantity)
    plt.ylabel("Predicted " + quantity)
    cnts, xbins, ybins, img = plt.hist2d(true, predicted, weights=weights, bins=100, range=[[xymin,xymax],[xymin,xymax]], norm=matplotlib.colors.LogNorm())
    x, y_med, y_lower, y_upper = find_contours_2D(true, predicted, xbins, weights=weights)
    plt.plot(x, y_med, color='r', label="Median")
    plt.plot(x, y_lower, color='r', linestyle="dashed", label="68% band")
    plt.plot(x, y_upper, color='r', linestyle="dashed")
    plt.legend(loc="best")
    plt.grid()
    bar = plt.colorbar()
    bar.set_label("Counts")
    plt.plot([xymin,xymax], [xymin,xymax], color="black", linestyle="dashed")
    imgname = gen_filename + file_abbrev(quantity) + "_contours_2D.png"
    plt.savefig(imgname)

def plot_2dhist(true, predicted, xymin, xymax, quantity, weights, gen_filename="path/save_folder/"):
    plt.figure()
    plt.title("Predicted vs. True " + strip_units(quantity))
    plt.xlabel("True " + quantity)
    plt.ylabel("Predicted " + quantity)
    plt.hist2d(true, predicted, weights=weights, bins=100, range=[[xymin,xymax],[xymin,xymax]], norm=matplotlib.colors.LogNorm())
    bar = plt.colorbar()
    bar.set_label("Counts")
    plt.plot([xymin,xymax], [xymin,xymax], color='r')
    imgname = gen_filename + file_abbrev(quantity) + "_2D.png"
    plt.savefig(imgname)

def plot_1dhist(true, predicted, minimum, maximum, quantity, weights, gen_filename="path/save_folder/"):
    plt.figure()
    plt.title("Predicted vs. True " + strip_units(quantity))
    plt.xlabel(quantity)
    plt.hist(true, bins=100, range=[minimum,maximum], histtype="step", weights=weights, label="True")
    plt.hist(predicted, bins=100, range=[minimum,maximum], histtype="step", weights=weights, label="Predicted")
    plt.legend(loc="best")
    imgname = gen_filename + file_abbrev(quantity) + "_1D.png"
    plt.savefig(imgname)

def plot_inputs(pulse_time_data, pulse_charge_data, num_use, log_charge=False, gen_filename="/path/save_folder/"):
    first_pulses = []
    last_pulses = []
    event_charge = []
    if num_use and num_use < len(pulse_time_data):
        for i in range(num_use):
            first_pulses.append(pulse_time_data[i][0])
            last_pulses.append(pulse_time_data[i][-1])
            event_charge.append(np.sum(pulse_charge_data[i]))
    else:
        for i in range(len(pulse_time_data)):
            first_pulses.append(pulse_time_data[i][0])
            last_pulses.append(pulse_time_data[i][-1])
        for i in range(len(pulse_charge_data)):
            event_charge.append(np.sum(pulse_charge_data[i]))

    plt.figure()
    plt.hist(first_pulses, bins=100, histtype="stepfilled")
    plt.xlabel("Time [ns]")
    if not num_use or num_use >= len(first_pulses):
        plt.title("Time of First Pulse Distribution, n=%i"%len(first_pulses))
    else:
        plt.title("Time of First Pulse Distribution, n=%i"%num_use)
    if not num_use or num_use >= len(first_pulses):
        imgname = gen_filename + "time_firstpulse_dist_all.png"
    else:
        imgname = gen_filename + "time_firstpulse_dist_" + str(num_use) + ".png"
    plt.savefig(imgname)

    plt.figure()
    plt.hist(last_pulses, bins=100, histtype="stepfilled")
    plt.xlabel("Time [ns]")
    if not num_use or num_use >= len(last_pulses):
        plt.title("Time of Last Pulse Distribution, n=%i"%len(last_pulses))
    else:
        plt.title("Time of Last Pulse Distribution, n=%i"%num_use)
    if not num_use or num_use >= len(last_pulses):
        imgname = gen_filename + "time_lastpulse_dist_all.png"
    else:
        imgname = gen_filename + "time_lastpulse_dist_" + str(num_use) + ".png"
    plt.savefig(imgname)

    plt.figure()
    plt.hist(event_charge, bins=100, histtype="stepfilled")#, range=(0,500))
    plt.xlabel("Charge")
    if not num_use or num_use >= len(event_charge):
        plt.title("Sum of Event Charge Distribution, n=%i"%len(event_charge))
    else:
        plt.title("Sum of Event Charge Distribution, n=%i"%num_use)
    if log_charge == True:
        plt.yscale("log")
    if (not num_use or num_use >= len(pulse_charge_data)) and log_charge == False:
        imgname = gen_filename + "sumcharge_dist_all.png"
    elif not num_use or num_use >= len(pulse_charge_data):
        imgname = gen_filename + "sumcharge_dist_ylog_all.png"
    elif log_charge == True:
        imgname = gen_filename + "sumcharge_dist_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "sumcharge_dist_" + str(num_use) + ".png"
    plt.savefig(imgname)

def plot_outputs(true, minimum, maximum, quantity, weights, num_use, logscale=False, gen_filename="path/save_folder/"):
    plt.figure()
    if not num_use or num_use >= len(true):
        plt.title("True " + strip_units(quantity) + ", n=%i"%len(true))
    else:
        plt.title("True " + strip_units(quantity) + ", n=%i"%num_use)
    plt.xlabel(quantity)
    plt.ylabel("Counts")
    if logscale == True:
        plt.yscale("log")
    if not num_use or num_use >= len(true):
        plt.hist(true, bins=100, range=[minimum,maximum], histtype="stepfilled", weights=weights)
    else:
        plt.hist(true[:num_use], bins=100, range=[minimum,maximum], histtype="stepfilled", weights=weights[:num_use])
    if (not num_use or num_use >= len(true)) and logscale == False:
        imgname = gen_filename + "true_" + file_abbrev(quantity) + "_all.png"
    elif not num_use or num_use >= len(true):
        imgname = gen_filename + "true_" + file_abbrev(quantity) + "_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "true_" + file_abbrev(quantity) + "_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "true_" + file_abbrev(quantity) + '_' + str(num_use) + ".png"
    plt.savefig(imgname)

def plot_outputs_classify(true1, true2, true3, minimum, maximum, quantity1, quantity2, quantity3, labels, num_use, logscale=False, gen_filename="path/save_folder/"):
    var1 = []
    var2 = []
 
    if not num_use or num_use >= len(true3):
        for i in range(len(true3)):
            if true1[i] == True or true1[i] == 1.0:
                var1.append(true3[i])
            else:
                var2.append(true3[i])
    else:
        for i in range(num_use):
            if true1[i] == True or true1[i] == 1.0:
                var1.append(true3[i])
            else:
                var2.append(true3[i])

    if len(var1) == 0:
        print("Aborting plotting -- no %s entries in dataset"%quantity1)
        return
    if len(var2) == 0:
        print("Aborting plotting -- no %s entries in dataset"%quantity2)
        return

    plt.figure()
    if not num_use or num_use >= len(true3):
        plt.title("True " + labels[0] + '/' + labels[1] + ", n=%i"%len(true3))
    else:
        plt.title("True " + labels[0] + '/' + labels[1] + ", n=%i"%num_use)
    plt.xlabel(quantity3)
    plt.ylabel("Counts")
    if logscale == True:
        plt.yscale("log")
    plt.hist(var1, bins=100, range=[minimum,maximum], histtype="stepfilled", alpha=0.5, label=labels[0])
    plt.hist(var2, bins=100, range=[minimum,maximum], histtype="stepfilled", alpha=0.5, label=labels[1])
    plt.legend(loc="best")
    if (not num_use or num_use >= len(true3)) and logscale == False:
        imgname = gen_filename + "true_" + quantity1 + quantity2 + '_' + file_abbrev(quantity3) + "_all.png"
    elif not num_use or num_use >= len(true3):
        imgname = gen_filename + "true_" + quantity1 + quantity2 + '_' + file_abbrev(quantity3) + "_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "true_" + quantity1 + quantity2 + '_' + file_abbrev(quantity3) + "_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "true_" + quantity1 + quantity2 + '_' + file_abbrev(quantity3) + '_' + str(num_use) + ".png"
    plt.savefig(imgname)

def plot_hit_info(pulse_charge, pmt_index, true_energies, num_use, logscale=False, gen_filename="path/save_folder/"):
    no_hits = numpy.zeros(len(pulse_charge))

    if len(no_hits) != len(true_energies):
        raise RuntimeError("Length of energies (%i) and hits (%i) do not match"%(len(true_energies), len(no_hits)))

    for i in range(len(no_hits)):
        no_hits[i] = len(pulse_charge[i])

    plt.figure()
    if num_use and num_use <= len(no_hits):
        plt.title("Average Number of Hits per Energy Bin, n=%i"%num_use)
    else:
        plt.title("Average Number of Hits per Energy Bin, n=%i"%len(no_hits))
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Average Number of Hits")
    if logscale == True:
        plt.yscale("log")
    hit_dist = numpy.array([])
    energy_dist = numpy.array([])
    for i in range(0,int(max(true_energies))):
        energy_dist = numpy.append(energy_dist,i)
        relevant_energies = numpy.logical_and(true_energies >= i,true_energies < i+1)
        relevant_hits = no_hits[relevant_energies]
        if len(relevant_hits) == 0:
            hit_dist = numpy.append(hit_dist,0)
        else:
            if num_use and len(relevant_hits) > num_use:
                relevant_hits = relevant_hits[:num_use]
                hit_dist = numpy.append(hit_dist,sum(relevant_hits)/len(relevant_hits))
            else:
                hit_dist = numpy.append(hit_dist,sum(relevant_hits)/len(relevant_hits))
    plt.bar(energy_dist,hit_dist,8)
    if (not num_use or num_use >= len(pulse_charge)) and logscale == False:
        imgname = gen_filename + "dist_hits_energybins_all.png"
    elif not num_use or num_use >= len(pulse_charge):
        imgname = gen_filename + "dist_hits_energybins_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "dist_hits_energybins_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "dist_hits_energybins_" + str(num_use) + ".png"
    plt.savefig(imgname)

    plt.figure()
    if num_use and num_use <= len(no_hits):
        plt.title("Average Hits/Energy per Energy Bin, n=%i"%num_use)
    else:
        plt.title("Average Hits/Energy per Energy Bin, n=%i"%len(no_hits))
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Average Hits/Energy")
    if logscale == True:
        plt.yscale("log")
    hit_dist = numpy.array([])
    energy_dist = numpy.array([])
    for i in range(0,int(max(true_energies))):
        energy_dist = numpy.append(energy_dist,i)
        relevant_energies = numpy.logical_and(true_energies >= i,true_energies < i+1)
        relevant_hits = numpy.divide(no_hits[relevant_energies],true_energies[relevant_energies])
        if len(relevant_hits) == 0:
            hit_dist = numpy.append(hit_dist,0)
        else:
            if num_use and len(relevant_hits) > num_use:
                relevant_hits = relevant_hits[:num_use]
                hit_dist = numpy.append(hit_dist, sum(relevant_hits)/len(relevant_hits))
            else:
                hit_dist = numpy.append(hit_dist,sum(relevant_hits)/len(relevant_hits))
    plt.bar(energy_dist,hit_dist,8)
    if (not num_use or num_use >= len(pulse_charge)) and logscale == False:
        imgname = gen_filename + "dist_hitsperenergy_energybins_all.png"
    elif not num_use or num_use >= len(pulse_charge):
        imgname = gen_filename + "dist_hitsperenergy_energybins_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "dist_hitsperenergy_energybins_ylog_" + str(num_use) + ".png")
    else:
        imgname = gen_filename + "dist_hitsperenergy_energybins_" + str(num_use) + ".png"
    plt.savefig(imgname)

    plt.figure()
    if num_use and num_use <= len(no_hits):
        plt.title("Distribution of Hit Numbers, n=%i"%num_use)
    else:
        plt.title("Distribution of Hit Numbers, n=%i"%len(no_hits))
    plt.xlabel("Hits per Event")
    plt.ylabel("Counts")
    if logscale == True:
        plt.yscale("log")
    max_hits = math.ceil(max(no_hits)/100.0)*100
    plt.hist(no_hits, bins=50, range=[0,max_hits], histtype="step")
    if (not num_use or num_use >= len(pulse_charge)) and logscale == False:
        imgname = gen_filename + "dist_hits_all.png"
    elif not num_use or num_use >= len(pulse_charge):
        imgname = gen_filename + "dist_hits_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "dist_hits_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "dist_hits_" + str(num_use) + ".png"
    plt.savefig(imgname)

    plt.figure()
    if num_use and num_use <= len(no_hits):
        plt.title("Distribution of Hit Numbers per Energy Bin, n=%i"%num_use)
    else:
        plt.title("Distribution of Hit Numbers per Energy Bin, n=%i"%len(no_hits))
    plt.xlabel("Energy [GeV]")
    plt.ylabel("Hits per Event")
    if logscale == True:
        plt.yscale("log")
    max_hits = math.ceil(max(no_hits)/100.0)*100
    max_energy = int(math.ceil(max(true_energies)))
    plt.hist2d(true_energies, no_hits, bins=100, range=[[0,max_energy],[0,max_hits]], norm=matplotlib.colors.LogNorm())
    bar = plt.colorbar()
    bar.set_label("Counts")
    if (not num_use or num_use >= len(pulse_charge)) and logscale == False:
        imgname = gen_filename + "dist2D_hits_energy_all.png"
    elif not num_use or num_use >= len(pulse_charge):
        imgname = gen_filename + "dist2D_hits_energy_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "dist2D_hits_energy_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "dist2D_hits_energy_" + str(num_use) + ".png"
    plt.savefig(imgname)

    fractions_file = open(gen_filename + "hits_PMTs_fractions.txt", 'w')
    for val in [10,50,100,150,200,250,300,400,500]:
        over_frac = len(numpy.argwhere(no_hits > val))/len(no_hits)
        fractions_file.write("Fraction of events with more than %i hits: %.5f\n"%(val, over_frac))
    fractions_file.write('-'*50+'\n')

    no_pmts = numpy.zeros(len(pmt_index))
    for i in range(len(no_pmts)):
        no_pmts[i] = len(numpy.unique(pmt_index[i]))

    plt.figure()
    if num_use and num_use <= len(no_pmts):
        plt.title("Distribution of PMTs Triggered, n=%i"%num_use)
    else:
        plt.title("Distribution of PMTs Triggered, n=%i"%len(no_pmts))
    plt.xlabel("PMTs Triggered")
    plt.ylabel("Counts")
    if logscale == True:
        plt.yscale("log")
    plt.hist(no_pmts, bins=100, range=[0,int(math.ceil(max(no_pmts)))], histtype="step")
    if (not num_use or num_use >= len(pmt_index)) and logscale == False:
        imgname = gen_filename + "dist_PMTs_all.png"
    elif not num_use or num_use >= len(pmt_index):
        imgname = gen_filename + "dist_PMTs_ylog_all.png"
    elif logscale == True:
        imgname = gen_filename + "dist_PMTs_ylog_" + str(num_use) + ".png"
    else:
        imgname = gen_filename + "dist_PMTs_" + str(num_use) + ".png"
    plt.savefig(imgname)

    for val in [5,15,30,50,75,100,150,200]:
        over_frac = len(numpy.argwhere(no_pmts > val))/len(no_pmts)
        fractions_file.write("Fraction of events with more that %i PMTs triggered: %.5f\n"%(val, over_frac))
    fractions_file.close()

def plot_vertex(x, y, z, num_use, gen_filename="path/save_folder/"):
    x_origin = 46.290000915527344
    y_origin = -34.880001068115234

    r = numpy.sqrt((x-x_origin)**2 + (y-y_origin)**2)
    r_max = max(r)
    
    plt.figure()
    if num_use and num_use <= len(z):
        plt.title("Distribution of Vertex Locations, n=%i"%num_use)
    else:
        plt.title("Distribution of Vertex Locations, n=%i"%len(z))
    plt.xlabel("Radius from String 36 [m]")
    plt.ylabel("Vertical Position [m]")
    if num_use and num_use <= len(z):
        plt.scatter(r[:num_use],z[:num_use], s=180/r_max)
    else:
        plt.scatter(r,z, s=2)
    # plot DeepCore and IC7 areas
    plt.hlines([-505,-155], 0, 90, color="green")
    plt.vlines(90, -505, -155, color="green", label="DeepCore")
    plt.hlines([-505,-155], 0, 150, color="blue")
    plt.vlines(150, -505, -155, color="blue", label="IC7")
    plt.legend(loc="best")
    if not num_use or num_use >= len(z):
        imgname = gen_filename + "vertex_positions_all.png"
    else:
        imgname = gen_filename + "vertex_positions_" + str(num_use) + ".png"
    plt.savefig(imgname)

def plot_error(true, predicted, minimum, maximum, quantity, quantity2=0, x=0, gen_filename="path/save_folder/"):
    if quantity2 == 0:
        x = true
        quantity2 = quantity

    fractional_errors = predicted-true

    if quantity == "Energy [GeV]":
        fractional_errors = ((predicted-true)/true)*100. # in percent
    elif quantity == "Azimuth [degrees]":
        fractional_errors = numpy.array([(predicted[i]-true[i]) if math.fabs((predicted[i]-true[i])) < 180 else ((predicted[i]-true[i]-360) if predicted[i] > true[i] else (predicted[i]-true[i]+360)) for i in range(len(true))])
    else:
        fractional_errors = predicted-true

    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    ranges  = numpy.linspace(minimum, maximum, num=10)
    centers = (ranges[1:] + ranges[:-1])/2.

    medians  = numpy.zeros(len(centers))
    err_from = numpy.zeros(len(centers))
    err_to   = numpy.zeros(len(centers))

    for i in range(len(ranges)-1):
        val_from = ranges[i]
        val_to   = ranges[i+1]

        cut = (x >= val_from) & (x < val_to)
        lower_lim = numpy.percentile(fractional_errors[cut], left_tail_percentile)
        upper_lim = numpy.percentile(fractional_errors[cut], right_tail_percentile)
        median = numpy.percentile(fractional_errors[cut], 50.)

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim

    plt.figure()
    plt.errorbar(centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], fmt='o')
    plt.plot([minimum,maximum], [0,0], color='k')
    plt.xlim(minimum,maximum)

    plt.title(strip_units(quantity) + " Error vs. " + strip_units(quantity2))
    plt.xlabel(quantity2)

    if quantity == "Energy [GeV]":
        plt.ylabel("Energy Percent Error")
    else :
        plt.ylabel(strip_units(quantity) + " Error " + get_units(quantity))
    imgname = gen_filename + file_abbrev(quantity) + '_' + file_abbrev(quantity2) + "_err.png"
    plt.savefig(imgname)

def plot_error_contours(true, predicted, minimum, maximum, quantity, quantity2=0, x=0, gen_filename="path/save_folder/"):
    if quantity2 == 0:
        x = true
        quantity2 = quantity

    fractional_errors = predicted-true

    if quantity == "Energy [GeV]":
        fractional_errors = ((predicted-true)/true)*100. # in percent
    elif quantity == "Azimuth [degrees]":
        fractional_errors = numpy.array([(predicted[i]-true[i]) if math.fabs((predicted[i]-true[i])) < 180 else ((predicted[i]-true[i]-360) if predicted[i] > true[i] else (predicted[i]-true[i]+360)) for i in range(len(true))])
    else:
        fractional_errors = predicted-true

    plt.figure()
    if quantity == "Energy [GeV]":
        cnts, xbins, ybins, img = plt.hist2d(x, fractional_errors, bins=100, range=[[minimum,maximum],[-100,100]], norm=matplotlib.colors.LogNorm()) # -100 to 100 percent y-axis
    else:
        cnts, xbins, ybins, img = plt.hist2d(x, fractional_errors, bins=100, range=[[minimum,maximum],[-1*max(true),max(true)]], norm=matplotlib.colors.LogNorm())
    x, y_med, y_lower, y_upper = find_contours_2D(x, fractional_errors, xbins)
    plt.hlines(0, minimum, maximum, color="black", linestyle="dashed")
    plt.plot(x, y_med, color='r', label="Median")
    plt.plot(x, y_lower, color='r', linestyle="dashed", label="68% band")
    plt.plot(x, y_upper, color='r', linestyle="dashed")
    plt.legend(loc="best")
    plt.grid()
    bar = plt.colorbar()
    bar.set_label("Counts")

    plt.title(strip_units(quantity) + " Error vs. " + strip_units(quantity2))
    plt.xlabel(quantity2)

    if quantity == "Energy [GeV]":
        plt.ylabel("Energy Percent Error")
    else:
        plt.ylabel(strip_units(quantity) + " Error " + get_units(quantity))

    imgname = gen_filename + file_abbrev(quantity) + '_' + file_abbrev(quantity2) + "_err_contours.png"
    plt.savefig(imgname)

def plot_error_vs_reco(true, predicted, reco, minimum, maximum, quantity, quantity2=0, x=0, gen_filename="path/save_folder/"):
    if quantity2 == 0:
        x = numpy.copy(true)
        quantity2 = quantity

    x = numpy.copy(x[reco > 1e-3])
    true = numpy.copy(true[reco > 1e-3])
    predicted = numpy.copy(predicted[reco > 1e-3])

    x_reco = numpy.copy(x)
    true_reco = numpy.copy(true)
    reco = numpy.copy(reco[reco > 1e-3])

    if quantity == "Energy [GeV]":
        fractional_errors = ((predicted-true)/true)*100. # in percent
        fractional_errors_reco = ((reco-true_reco)/true_reco)*100.
    elif quantity == "Azimuth [degrees]":
        fractional_errors = numpy.array([(predicted[i]-true[i]) if math.fabs((predicted[i]-true[i])) < 180 else ((predicted[i]-true[i]-360) if predicted[i] > true[i] else (predicted[i]-true[i]+360)) for i in range(len(true))])
        fractional_errors_reco = numpy.array([(reco[i]-true_reco[i]) if math.fabs((reco[i]-true_reco[i])) < 180 else ((reco[i]-true_reco[i]-360) if reco[i] > true_reco[i] else (reco[i]-true_reco[i]+360)) for i in range(len(true_reco))])
    else:
        fractional_errors = predicted-true
        fractional_errors_reco = reco-true_reco

    percentile_in_peak = 68.27
    left_tail_percentile  = (100.-percentile_in_peak)/2
    right_tail_percentile = 100.-left_tail_percentile

    ranges  = numpy.linspace(minimum, maximum, num=10)
    centers = (ranges[1:] + ranges[:-1])/2.

    medians  = numpy.zeros(len(centers))
    err_from = numpy.zeros(len(centers))
    err_to   = numpy.zeros(len(centers))
    medians_reco = numpy.zeros(len(centers))
    err_from_reco = numpy.zeros(len(centers))
    err_to_reco = numpy.zeros(len(centers))

    for i in range(len(ranges)-1):
        val_from = ranges[i]
        val_to   = ranges[i+1]

        cut = (x >= val_from) & (x < val_to)
        lower_lim = numpy.percentile(fractional_errors[cut], left_tail_percentile)
        upper_lim = numpy.percentile(fractional_errors[cut], right_tail_percentile)
        median = numpy.percentile(fractional_errors[cut], 50.)
        cut_reco = (x_reco >= val_from) & (x_reco < val_to)
        lower_lim_reco = numpy.percentile(fractional_errors_reco[cut_reco], left_tail_percentile)
        upper_lim_reco = numpy.percentile(fractional_errors_reco[cut_reco], right_tail_percentile)
        median_reco = numpy.percentile(fractional_errors_reco[cut_reco], 50.)

        medians[i] = median
        err_from[i] = lower_lim
        err_to[i] = upper_lim
        medians_reco[i] = median_reco
        err_from_reco[i] = lower_lim_reco
        err_to_reco[i] = upper_lim_reco

    plt.figure()
    plt.errorbar(centers, medians, yerr=[medians-err_from, err_to-medians], xerr=[ centers-ranges[:-1], ranges[1:]-centers ], fmt='o',label="RNN", capsize=5)
    plt.errorbar(centers, medians_reco, yerr=[medians_reco-err_from_reco, err_to_reco-medians_reco], xerr=[centers-ranges[:-1], ranges[1:]-centers], fmt='o', label="PegLeg", capsize=5)
    plt.plot([minimum,maximum], [0,0], color='k')
    plt.xlim(minimum,maximum)

    plt.title(strip_units(quantity) + " Error vs. " + strip_units(quantity2))
    plt.xlabel(quantity2)
    if quantity == "Energy [GeV]":
        plt.ylabel("Energy Percent Error")
    else:
        plt.ylabel(strip_units(quantity) + " Error " + get_units(quantity))
    plt.legend(loc="best")
    imgname = gen_filename + fle_abbrev(quantity) + '_' + file_abbrev(quantity2) + "_err_comp.png"
    plt.savefig(imgname)
