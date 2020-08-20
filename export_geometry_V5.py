import h5py
import numpy
from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units
import sys
import h5py
numpy.set_printoptions(threshold=sys.maxsize)

inpath = "/mnt/scratch/priesbr1/Simulation_Files/"
outpath = "/mnt/scratch/priesbr1/Processed_Files/"
infilename = "GeoCalibDetectorStatus_ICUpgrade.v55.mixed.V5.i3.bz2"
outfilename = "geometry_upgrade.hdf5"
full_infile_name = inpath+infilename
full_outfile_name = outpath+outfilename
geo_file = dataio.I3File(full_infile_name)
outfile = h5py.File(full_outfile_name, "w")

frame = geo_file.pop_frame()
while not (frame.Stop == icetray.I3Frame.Geometry):
    frame = geo_file.pop_frame()
    if not geo_file.more():
        print("No Geometry frame in this file!")
        continue
g = frame["I3Geometry"]
omgeo = g.omgeo
omgeomap = frame["I3OMGeoMap"]
list_of_omkeys = sorted(list(omgeomap.keys()))
num_pmt = len(list_of_omkeys)    

pmt_indices = numpy.zeros((10,10,60))
pmt_positions = numpy.zeros((num_pmt,3))
pmt_directions = numpy.zeros((num_pmt,2))
pmt_indices.fill(-1)

x_coord = 2
y_coord = 0
str_no_prev = 0
for omkey, geo in omgeo:
    str_no = omkey[0]-1
    dom_no = omkey[1]-1
    pmt_no = omkey[2]
    x_pos = geo.position.x
    y_pos = geo.position.y
    z_pos = geo.position.z
    azi = geo.direction.azimuth
    zen = geo.direction.zenith
    for i in range(num_pmt):
        if (omkey.string == list_of_omkeys[i].string and omkey.om == list_of_omkeys[i].om and omkey.pmt == list_of_omkeys[i].pmt):
            pmt_index = i
    pmt_positions[pmt_index] = numpy.array([x_pos,y_pos,z_pos])
    pmt_directions[pmt_index] = numpy.array([zen, azi])

    if str_no < 78:
        if str_no != str_no_prev:
            if x_pos < x_pos_prev:
                y_coord += 1
                if y_coord == 8 or y_coord == 9:
                    x_coord = 2
                elif y_coord == 1 or y_coord == 2 or y_coord == 6 or y_coord == 7:
                    x_coord = 1
                else:
                    x_coord = 0
            else:
                x_coord += 1
        if dom_no < 60:
            pmt_indices[x_coord,y_coord,dom_no] = pmt_index
        x_pos_prev = x_pos
        str_no_prev = str_no

dataset = outfile.create_dataset("indices", data=pmt_indices)
dataset = outfile.create_dataset("positions", data=pmt_positions)
dataset = outfile.create_dataset("directions", data=pmt_directions)
outfile.close()
