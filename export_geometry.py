import h5py
import numpy
from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units
import sys
import h5py
numpy.set_printoptions(threshold=sys.maxsize)

infilename = "GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
outfilename = "geometry.hdf5"
geo_file = dataio.I3File(infilename)
outfile = h5py.File(outfilename, "w")

frame = geo_file.pop_frame()
frame = geo_file.pop_frame()
g = frame["I3Geometry"]    
omgeo = g.omgeo

dom_indices = numpy.zeros((10,10,60))
dom_positions = numpy.zeros((86*60,3))
dom_indices.fill(-1)

x_coord = 2
y_coord = 0
str_no_prev = 0
for omkey, geo in omgeo:
    str_no = omkey[0]-1
    dom_no = omkey[1]-1
    x_pos = geo.position.x
    y_pos = geo.position.y
    z_pos = geo.position.z
    dom_index = str_no*60+dom_no
    dom_positions[dom_index] = numpy.array([x_pos,y_pos,z_pos])
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
            dom_indices[x_coord,y_coord,dom_no] = dom_index
        x_pos_prev = x_pos
        str_no_prev = str_no

dataset = outfile.create_dataset("indices", data=dom_indices)
dataset = outfile.create_dataset("positions", data=dom_positions)
outfile.close()
