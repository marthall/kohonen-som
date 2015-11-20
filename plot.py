import csv
import matplotlib.pyplot as plt
import numpy as np
import os

iterations = 10000
filter_data = False

f = open("data%i.csv" % iterations, "rt")
reader = csv.reader(f)

data = np.array(list(reader))
data = data.transpose().astype(np.float)

if filter_data:
	flt = data[1,:] >= 0.005
	data = data[:,flt]
# print data
sigma, eta, size, error = data

def get_image_savepath(name, ext):
    directory = os.path.split(name)[0]
    filename = "%s.%s" % (os.path.split(name)[1], ext)

    if not os.path.exists(directory):
        os.makedirs(directory)

    return os.path.join(directory, filename)

def plot(data, name):

	x, y, area, color = data
	
	area -= np.min(area)
	area /= np.max(area)
	area *= 10


	plt.figure()
	plt.scatter(x, y, s=(np.pi * (area+4)**2), alpha=0.5, c=color, vmin=np.min(color), vmax=np.max(color), label=name[2], linewidth=0.5)
	plt.xlabel(name[0])
	plt.ylabel(name[1])
	# plt.title(" ".join(name))
	plt.legend()
	clb = plt.colorbar()
	clb.set_label(name[3])
	plt.savefig(get_image_savepath("scatter%i/" % iterations + "-".join(x.lower() for x in name), "png"))
	plt.close()

ds = {
	"Sigma": sigma,
	"Error": error,
	"Size": size,
	"Eta": eta,	
}

import itertools

comb = itertools.permutations(ds.keys())	

for key in list(comb):
	# if key[1] != "Error": continue
	data = [ds[k].copy() for k in key]
	plot(data, key)

