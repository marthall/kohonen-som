
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import euclidean
import csv
import random


class Kohonen:
    def __init__(self, data, labels, iterations, size, sigma, eta, variable_sigma=True, log_frec=None, image_frec=None, image_folder="images", sample_size=None):
        plt.close('all')

        self.data = data
        self.labels = labels
        self.iterations = iterations
        self.size = size
        self.sigma = sigma
        self.eta = eta
        self.variable_sigma = variable_sigma
        self.image_folder = image_folder
        self.sample_size = sample_size if sample_size else None

        self.log_frec = self.iterations / log_frec if log_frec else None
        self.image_frec = self.iterations / image_frec if image_frec else None
        
        self.delta_sigma = float(self.sigma - 1) / self.iterations
        self.delta_eta = float(self.eta - 0.01) / self.iterations
    
        dim = 28*28
        data_range = 255.0
        
        #initialise the centers randomly
        self.centers = np.random.rand(self.size_xy, dim) * data_range

        #build a neighborhood matrix
        self.neighbor = np.arange(self.size_xy).reshape((self.size["x"], self.size["y"]))

        self.stats = {
            "delta_weights": [],
            "delta_weights_average": [],
            "error": [],
            "distances": []
        }

    @property
    def size_xy(self):
        return self.size["x"] * self.size["y"]

    @property
    def image_path(self):
        folder = self.image_folder
        if not self.variable_sigma:
            folder += "_const_sigma"


        folder += "/tmax%i-size%i:%i-sigma%.2f-eta%.3f/" % (self.iterations, self.size["x"], self.size["y"], self.sigma, self.eta)
        return folder
    
    def get_image_savepath(self, name, ext, number=None):
        
        path = self.image_path + name
        if number != None:
            path += "-%05d" % number

        directory = os.path.split(path)[0]
        filename = "%s.%s" % (os.path.split(path)[1], ext)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return os.path.join(directory, filename)

    def run(self):
        # print ""
        # print self.image_path
        
        self.tmp_sigma = self.sigma
        self.tmp_eta = self.eta

        dy, dx = self.data.shape

        #set the random order in which the datapoints should be presented
        i_random = np.arange(self.iterations) % dy
        np.random.shuffle(i_random)

        for t, i in enumerate(i_random):
            if (t % (self.iterations / 10) == 0):
                # print t
                pass

            self.som_step(self.data[i,:])
           
            if (self.log_frec and t % self.log_frec == 0):
                self.quantization_error(self.sample_size)
                self.classify()
                self.stats["delta_weights_average"].append(sum(self.stats["delta_weights"][-100:])/min(len(self.stats["delta_weights"]), 100))

                if  (t % self.log_frec * 10) == 0:
                    self.plot_graph(self.stats['distances'], "distances")
                    self.plot_graph(self.stats['delta_weights'], "delta_weights")
                    self.plot_graph(self.stats['delta_weights_average'], "delta_weights_average")
                    self.plot_graph(self.stats['error'], "error")

            if (self.image_frec and t % self.image_frec == 0):
                self.plot_image(t)

            if self.variable_sigma:
                self.tmp_sigma = (self.sigma+1)**(1 - float(t)/self.iterations) - 1  # The ones make the range (sigma, 0)
                # self.tmp_eta -= self.delta_eta

        self.quantization_error(sample_size=None)

    def som_step(self, point):
        """Performs one step of the sequential learning for a 
        self-organized map (SOM).
        
          centers = som_step(centers,data,neighbor,eta,sigma)
        
          Input and output arguments: 
           centers  (matrix) cluster centres. Have to be in format:
                             center X dimension
           data     (vector) the actually presented datapoint to be presented in
                             this timestep
           neighbor (matrix) the coordinates of the centers in the desired
                             neighborhood.
           eta      (scalar) a learning rate
           sigma    (scalar) the width of the gaussian neighborhood function.
                             Effectively describing the width of the neighborhood
        """
        
        #find the best matching unit via the minimal distance to the datapoint
        b = np.argmin(np.sum((self.centers - np.resize(point, (self.size_xy, point.size)))**2, 1))

        # find coordinates of the winner
        a,b = np.nonzero(self.neighbor == b)

        dweights = []
        # update all units
        for j in range(self.size_xy):
            # find coordinates of this unit
            a1,b1 = np.nonzero(self.neighbor==j)
            
            # calculate the distance and discounting factor
            disc = gauss(np.sqrt((a-a1)**2 + (b-b1)**2), [0, self.tmp_sigma])
            
            # update weights
            dw = disc * self.tmp_eta * (point - self.centers[j,:])

            dweights.append(np.sum(np.abs(dw)))

            self.centers[j,:] += dw

        # print dweights
        # print np.sum(dweights)
        self.stats["delta_weights"].append(np.sum(dweights))

    def plot(self):
        self.plot_graph(self.stats['distances'], "distances")
        self.plot_graph(self.stats['delta_weights'], "delta_weights")
        self.plot_graph(self.stats['delta_weights_average'], "delta_weights_average")
        self.plot_graph(self.stats['error'], "error")
        self.plot_image(self.iterations)
        pass

    def plot_image(self, number=None):
        fig = plt.figure(1, (self.size["x"], self.size["y"]))

        grid = ImageGrid(fig, 111, nrows_ncols=(self.size["y"], self.size["x"]), axes_pad=0.1)

        for i in range(self.size_xy):
            ax = grid[i]
            ax.imshow(np.reshape(self.centers[i,:], [28, 28]))
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        savepath = self.get_image_savepath("image", "png", number)

        plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.close()

    def plot_graph(self, data, name):
        plt.figure(name)
        plt.plot(data)
        savepath = self.get_image_savepath(name, "png")
        plt.savefig(savepath)
        plt.close()

    # def plot_weights(self):

    def quantization_error(self, sample_size):

        point_distances = []
        if sample_size:
            points = np.random.choice(self.data[0].flatten(), sample_size)
        else:
            points = self.data[0]


        for point in points:
            #find the best matching unit via the minimal distance to the datapoint
            b = np.argmin(np.sum((self.centers - np.resize(point, (self.size_xy, point.size)))**2, 1))

            # print np.average(point_distances)
            # find coordinates of the winner
            # a,b = np.nonzero(self.neighbor == b)

            point_distances.append(euclidean(self.centers[b], point))

        self.stats['distances'].append(np.average(point_distances))

        # dist = np.sqrt((a-a1)**2 + (b-b1)**2)

    def classify(self):

        labeled_centers = np.zeros(self.size_xy*10).reshape((self.size_xy, 10))


        errors = np.array(self.size_xy)

        for i, point in enumerate(self.data):
            b = np.argmin(np.sum((self.centers - np.resize(point, (self.size_xy, point.size)))**2, 1))
            labeled_centers[b][self.labels[i]] += 1


        error_matrix = np.sum(labeled_centers, axis=1) - np.max(labeled_centers, axis=1)
        self.stats['error'].append(np.sum(error_matrix))
        print self.stats['error'][-1]

        
        labeled = np.argmax(labeled_centers, axis=1)
        rows = np.nonzero(((labeled_centers == 0).sum(1) == 10))
        for r in rows:
            labeled[r] = -1
        print labeled.reshape(self.size['x'], self.size["y"])

def get_data(name):
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    targetdigits = name2digits(name)

    filtered_data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    
    # print reduce(lambda x: x if x in targetdigits else None, list(targetdigits))
    filtered_labels = np.array(filter(lambda x: x in targetdigits, labels))

    return filtered_data, filtered_labels


def gauss(x,p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0])**2) / (2 * p[1]**2))

def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """
    
    name = name.lower()
    
    if len(name)>25:
        name = name[0:25]
        
    primenumbers = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    
    n = len(name)
    
    s = 0.0
    
    for i in range(n):
        s += primenumbers[i]*ord(name[i])*2.0**(i+1)

    import scipy.io.matlab
    Data = scipy.io.matlab.loadmat('hash.mat',struct_as_record=True)
    x = Data['x']
    t = np.mod(s,x.shape[0])

    return np.sort(x[t,:])

if __name__ == "__main__":
    data, labels = get_data("AVALOSdiana_HALLENmartin")

    # sigma = 4
    # size = 8
    # eta = 0.02
    # iterations = 5000

    # kohonen = Kohonen(data, labels, iterations, {"x": size, "y": size}, sigma, eta, variable_sigma=False, image_frec=100, image_folder="fixed-sigma", log_frec=100)

    # writer.writerow(("sigma", "eta", "size", "result"))

    count = 1
    iterations = 30000

    for i in range(count):
        
        # print "%i/%i " % (i, count)
        # eta = random.random() / 10
        # sigma = random.randint(2, 14)
        # size = random.randint(14, 16)
        sigma = 1
        size = 8
        eta = 0.01
        kohonen = Kohonen(data, labels, iterations, {"x": size, "y": size}, sigma, eta, image_folder="classify", log_frec=10, variable_sigma=False)
        kohonen.run()
        kohonen.classify()
        kohonen.plot()
        result = kohonen.stats['distances'][-1]

        # print (sigma, eta, size, result)

        # f = open("data-fixed-sigma.csv" % iterations, "a")
        # writer = csv.writer(f)
        # writer.writerow((sigma, eta, size, result))
        # f.close()


    # sigmas = np.linspace(1, 10, 10)
    # etas = np.linspace(0.01, 0.1, 10)
    # # sizes = np.linspace(2, 10, 4)

    # print len(sigmas) * len(etas)

    # x = []
    # y = []
    # area = []

    # for sigma in sigmas:
    #     for eta in etas:
    #         # for size in sizes:
    #         kohonen = Kohonen(data, 500, {"x": 8, "y": 8}, sigma, eta)
    #         kohonen.run()
    #         result = kohonen.stats['distances'][-1]
    #         x.append(sigma)
    #         y.append(result)
    #         area.append(np.pi * (100*eta)**2)
    #     # kohonen.plot()

    # plt.figure("newfig")
    # plt.scatter(x, y, s=area, alpha=0.5)

    # plt.savefig("scatter.png")
    # plt.close()