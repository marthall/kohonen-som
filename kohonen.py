
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial.distance import euclidean

DEBUG = True

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

        self.log_frec = log_frec
        self.image_frec = image_frec
        
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
            "distances": [],
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
        self.tmp_sigma = self.sigma
        self.tmp_eta = self.eta

        dy, dx = self.data.shape

        #set the random order in which the datapoints should be presented
        i_random = np.arange(self.iterations) % dy
        np.random.shuffle(i_random)

        for t, i in enumerate(i_random):

            self.som_step(self.data[i,:])
           
            if t % 500 == 0:
                print "Iteration %i" % t
                self.has_converged()

            if (self.log_frec and t % self.log_frec == 0):

                self.quantization_error(self.sample_size)
                self.classify()
                self.stats["delta_weights_average"].append(sum(self.stats["delta_weights"][-100:])/min(len(self.stats["delta_weights"]), 100))


                if (DEBUG):
                    self.plot()

            if (self.image_frec and t % self.image_frec == 0):
                self.plot_image(t)

            if self.variable_sigma:
                self.tmp_sigma = (self.sigma+1)**(1 - float(t)/self.iterations) - 1  # The ones make the range (sigma, 0)
                # self.tmp_eta -= self.delta_eta

        self.quantization_error(sample_size=None)

    def som_step(self, point):
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

        self.stats["delta_weights"].append(np.sum(dweights))

    def plot(self):
        if not self.log_frec:
            return
        for key, value in self.stats.iteritems():
            self.plot_graph(value, key)

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

        if name == "delta_weights":
            x = range(len(data))
        else:
            x = [self.log_frec * i for i, _ in enumerate(data)]

        plt.figure(name)
        plt.plot(x, data)
        savepath = self.get_image_savepath(name, "png")
        plt.savefig(savepath)
        plt.close()

    def quantization_error(self, sample_size):
        point_distances = []

        if sample_size:
            points = np.random.choice(self.data[0].flatten(), sample_size)
        else:
            points = self.data[0]

        for point in points:
            #find the best matching unit via the minimal distance to the datapoint
            b = np.argmin(np.sum((self.centers - np.resize(point, (self.size_xy, point.size)))**2, 1))
            point_distances.append(euclidean(self.centers[b], point))

        self.stats['distances'].append(np.average(point_distances))

    def classify(self):

        labeled_centers = np.zeros(self.size_xy*10).reshape((self.size_xy, 10))


        errors = np.array(self.size_xy)

        for i, point in enumerate(self.data):
            b = np.argmin(np.sum((self.centers - np.resize(point, (self.size_xy, point.size)))**2, 1))
            labeled_centers[b][self.labels[i]] += 1


        error_matrix = np.sum(labeled_centers, axis=1) - np.max(labeled_centers, axis=1)
        self.stats['error'].append(np.sum(error_matrix))
        
        labeled = np.argmax(labeled_centers, axis=1)
        rows = np.nonzero(((labeled_centers == 0).sum(1) == 10))
        for r in rows:
            labeled[r] = -1
        print labeled.reshape(self.size['x'], self.size["y"])

    def has_converged(self):

        size = len(self.stats['delta_weights'])

        window_size = 1000

        gliding_delta_average = []
        for i in np.arange(size-window_size, size, 1):
            gliding_delta_average.append(np.sum(self.stats['delta_weights'][i-window_size:i]))

        print "Variance: %i" % np.std(gliding_delta_average)


def get_data(name):
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    targetdigits = name2digits(name)

    filtered_data = data[np.logical_or.reduce([labels==x for x in targetdigits]),:]
    
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

    iterations = 100000
    size = 6
    sigma = 1.5
    eta = 0.01
    image_frec = 1000

    kohonen = Kohonen(data, labels, iterations, {"x": size, "y": size}, sigma, eta, variable_sigma=False, image_folder="test", sample_size=100, image_frec=image_frec)

    kohonen.run()
    kohonen.classify()
    kohonen.plot()
    kohonen.plot_image()
