import numpy as np

class KNearestNeighbor(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x, k=1, num_of_loops=0):
        if num_of_loops == 0:
            dists = self.compute_distances_no_loop(x)
        elif num_of_loops == 1:
            dists = self.compute_distances_one_loop(x)
        elif num_of_loops == 2:
            dists = self.compute_distances_two_loops(x)
        else:
            raise ValueError("Invalid value %d for num_loops"%num_of_loops)

        return self.predict_labels(dists, k=k)

    def __compute_dist(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))

    def compute_distances_two_loops(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i,j] = self.__compute_dist(x[i], self.x_train[j])

        return dists

    def __comp_distance(self, x):
        return np.sqrt(np.sum((x-self.x_train)**2, axis=1))

    def compute_distances_one_loop(self, x):
        num_test = x.shape[0]
        num_train = self.x_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i,:] = self.__comp_distance(x[i])#np.matmul(x[i], self.x_train.T)

        return dists

    def compute_distances_no_loop(self, x):
        #return np.matmul(x, self.x_train.T)
        dists = np.zeros((x.shape[0], self.x_train.shape[0]))
        dists = np.sqrt((x**2).sum(axis=1, keepdims=1)+(self.x_train**2).sum(
            axis=1)-2*x.dot(self.x_train.T))

        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        print(y_pred.shape)
        for i in range(num_test):
            closest_y = []
            test_row = dists[i]
            sorted_row = np.argsort(test_row)
            closest_y = self.y_train[sorted_row[0:k]]

            idx, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = idx[np.argmax(counts)]

        return y_pred
