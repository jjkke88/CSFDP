import matplotlib
matplotlib.use("Qt4Agg")
import numpy as np
from util import *
import matplotlib.pyplot as plt
import parameters as pms

"""process data"""
class DataProcess(object):
    def __init__(self, file_name, delimiter, d_c, rho_min, sigma_min):
        self.file_name = file_name
        self.delimiter = delimiter
        self.d_c = d_c
        self.rho_min = rho_min
        self.sigma_min = sigma_min
        self.get_data_from_file()
        self.calculate_sigma()
        self.plt_show()
        self.get_center()
        self.classify_others()

    def get_data_from_file(self):
        """
        get pre deal data from file
        :param file_name: data file path
        :param delimiter: delimiter
        :return: data, np.array
        """
        raw_data = np.genfromtxt(self.file_name , delimiter=self.delimiter , names=['x', 'y', 'distance'], dtype="i8,i8,f8")
        self.sample_number = raw_data[-1][1] # number of sample points
        distance_matrix = np.zeros((self.sample_number, self.sample_number))

        rho_matrix = np.zeros((self.sample_number, self.sample_number))
        for index in raw_data:
            distance_matrix[index[0] - 1][index[1] - 1] = index[2]
            distance_matrix[index[1] - 1][index[0] - 1] = index[2]
        # distance_vector = np.reshape(distance_matrix, (self.sample_number * self.sample_number))
        # distance_vector_nonzero = []
        # for i in distance_vector:
        #     if i!=0:
        #         distance_vector_nonzero.append(i)
        # distance_vector_sort_nonzero = np.sort(np.array(distance_vector_nonzero))
        # self.d_c = distance_vector_nonzero[self.sample_number * pms.d_c_ratio]
        average_distance = np.mean(distance_matrix)
        self.d_c = average_distance * pms.d_c_ratio
        print "d_c:"+str(self.d_c)
        for index in raw_data:
            rho_matrix[index[0] - 1][index[1] - 1] += kafang_distribution(index[2] - self.d_c)
            rho_matrix[index[1] - 1][index[0] - 1] += kafang_distribution(index[2] - self.d_c)
        self.distance_matrix = distance_matrix
        self.rho_matrix = rho_matrix
        self.rho_vector = np.sum(self.rho_matrix, axis=1)
        print "average_neighbour_number:"+str(np.mean(self.rho_vector))

    def calculate_sigma(self):
        sigma_vector = np.zeros(self.sample_number)
        for i in range(self.sample_number):
            fit_index_array = np.where(self.rho_vector>self.rho_vector[i])[0]  # sample index j that rho_j>rho_i
            if fit_index_array.shape[0]>0:
                fit_distance_vector = self.distance_matrix[i][fit_index_array]
                sigma_vector[i] = np.min(fit_distance_vector)
            else:
                # for the sample point that has largest rho
                sigma_vector[i] = np.max(self.distance_matrix[i])
        self.sigma_vector = sigma_vector

    def get_center(self):
        """
        get cluster center
        :return:
        """
        center = []
        for i in range(self.sample_number):
            if self.rho_vector[i]>self.rho_min and self.sigma_vector[i]>self.sigma_min:
                center.append(i)
        self.center = center

    def classify_others(self):
        result = np.zeros(self.sample_number)
        for center_no in range(len(self.center)):
            result[self.center[center_no]] = center_no
        rho_sort_desc_index = np.argsort(-self.rho_vector)  # sort sample points descent by rho
        for i in range(self.sample_number):
            now_point_index = rho_sort_desc_index[i] # the point now is going to be classified
            if now_point_index not in self.center:
                index_array = rho_sort_desc_index[:i]
                nearest_point_index = self.get_cluster_result(now_point_index, index_array)
                result[now_point_index] = result[nearest_point_index]
            self.result = result

    def calculate_distance(self, index, index_array):
        """
        calculate distance between point index with points in index_array
        :param index:
        :param index_array:
        :return:
        """
        if index_array.shape[0]==0:
            raise NameError("index_array can not be empty")
        return self.distance_matrix[index][index_array]

    def get_cluster_result(self, index, index_array):
        """
        :param index:
        :return:
        """
        index_array = np.append(index_array, self.center)
        distance = self.calculate_distance(index, index_array)
        nearest_index = np.argmin(distance)
        return index_array[nearest_index]

    def plt_show(self):
        ax = plt.gca()
        plt.plot(self.rho_vector , self.sigma_vector , 'rs')
        ax.set_xlabel('rho')
        ax.set_ylabel('sigma')
        plt.show()

if __name__ == "__main__":
    def draw_different_point(raw_data, cluster_result, cls=0):
        plt.figure()
        x = []
        y = []
        color = []
        for raw in range(len(raw_data)):
            if cls is None:
                x.append(raw_data[raw][0])
                y.append(raw_data[raw][1])
                color.append(np.cos(cluster_result[raw] * 5))
            elif cluster_result[raw]==cls:
                x.append(raw_data[raw][0])
                y.append(raw_data[raw][1])
                color.append(np.cos(cluster_result[raw]*5))
        plt.scatter(x , y , c=color , s=25 , marker='o')

    def draw_center_point(raw_data, center, cluster_result):
        plt.figure()
        x = [raw_data[i][0] for i in center]
        y = [raw_data[i][1] for i in center]
        color = [np.cos(cluster_result[i]) for i in center]
        plt.scatter(x , y , c=color, s=25 , marker='o')

    import time
    start = time.time()
    data_process = DataProcess("dist_data.dat", '\t', 0, 90, 20)
    end = time.time()
    print "process time:"+str(end-start)
    # print data_process.result
    raw_data = np.genfromtxt("raw_data.txt", delimiter=" ", names=['x' , 'y'] ,
                             dtype="f8,f8")

    draw_different_point(raw_data, data_process.result, cls=None)
    # draw_center_point(raw_data, data_process.center, data_process.result)
    plt.show()