import numpy as np
from util import *


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
            rho_matrix[index[0] - 1][index[1] - 1] += kafang_distribution(index[2] - self.d_c)
            rho_matrix[index[1] - 1][index[0] - 1] += kafang_distribution(index[2] - self.d_c)

        self.distance_matrix = distance_matrix
        self.rho_matrix = rho_matrix
        self.rho_vector = np.sum(self.rho_matrix, axis=1)


    def calculate_sigma(self):
        sigma_vector = np.zeros(self.sample_number)
        for i in range(self.sample_number):
            fit_index_array = np.where(self.rho_vector>self.rho_vector[i])[0]  # sample index j that rho_j>rho_i
            if fit_index_array.shape[0]>0:
                fit_distance_vector = self.distance_matrix[i][fit_index_array]
                sigma_vector[i] = np.min(fit_distance_vector)
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
            index_array = rho_sort_desc_index[:i]
            nearest_point_index = self.get_cluster_result(rho_sort_desc_index[i], index_array)
            result[i] = result[nearest_point_index]
        self.result = result

    # def calculate_distance_with_center(self, index):
    #     """
    #     calculater distance with center point
    #     :param index: the index of sample point
    #     :return: distance with center point
    #     """
    #     return self.calculate_distance(index, self.center)
    #
    # def get_first_cluster(self, index):
    #     """
    #     choose a nearest cluster for sample points have largest rho
    #     :param index: the index of sample point
    #     :return: cluster number
    #     """
    #     distance_with_center = self.calculate_distance_with_center(index)
    #     cluster = np.argmin(distance_with_center)
    #     return cluster
    #
    def calculate_distance(self, index, index_array):
        """
        calculate distance between point index with points in index_array
        :param index:
        :param index_array:
        :return:
        """
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

if __name__ == "__main__":
    data_process = DataProcess("data/test.dat", '\t', 0.1, 90, 0.03)
    # print data_process.rho_vector
    # print data_process.rho_vector
    # print
    # print data_process.sigma_vector
    print data_process.result