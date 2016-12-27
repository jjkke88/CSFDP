import matplotlib
# matplotlib.use("Qt4Agg")
import numpy as np
import random
import matplotlib.pyplot as plt




class CreateData(object):
    def __init__(self):
        print "create data"

    def get_random_number(self, number, point_range):
        """
        get random points
        :param point_range:
        :return:
        """
        points = np.zeros((number))
        for i in range(number):
            x = random.uniform(point_range[0], point_range[1])
            points[i] = x
        return points

    def create_data(self, points_number, class_number, raw_data_file, dist_data_file, type="square"):
        """
        for create test data
        :param points_number: points number in each class
        :param class_number:
        :param raw_data_file: file path to store raw data
        :param dist_data_file: file path to store distance data
        :param type: test data type, support ["square", "linear", "poligan"]
        :return:
        """

        x_all = []
        y_all = []
        if type == "square":
            points_range = np.array((0, 100))
            for i in range(class_number):
                x = self.get_random_number(points_number / 2, (i * 30, i * 30 + 30))
                y = self.get_random_number(points_number / 2, (i * 30, i * 30 + 30))
                x_all.append(x)
                y_all.append(y)

            x_all = np.concatenate(np.array(x_all))
            y_all = np.concatenate(np.array(y_all))
        elif type == "linear":
            x = np.array(range(points_number))
            for i in xrange(class_number):
                y = x + i * 50
                x_all.append(x)
                y_all.append(y)
        elif type == "poligan":
            x = np.array(range(points_number))*10
            for i in xrange(class_number):
                y = np.square(x) + x[i] * 20 * x + (i + 1) * 500
                x_all.append(x)
                y_all.append(y)
        elif type == "circle":
            x = np.array(range(points_number))
            for i in xrange(class_number):
                y = np.square(x) + i * 20 * x + (i + 1) * 500
                x_all.append(x)
                y_all.append(y)
        x_all = np.concatenate(np.array(x_all))
        y_all = np.concatenate(np.array(y_all))
        plt.plot(x_all, y_all, 'rs')
        plt.show()
        fp = open(raw_data_file, 'wb')
        for i in range(x_all.shape[0]):
            fp.write(str(x_all[i]) + " " + str(y_all[i]) + '\n')
        fp.close()
        fp = open(dist_data_file, "wb")
        for i in range(x_all.shape[0]):
            for j in range(x_all.shape[0]):
                dist = np.sqrt(np.square(x_all[i] - x_all[j])+np.square(y_all[i] - y_all[j]))
                fp.write(str(i + 1) + '\t' + str(j + 1) + '\t' + str(dist) + '\n')
        fp.close()


if __name__ == "__main__":
    create = CreateData()
    create.create_data(50, 3, "raw_data_poligan.dat", "dist_data_poligan.dat", type="poligan")