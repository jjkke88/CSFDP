import matplotlib
matplotlib.use("Qt4Agg")
import numpy as np
import random
import matplotlib.pyplot as plt


if __name__ == "__main__":
    def get_random_number(number , point_range):
        points = np.zeros((number))
        for i in range(number):
            x = random.uniform(point_range[0] , point_range[1])
            points[i] = x
        return points
    points_number = 300
    points_range = np.array((0, 100))
    x = get_random_number(points_number/2, points_range/2)
    y = get_random_number(points_number/2, points_range/2)
    x1 = get_random_number(points_number / 2 , points_range / 2+50)
    y1 = get_random_number(points_number / 2 , points_range / 2+50)

    x = np.append(x, x1)
    y = np.append(y, y1)
    plt.plot(x , y , 'rs')
    # plt.show()
    fp = open("raw_data.txt" , 'wb')
    for i in range(points_number):
        fp.write(str(x[i])+" "+str(y[i])+'\n')
    fp.close()
    fp = open("dist_data.dat", "wb")
    for i in range(points_number):
        for j in range(points_number):
            dist = np.sqrt(np.square(x[i]-x[j]) + np.square(y[i]-y[j]))
            fp.write(str(i+1)+'\t'+str(j+1)+'\t'+str(dist)+'\n')
    fp.close()

