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
    x_all = []
    y_all =[]
    for i in range(3):
        x = get_random_number(points_number/2, (i*30, i*30+30))
        y = get_random_number(points_number/2, (i*30, i*30+30))
        x_all.append(x)
        y_all.append(y)

    x_all=np.concatenate(np.array(x_all))
    y_all = np.concatenate(np.array(y_all))
    plt.plot(x_all , y_all , 'rs')
    plt.show()
    fp = open("raw_data.txt" , 'wb')
    for i in range(x_all.shape[0]):
        fp.write(str(x_all[i])+" "+str(y_all[i])+'\n')
    fp.close()
    fp = open("dist_data.dat", "wb")
    for i in range(x_all.shape[0]):
        for j in range(x_all.shape[0]):
            dist = np.sqrt(np.square(x_all[i]-x_all[j]) + np.square(y_all[i]-y_all[j]))
            fp.write(str(i+1)+'\t'+str(j+1)+'\t'+str(dist)+'\n')
    fp.close()

