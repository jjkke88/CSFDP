import math
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

class CFSDP(object):
    def __init__(self, dc_ratio, delta_ratio):
        self.filename = ""
        self.points = []
        self.dc_ratio = dc_ratio
        self.delta_ratio = delta_ratio
        self.dc = 0
        self.dim = 2
        self.distance = {}
        self.rho = []
        self.rho_sort = []
        self.delta = []
        self.delta_sort = []
        self.centroids = []
        self.clusters = []
        self.num_cluster = 0
        self.max_d = 0.0
        self.min_d = 0.0
        self.threshold_delta = 10.0

    def get_dist(self, p1, p2):
        # euclid dist
        sigma = 0.0
        for i in xrange(self.dim):
            sigma += (p1[i] - p2[i]) ** 2
        return math.sqrt(sigma)

    # dist cutoff = mean of dist * dc_ratio
    # also record distance
    def get_dc(self, points):
        point_num = len(points)
        dist = []
        self.distance = {}

        # calculate distance
        for i in xrange(point_num):
            for j in xrange(point_num):
                if (i, j) not in self.distance:
                    d = self.get_dist(points[i], points[j])
                    dist.append(d)
                    self.distance[(i, j)] = d
                    self.distance[(j, i)] = d

        dist.sort()
        # record dist cutoff
        self.dc = dist[int(len(dist) * self.dc_ratio)]
        # record min and max dist to debug
        self.min_d = dist[0]
        self.max_d = dist[len(dist) - 1]

    def read_data(self, filename):

        self.filename = filename
        input_file = open(filename, 'r')

        # read file here
        for line in input_file.readlines():
            line = line.strip()
            pos = line.split()
            point = []
            # point = [ 1, 2, ... , 0.9 ]
            for i in pos:
                point.append(float(i))
            self.points.append(point)
        self.dim = len(self.points[0])

        self.get_dc(self.points)

        input_file.close()

    def cal_rho(self):
        # define rho function x(dij - dc)
        # if dij - dc < 0 -> x(<0) gets 1
        # else gets 0

        num_points = len(self.points)
        self.rho = [0 for x in xrange(num_points)]
        for i in xrange(num_points):
            for j in xrange(num_points):
                if i != j:
                    if self.distance[(i, j)] < self.dc:
                        self.rho[i] += 1

        # rho_sort = [ (index, rho), ... , () ] sort by rho value
        self.rho_sort = sorted(((x, self.rho[x]) for x in xrange(num_points)), key=lambda v:v[1], reverse=True)

    def cal_delta(self):
        # delta = min (rho j > rho i) (dij)

        max_d = self.max_d
        num_points = len(self.points)
        self.delta = [max_d for x in xrange(num_points)]
        # record nearby centroid
        self.centroids = [-1 for x in xrange(num_points)]
        # use sorted rho list to calculate delta
        for i in xrange(num_points):
            for j in xrange(i):
                centroid_id = self.rho_sort[j][0]
                nearby_id = self.rho_sort[i][0]
                if self.delta[nearby_id] > self.distance[(nearby_id, centroid_id)]:
                    self.delta[nearby_id] = self.distance[(nearby_id, centroid_id)]
                    self.centroids[nearby_id] = centroid_id

        # decide the threshold of centroid
        self.delta_sort = sorted(((x, self.delta[x]) for x in xrange(num_points)), key=lambda v:v[1], reverse=True)
        self.threshold_delta = self.delta_sort[int(len(self.delta_sort) * self.delta_ratio)][-1]

    def write_result(self, points, clusters):
        filename = self.filename.split(".")[0]
        out = open(filename+"-out.txt", "w")

        num_points = len(points)
        for i in xrange(num_points):
            out.write(str(i) + "\t")
            for j in xrange(self.dim):
                out.write(str(points[i][j]) + "\t")
            out.write(str(clusters[i]) + "\n")

        out.close()

    def clustering(self):
        self.cal_rho()
        self.cal_delta()

        print "start clustering.."
        # assign clusters
        num_points = len(self.points)
        self.clusters = [-1 for x in xrange(num_points)]
        cluster = 0

        for i in xrange(len(self.rho_sort)):
            p_id = self.rho_sort[i][0]
            if self.clusters[p_id] == -1 and self.delta[p_id] > self.threshold_delta:
                self.clusters[p_id] = cluster
                cluster += 1
            else:
                if self.clusters[p_id] == -1 and self.clusters[self.centroids[p_id]] != -1:
                    self.clusters[p_id] = self.clusters[self.centroids[p_id]]
        self.num_cluster = cluster

        print "distance cutoff: " + str(self.dc)
        print "delta thres: " + str(self.threshold_delta)
        print "clusters: " + str(self.num_cluster)

        self.write_result(self.points, self.clusters)

    def drawOriginGraph(self, plt, points, clusters, num_cluster):
        plt.title("Origin")
        # has x dimensions
        num_points = len(points)
        # store the list of i-th axis position
        axis = [[] for x in xrange(self.dim)]
        for i in xrange(self.dim):
            for j in xrange(num_points):
                axis[i].append(points[j][i])

        color_code = plt.get_cmap("Oranges")
        for i in range(len(points)):
            # 2 dimension
            plt.plot(axis[0][i], axis[1][i], marker='o', color=color_code(float(clusters[i])/num_cluster))
        plt.xlabel("x")
        plt.ylabel("y")

    def drawProjectedGraph(self, plt, points, clusters, num_cluster):
        plt.title("Projected")
        # project multi-dimensions points to 2-D using PCA
        pca = PCA(n_components=2, copy=True)
        proj_p = pca.fit_transform(points)

        color_code = plt.get_cmap("Oranges")
        for i in range(len(points)):
            # 2 dimension
            plt.plot(proj_p[i][0], proj_p[i][1], marker='o', color=color_code(float(clusters[i]) / num_cluster))
        plt.xlabel("x")
        plt.ylabel("y")

    def drawResultGraph(self, plt, rho, delta, clusters, num_cluster):
        plt.title("Result")
        color_code = plt.get_cmap("Oranges")
        for i in range(len(rho)):
            plt.plot(rho[i], delta[i], marker='o', color=color_code(float(clusters[i])/num_cluster))
        plt.xlabel("rho")
        plt.ylabel("delta")

    def make_plot(self):
        print "start drawing.."
        fig = plt.figure("Clustering", figsize=(12, 7))
        # figure has 1 row 2 col
        plt.subplot(131)
        self.drawOriginGraph(plt, self.points, self.clusters, self.num_cluster)
        plt.subplot(132)
        self.drawProjectedGraph(plt, self.points, self.clusters, self.num_cluster)
        plt.subplot(133)
        self.drawResultGraph(plt, self.rho, self.delta, self.clusters, self.num_cluster)
        plt.show()

if __name__ == "__main__":
    model = CFSDP(dc_ratio=0.2, delta_ratio=0.05)
    model.read_data(filename="raw_data_linear.txt")
    model.clustering()
    model.make_plot()