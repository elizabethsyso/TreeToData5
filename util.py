from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from decimal import Decimal
import math


class LineSegmentDetection(object):
    OUTPUT_DEFAULT = 'tree.txt'
    PGM_PLACEHOLDER = 'tree.pgm'
    LINE_POINTS = ('x1', 'y1', 'x2', 'y2')
    THRESHOLD = 25
    DISTANCE_THRESHOLD = 51
    CLUSTER_SIZE_THRESHOLD = 3
    RESOLUTION = 35
    _data = None
    resolution_values = [value for value in range(0, 5000, RESOLUTION)]


    def __init__(self, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = None if output_file else self.OUTPUT_DEFAULT
        self.convert()
        self.seenx = []
        self.seeny = []
        self.output_data = self.populate_output()
        self.nodes = []
        self.leaves = []

    def populate_output(self):
        lines = set()
        seen_x = set()
        seen_y = set()
        with open(self.output_file, 'r') as file_:
            for line in file_.read().splitlines():
                if not line:
                    continue

                values = map(float, line.split(' ')[:4])
                for operation in (float, round, int):
                    values = map(operation, values)

                if not self.valid_distance((values[:2], values[2:])):
                    continue

                x, y = values[:3:2], values[1::2]

                for dim, seen_dim in zip((x, y), (seen_x, seen_y)):
                    for i, elem in enumerate(dim):
                        dim[i] = self.fuzzy_match(elem, seen_dim)

                seen_x.update(x)
                seen_y.update(y)
                p1, p2 = (x[0], y[0]), (x[1], y[1])
                lines.add((tuple(p1), tuple(p2)))

        return filter(None, lines)

    @classmethod
    def as_pairs(cls, points=None):
        """return points as ``list`` of (x, y) pairs """
        return [points[x:x + 2] for x in range(0, len(points), 2)]

    def as_lines(self):
        return [dict(**{a: b for a, b in zip(self.LINE_POINTS, line)})
                for line in self.output_data]



    @classmethod
    def parse_line(cls, line):
        return (line[0][0], line[1][0]), \
               (line[0][1], line[1][1])

    @property
    def convert_to_pnm_command(self):
        return u'gm convert {} {}'.format(self.input_file,
                                          self.PGM_PLACEHOLDER)

    @property
    def convert_to_svg_command(self):
        return u'./lsd-1.5/lsd {} {}'.format(self.PGM_PLACEHOLDER,
                                             self.output_file)

    def convert(self):
        for step in (self.convert_to_pnm_command,
                     self.convert_to_svg_command):
            call(step, shell=True)

        return True

    @classmethod
    def valid_distance(cls, line):
        distance = cls.distance(*cls.parse_line(line))
        return distance >= cls.DISTANCE_THRESHOLD

    @staticmethod
    def distance(xdata, ydata):
        return (abs(xdata[0] - xdata[1])**2 + abs(ydata[0] - ydata[1])**2)**float(0.5)

    @staticmethod
    def slope(xdata, ydata):
        if (xdata[0] - xdata[1]) == 0:
            return 0
        longslope = (ydata[1] - ydata[0]) / (xdata[0] - xdata[1])
        return round(longslope, 2)


    @classmethod
    def fuzzy_match(cls, num, arr):
        for seen in arr:
            if seen - cls.THRESHOLD < num < seen + cls.THRESHOLD:
                return seen
        return num

    @classmethod
    def is_in_vicinity(cls, point_of_interest, reference_point):
        return (reference_point[0] - cls.THRESHOLD <
                point_of_interest[0] <
                reference_point[0] + cls.THRESHOLD and
                reference_point[1] - cls.THRESHOLD <
                point_of_interest[1] <
                reference_point[1] + cls.THRESHOLD)

    @classmethod
    def contained_on_line(cls, line, point):
        x, y = cls.parse_line(line)
        slope = cls.slope(x, y)

        b = y[0]
        for i in range(math.floor(x[0]), math.floor(x[1]), 1):
            line_point_x = i
            line_point_y = slope * i + b
            if cls.is_in_vicinity(point, (line_point_x, line_point_y)):
                return True

        return False

    @property
    def all_points(self):
        points = []
        for i in range(2):
            points.extend(
                [line[i] for line in self.output_data]
            )
        return points

    def contains(self, cluster, point):
        for cluster_point in cluster:
            if self.is_in_vicinity(point, cluster_point):
                return True
        return False

    def find_all_clusters(self):
        all_points = self.all_points
        first_point = all_points[0]
        clusters = [
            [first_point]
        ]
        for point in self.all_points[1:]:
            belongs_in_existing_cluster = False
            for cluster in clusters[:]:
                belongs_to_cluster = self.contains(cluster, point)
                if belongs_to_cluster:
                    belongs_in_existing_cluster = True
                    cluster.append(point)
                    break
            if not belongs_in_existing_cluster:
                clusters.append(
                    [point]
                )

        return clusters

    def major_clusters(self):
        big_clusters = [cluster for cluster in self.find_all_clusters()
                        if len(cluster) >= self.CLUSTER_SIZE_THRESHOLD]

        return big_clusters

    def define_nodes_and_leaves(self):
        nodes = []
        leaves = []
        for cluster in self.find_all_clusters():
            if len(cluster) >= self.CLUSTER_SIZE_THRESHOLD:
                nodes.append(cluster)
            else:
                leaves.append(cluster)
        self.nodes = nodes
        self.leaves = leaves
        return nodes, leaves

    def simplify_clusters(self):
        simple_clusters = {}
        for cluster in self.find_all_clusters():
            point = cluster[0]
            simple_clusters[point] = cluster

        return simple_clusters

    def find_cluster_points(self):
        if not len(self.nodes):
            self.define_nodes_and_leaves()

        left_most_node = min(self.nodes, key=lambda node: node[0][0])
        connection_line = None
        for line in self.as_lines():
            point1, point2 = line
            if point1[0] == left_most_node[0]:
                connection_line = line
                break

        print connection_line

    def graph(self):
        fig = plt.figure()

        ax = fig.add_subplot(111)

        total_x = []
        total_y = []

        # for line in self.as_lines():
        for line in self.output_data:
            x_data, y_data = self.parse_line(line)
            total_x.extend(x_data)
            total_y.extend(y_data)
            trace = mlines.Line2D(xdata=x_data, ydata=y_data)
            ax.add_line(trace)

        ax.set_xlim(min(total_x) - 40, max(total_x) + 200)
        ax.set_ylim(min(total_y) - 40, max(total_y) + 200)

        self.define_nodes_and_leaves()
        for cluster in self.nodes:
            first_point = cluster[0]
            circ = plt.Circle(first_point, radius=10, color='g', fill=True)
            ax.add_patch(circ)

        for cluster in self.leaves:
            first_point = cluster[0]
            circ = plt.Circle(first_point, radius=10, color='r', fill=True)
            ax.add_patch(circ)


        print list(sorted(self.output_data, key=lambda x: x[0][0]))
        plt.show()











