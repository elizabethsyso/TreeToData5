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
    THRESHOLD = 6
    DISTANCE_THRESHOLD = 40
    _data = None

    def __init__(self, input_file=None, output_file=None):
        self.input_file = input_file
        self.output_file = None if output_file else self.OUTPUT_DEFAULT
        self.convert()
        self.output_data = self.populate_output()

    def populate_output(self):
        lines = []
        with open(self.output_file, 'r') as file_:
            for line in file_.read().splitlines():
                if not line:
                    continue

                lines.append(line.split(' ')[:4])

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
        x_data = [float(line['x1']), float(line['x2'])]
        y_data = [float(line['y1']), float(line['y2'])]
        return x_data, y_data

    @classmethod
    def parse_points(cls, line):
        point1 = float(line['x1']), float(line['y1'])
        point2 = float(line['x2']), float(line['y2'])
        return point1, point2

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

    @staticmethod
    def distance(xdata, ydata):
        return (abs(xdata[0] - xdata[1])**2 + abs(ydata[0] - ydata[1])**2)**float(0.5)

    @staticmethod
    def slope(xdata, ydata):
        longslope = (ydata[1] - ydata[0]) / (xdata[0] - xdata[1])
        return round(longslope, 2)

    @staticmethod
    def regression(line, point):
        pass

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

    def graph(self):
        fig = plt.figure()

        ax = fig.add_subplot(111)

        total_x = []
        total_y = []
        seen_distances = []
        seen_points = []
        seen_slopes = []

        for line in self.as_lines():
            x_data, y_data = self.parse_line(line)
            point1, point2 = self.parse_points(line)

            duplication_score = 0

            # establish length of line and throw out short ones
            distance = self.distance(x_data, y_data)
            if not distance > self.DISTANCE_THRESHOLD:
                continue

            if round(distance, 2) in seen_distances:
                duplication_score += 1

            # figure out if slope is the same
            slope = self.slope(x_data, y_data)
            if slope in seen_slopes:
                duplication_score += 1

            # figure out if points are within proximity of seen_points
            for point in (point1, point2):
                for seen_point in seen_points:
                    if self.is_in_vicinity(point_of_interest=point,
                                           reference_point=seen_point):
                        duplication_score += 1
                        break

            if duplication_score >= 3:
                continue

            seen_points.extend([point1, point2])
            seen_slopes.append(slope)
            seen_distances.append(distance)

            total_x.extend(x_data)
            total_y.extend(y_data)
            trace = mlines.Line2D(xdata=x_data, ydata=y_data)
            ax.add_line(trace)

        print (total_x, total_y)
        ax.set_xlim(min(total_x), max(total_x) + 200)
        ax.set_ylim(min(total_y), max(total_y))

        plt.show()







