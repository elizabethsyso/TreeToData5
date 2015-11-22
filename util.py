from subprocess import call
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from decimal import Decimal


class LineSegmentDetection(object):
    OUTPUT_DEFAULT = 'tree.txt'
    PGM_PLACEHOLDER = 'tree.pgm'
    LINE_POINTS = ('x1', 'y1', 'x2', 'y2')
    THRESHOLD = 4
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

    def graph(self):
        fig = plt.figure()

        ax = fig.add_subplot(111)

        total_x = []
        total_y = []
        seen_lines = []
        seen_points = []

        for line in self.as_lines():
            x_data = [float(line['x1']), float(line['x2'])]
            y_data = [float(line['y1']), float(line['y2'])]

            distance = self.distance(x_data, y_data)
            if not distance > 20:
                continue

            point1 = float(line['x1']), float(line['y1'])
            point2 = float(line['x2']), float(line['y2'])

            seen_combination = 0
            for point in (point1, point2):
                for seen_point in seen_points:
                    if (seen_point[0] - 4 < point[0] < seen_point[0] + 4 and
                        seen_point[1] - 4 < point[1] < seen_point[1] + 4):
                        seen_combination += 1
                        break

            if seen_combination == 2:
                continue

            seen_points.extend([point1, point2])

            slope = self.slope(x_data, y_data)
            rounded_distance = round(distance)
            potential_line = (rounded_distance, slope)



            # print '*'*10
            # print 'line:', line
            # print 'testing against:', potential_line
            # print have_seen_distance
            # print have_seen_slope
            # print seen_lines

            # if dontadd:
            #     continue

            # seen_lines.append(potential_line)

            total_x.extend(x_data)
            total_y.extend(y_data)
            trace = mlines.Line2D(xdata=x_data, ydata=y_data)
            ax.add_line(trace)

        print total_x, total_y
        print seen_lines
        ax.set_xlim(min(total_x), max(total_x) + 200)
        ax.set_ylim(min(total_y), max(total_y))

        plt.show()







