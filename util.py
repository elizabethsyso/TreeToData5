from subprocess import call
# import matplotlib

class LineSegmentDetection(object):
    OUTPUT_DEFAULT = 'tree.txt'
    PGM_PLACEHOLDER = 'tree.pgm'
    LINE_POINTS = ('x1', 'y1', 'x2', 'y2')
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

        _data = filter(None, lines)
        return _data

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


