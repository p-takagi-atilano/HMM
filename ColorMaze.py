# Paolo Takagi-Atilano: COSC 76, October 30, 2017
# refactored from provided "maze.py" to suit my needs for this assignment


class ColorMaze:
    def __init__(self, mazefilename):

        self.robotloc = []
        # read the maze file into a list of strings
        f = open(mazefilename)
        self.lines = []
        for line in f:
            line = line.strip()
            # ignore blank limes
            if len(line) == 0:
                pass
            else:
                self.lines.append(line)
        f.close()

        self.width = len(self.lines[0])
        self.height = len(self.lines)

        self.map = list("".join(self.lines))
        self.colors = set()

        for line in self.lines:
            for char in line:
                if char != '#':
                    self.colors.add(char)

    def get_colors(self):
        return self.colors

    def index(self, x, y):      # maze coordinates now consistent with matrix coordinates
        return y * self.width + x

    # returns char of given tile, None if wall
    def get_tile(self, x, y):
        if x < 0 or x >= self.width:
            return None
        if y < 0 or y >= self.height:
            return None

        index = self.index(x, y)
        if self.map[index] != '#':
            return self.map[index]
        return None

    def __str__(self):

        s = ""
        for line in self.lines:
            s += line
            s += "\n"

        return s

#maze = ColorMaze("maze1.maz")
#print(maze.get_tile(0,0))