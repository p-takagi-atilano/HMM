import numpy
from ColorMaze import ColorMaze

PROB_SUCCESS = 0.88     # probability of sensor reading correctly

# probabilities of moving in each direction:
PROB_NORTH = 0.25
PROB_EAST = 0.25
PROB_WEST = 0.25
PROB_SOUTH = 0.25


class HMM:
    def __init__(self, maze_filename):
        self.maze = ColorMaze(maze_filename)
        self.colors = list(self.maze.get_colors())  # index of this corresponds to order of sensor matrices
        self.prob_fail = (1 - PROB_SUCCESS)/(len(self.colors) - 1)  # prob of detecting specific wrong color
        self.state_count = self.maze.height * self.maze.width       # number of states  in maze (including walls)

        self.sensor_matrices = self.set_sensor_matrices()
        self.transition_matrix = self.set_transition_matrix()
        self.transposed_transition_matrix = self.transition_matrix.transpose()

    # returns corresponding matrix for each color detected
    def set_sensor_matrices(self):
        sensor_matrices = {}

        # iterate through each color
        for color in self.colors:
            sensor_matrix = numpy.zeros((self.state_count, self.state_count))

            # iterate through each item in maze
            for y in range(self.maze.width):
                for x in range(self.maze.height):
                    c = self.maze.get_tile(x, y)
                    # only change non-wall tiles
                    if c != "#":
                        # set to prob success if same color, set to prob fail otherwise
                        if c == color:
                            prob = PROB_SUCCESS
                        else:
                            prob = self.prob_fail

                        # put in correct location in matrix
                        index = self.index(x, y)
                        sensor_matrix[index, index] = prob

            sensor_matrices[color] = sensor_matrix

        return sensor_matrices

    # returns transition matrix for corresponding maze
    def set_transition_matrix(self):
        transition_matrix = numpy.zeros((self.state_count, self.state_count))

        # iterate through maze
        for y in range(self.maze.width):
            for x in range(self.maze.height):

                if self.maze.get_tile(x, y) is not None:
                    i = self.index(x, y)

                    # prob of robot bumping into wall or wall edge, and such stay in plae
                    this_prob = 0

                    # north
                    if self.maze.get_tile(x, y - 1) is not None:
                        transition_matrix[self.index(x, y - 1), i] = PROB_NORTH
                    else:
                        this_prob += PROB_NORTH

                    # east
                    if self.maze.get_tile(x + 1, y) is not None:
                        transition_matrix[self.index(x + 1, y), i] = PROB_EAST
                    else:
                        this_prob += PROB_EAST

                    # south
                    if self.maze.get_tile(x, y + 1) is not None:
                        transition_matrix[self.index(x, y + 1), i] = PROB_SOUTH
                    else:
                        this_prob += PROB_SOUTH

                    # west
                    if self.maze.get_tile(x - 1, y) is not None:
                        transition_matrix[self.index(x - 1, y), i] = PROB_WEST
                    else:
                        this_prob += PROB_WEST

                    transition_matrix[i, i] = this_prob

        return transition_matrix

    # returns initial probability distribution vector
    def get_initial_distribution(self):
        count = 0
        locs = set()

        # iterate through maze
        for y in range(self.maze.width):
            for x in range(self.maze.height):
                # count all non-maze tiles, remember their location
                if self.maze.get_tile(x, y) is not None:
                    count += 1
                    locs.add((x, y))

        # for all non-maze locations, give corresponding probability
        initial_distribution = numpy.zeros((self.state_count, 1))
        if count != 0:
            prob = 1/count
            for loc in locs:
                index = self.index(loc[0], loc[1])
                initial_distribution[index, 0] = prob

        return initial_distribution

    # creates vector of state size, with every component equaling one, initial value of backwards vector
    def ones_vector(self):
        return numpy.ones((self.state_count, 1))

    # forward filtering algorithm
    def forward(self, sequence, output_filename):
        # get initial distribution vector
        matrix_list = []
        curr_matrix = self.get_initial_distribution()
        matrix_list.append(curr_matrix)

        # write to file if supposed to
        if output_filename is not None:
            f = open(output_filename, "w")
            f.write("time step 0:\n")
            f.write(str(self.write_distribution(curr_matrix)))
            f.write("\n")

        # make sure colors are all valid
        for color in sequence:
            if color not in self.maze.get_colors():
                print("Invalid color sequence.")
                return

        # iterate through sequence
        for i in range(len(sequence)):

            # the math:
            curr_matrix = numpy.dot(self.transposed_transition_matrix, curr_matrix)
            curr_matrix = numpy.dot(self.sensor_matrices[sequence[i]], curr_matrix)
            curr_matrix = curr_matrix/curr_matrix.sum(0)
            matrix_list.append(curr_matrix)

            # write to file if supposed to
            if output_filename is not None:
                # write to file
                f.write("\ntime step ")
                f.write(str(i + 1))
                f.write(":\n")
                f.write(str(self.write_distribution(curr_matrix)))
                f.write("\n")

        # close file if previously used
        if output_filename is not None:
            f.close()

        return matrix_list

    # forward backwards smoothing algorithm
    def forward_backward(self, sequence, output_filename):
        # setup, get forward list and initial backwards vector
        forward_list = self.forward(sequence, None)
        backward_vec = self.ones_vector()

        # new smoothed list, initialized to all None values
        smooth_list = []
        for i in range(len(forward_list)):
            smooth_list.append(None)

        # iterate backwards:
        for i in range(len(forward_list) - 1, 0, -1):
            smooth_list[i] = numpy.multiply(forward_list[i], backward_vec)
            smooth_list[i] = smooth_list[i]/smooth_list[i].sum(0)

            backward_vec = self.backward(backward_vec, sequence[i - 1])

        smooth_list[0] = forward_list[0]

        # write to file
        f = open(output_filename, "w")
        for i in range(len(smooth_list)):
            f.write("time step ")
            f.write(str(i))
            f.write(":\n")
            f.write(str(self.write_distribution(smooth_list[i])))
            f.write("\n\n")
        f.close()

        return smooth_list

    # returns backwards vector given information
    def backward(self, vec, c):
        return numpy.dot(self.transition_matrix, numpy.dot(self.sensor_matrices[c], vec))

    # given maze x and y values, returns corresponding location in matrix column
    def index(self, x, y):
        return y * self.maze.width + x

    # converts vector to matrix corresponding with that of the maze
    def write_distribution(self, matrix):
        distribution = numpy.zeros((self.maze.height, self.maze.width))

        for y in range(self.maze.width):
            for x in range(self.maze.height):
                distribution[x, y] = matrix[self.index(x, y), 0]

        return distribution.transpose()


# Tests:
hmm = HMM("maze1.maz")
hmm.forward(['r', 'g', 'r', 'g'], "output1f.out")
hmm.forward_backward(['r', 'g', 'r', 'g'], "output1fb.out")

hmm = HMM("maze2.maz")
hmm.forward(['g', 'g', 'r', 'b', 'r', 'r', 'r', 'b', 'y'], "output2f.out")
hmm.forward_backward(['g', 'g', 'r', 'b', 'r', 'r', 'r', 'b', 'y'], "output2fb.out")

hmm = HMM("maze3.maz")
hmm.forward(['r', 'g', 'g', 'b', 'r', 'y', 'y', 'y', 'r', 'b'], "output3f.out")
hmm.forward_backward(['r', 'g', 'g', 'b', 'r', 'y', 'y', 'y', 'r', 'b'], "output3fb.out")
