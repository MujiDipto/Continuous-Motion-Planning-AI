import random
import sys
import itertools
import numpy as np
from problem_spec import ProblemSpec
from robot_config import *
from collections import OrderedDict
# from robot_config import write_robot_config_list_to_file
from tester import *

"""
COMP3702 2020 Assignment 2
@author: Mujibul Islam Dipto
"""
MAX_ANGLE = 165
MIN_ANGLE = -165
PRIMITIVE_STEP = 1e-3


class GraphNode:
    """
    Class representing a node in the state graph. You should create an instance of this class each time you generate
    a sample.
    """

    def __init__(self, spec, config):
        """
        Create a new graph node object for the given config.

        Neighbors should be added by appending to self.neighbors after creating each new GraphNode.

        :param spec: ProblemSpec object
        :param config: the RobotConfig object to be stored in this node
        """
        self.spec = spec
        self.config = config
        self.neighbors = []

    def __eq__(self, other):
        return test_config_equality(self.config, other.config, self.spec)

    def __hash__(self):
        return hash(tuple(self.config.points))

    def get_successors(self):
        return self.neighbors

    @staticmethod
    def add_connection(n1, n2):
        """
        Creates a neighbor connection between the 2 given GraphNode objects.

        :param n1: a GraphNode object
        :param n2: a GraphNode object
        """
        n1.neighbors.append(n2)
        n2.neighbors.append(n1)


def find_graph_path(spec, init_node):
    """
    This method performs a breadth first search of the state graph and return a list of configs which form a path
    through the state graph between the initial and the goal. Note that this path will not satisfy the primitive step
    requirement - you will need to interpolate between the configs in the returned list.

    You may use this method in your solver if you wish, or can implement your own graph search algorithm to improve
    performance.

    :param spec: ProblemSpec object
    :param init_node: GraphNode object for the initial configuration
    :return: List of configs forming a path through the graph from initial to goal
    solution = find_graph_path( )
    output = []
    for 0 <= i < len(solution) - 1
        append interpolate(solution[i], solution[i+1]) to output

    """
    # search the graph

    init_container = [init_node]

    # here, each key is a graph node, each value is the list of configs visited on the path to the graph node
    init_visited = {init_node: [init_node.config]}

    while len(init_container) > 0:
        current = init_container.pop(0)

        if test_config_equality(current.config, spec.goal, spec):
            # found path to goal
            return init_visited[current]

        successors = current.get_successors()
        for suc in successors:
            if suc not in init_visited:
                init_container.append(suc)
                init_visited[suc] = init_visited[current] + [suc.config]

    return None


def test_config_distance(c1, c2, spec):
    max_ee1_delta = 0
    max_ee2_delta = 0
    for i in range(spec.num_segments):
        if abs((c2.ee1_angles[i] - c1.ee1_angles[i]).in_radians()) > max_ee1_delta:
            max_ee1_delta = abs((c2.ee1_angles[i] - c1.ee1_angles[i]).in_radians())

        if abs((c2.ee2_angles[i] - c1.ee2_angles[i]).in_radians()) > max_ee2_delta:
            max_ee2_delta = abs((c2.ee2_angles[i] - c1.ee2_angles[i]).in_radians())

    # measure leniently - allow compliance from EE1 or EE2
    max_delta = min(max_ee1_delta, max_ee2_delta)

    for i in range(spec.num_segments):
        if abs(c2.lengths[i] - c1.lengths[i]) > max_delta:
            max_delta = abs(c2.lengths[i] - c1.lengths[i])

    if max_delta > spec.PRIMITIVE_STEP + spec.TOLERANCE:
        return max_delta
    return 0


def solo_collision_test(config, spec):
    # test environment bounds
    # test angle constraints
    # test length constraints
    # test grapple point constraint
    # test self collision
    # test obstacle collision

    if test_environment_bounds(config) and test_angle_constraints(config, spec) and test_length_constraints(
            config, spec) and test_grapple_point_constraint(config, spec) and test_self_collision(config, spec) \
            and test_obstacle_collision(config, spec, spec.obstacles):
        return True
    else:
        return False


def path_collision_test(start, end, spec):
    configs = interpolate(start, end, spec, 1)

    for c in configs:
        if solo_collision_test(c, spec) is False:
            return False
    return True


def interpolate(start, end, spec, version):
    c1 = []
    c2 = []

    for a in start.ee1_angles:
        c1.append(a.in_radians())

    for d in start.lengths:
        c1.append(d)

    c1 = np.array(c1)

    for a in end.ee1_angles:
        c2.append(a.in_radians())
    for d in end.lengths:
        c2.append(d)

    # find Euclidean distance
    distance = np.linalg.norm(c2 - c1)
    if distance == 0:
        print("Yes")
    numOfSteps = math.floor(distance / spec.PRIMITIVE_STEP)

    delta = np.subtract(c2, c1)
    step_size = []
    for i in range(0, len(delta)):
        step_size.append(delta[i] / numOfSteps)

    step_size = np.array(step_size)

    m2 = []

    for i in range(0, numOfSteps):
        temp = np.multiply(i, step_size)
        temp += c1
        m2.append(temp)

    mid_configs = []
    mid_angles = []

    for i in range(0, len(m2)):
        for j in range(0, spec.num_segments):
            mid_angles.append(Angle(m2[i][j]))

    n = len(mid_angles) / spec.num_segments

    mid_angles = np.array_split(mid_angles, n)

    for i in range(0, numOfSteps):
        mid_configs.append(make_robot_config_from_ee1(spec.grapple_points[0][0], spec.grapple_points[0][1],
                                                      mid_angles[i],
                                                      m2[i][spec.num_segments:]))

    if version == 1:
        return mid_configs[1:-1]
    else:
        return mid_configs


def uniform_sampling(spec):
    w, h = spec.num_segments, 20
    angles = [[0 for x in range(w)] for y in range(h)]
    lengths = [[0 for x in range(w)] for y in range(h)]

    for i in range(0, h):
        for j in range(0, spec.num_segments):
            angles[i][j] = Angle(random.uniform(MIN_ANGLE, MAX_ANGLE))

    MIN_LENGTH = min(spec.min_lengths)
    MAX_LENGTH = max(spec.max_lengths)

    for i in range(0, h):
        for j in range(0, spec.num_segments):
            lengths[i][j] = random.uniform(MIN_LENGTH, MAX_LENGTH)

    configs = []
    for i in range(0, h):
        configs.append(make_robot_config_from_ee1(spec.grapple_points[0][0], spec.grapple_points[0][1], angles[i],
                                                  lengths[i], ee1_grappled=True))

    return configs


def prm(spec, init_node, goal_node):
    # find distance from initial node to goal node
    D = test_config_distance(init_node.config, goal_node.config, spec)
    D = 100000
    samples = []
    graph_nodes = []
    samples.append(goal_node.config)
    while True:
        # generate 20 new samples
        samples += uniform_sampling(spec)
        # samples.append(goal_node.config)
        c = 0
        while c < len(samples):
            if not solo_collision_test(samples[c], spec):
                samples.pop(c)
                continue
            graph_nodes.append(GraphNode(spec, samples[c]))
            c += 1
        # graph_nodes = list(set(graph_nodes))
        graph_nodes = list(OrderedDict.fromkeys(graph_nodes))
        for c in graph_nodes:
            if path_collision_test(init_node.config, c.config, spec):
                GraphNode.add_connection(init_node, c)

        # get every combination of two random samples
        pairs = list(itertools.combinations(graph_nodes, 2))

        # calculate distance between each pair
        for c in pairs:
            d = test_config_distance(c[0].config, c[1].config, spec)
            # perform collision check if d <= D * 1.2
            if d <= D:
                # check self collision
                if solo_collision_test(c[0].config, spec) and solo_collision_test(c[1].config, spec):
                    if path_collision_test(c[0].config, c[1].config, spec):
                        GraphNode.add_connection(c[0], c[1])

        path = find_graph_path(spec, init_node)
        if path is not None:
            return path


def main(arglist):
    input_file = arglist[0]
    output_file = arglist[1]

    spec = ProblemSpec(input_file)

    init_node = GraphNode(spec, spec.initial)
    goal_node = GraphNode(spec, spec.goal)

    steps = []
    path = []
    # steps.append(init_node.config)

    nodes = prm(spec, init_node, goal_node)

    # path = (interpolate(nodes[0], nodes[1], spec, 2))
    for i in range(0, len(nodes) - 1):
        path += interpolate(nodes[i], nodes[i + 1], spec, 2)

    steps += path
    steps.append(goal_node.config)

    #
    #
    # Code for your main method can go here.
    #
    # Your code should find a sequence of RobotConfig objects such that all configurations are collision free, the
    # distance between 2 successive configurations is less than 1 primitive step, the first configuration is the initial
    # state and the last configuration is the goal state.
    #
    #

    if len(arglist) > 1:
        write_robot_config_list_to_file(output_file, steps)

    #
    # You may uncomment this line to launch visualiser once a solution has been found. This may be useful for debugging.
    # *** Make sure this line is commented out when you submit to Gradescope ***
    #
    # v = Visualiser(spec, steps)


if __name__ == '__main__':
    main(sys.argv[1:])
