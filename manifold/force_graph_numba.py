import numba
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils import check_array

from .utils import compute_edges
from .utils import compute_nodes
from .utils import compute_graph
from .utils import init_layout

from time import time

@numba.njit
def apply_repulsion_force(node1, node2, strength, anti_collision):
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y
    distance = np.sqrt(x_dist * x_dist + y_dist * y_dist)

    if anti_collision:
        distance -= node1.size + node2.size
    if distance > 0:  # Clearly distance is always positive without collision detection
        factor = strength * node1.mass * node2.mass / distance**2
    elif distance < 0:  # If the distance is smaller than the sum of radiuses then increase the repulsion
        factor = 100 * strength * node1.mass * node2.mass

    else:  # If distance is 0 do nothing
        return
        # Apply the force
    node1.dx += x_dist * factor
    node1.dy += y_dist * factor
    node2.dx -= x_dist * factor
    node2.dy -= y_dist * factor


@numba.njit
def apply_repulsion_force_on_nodes(nodes, strength,  anti_collision):
    i = 0
    for node1 in nodes:
        j = i
        for node2 in nodes:
            if j == 0:
                break
            apply_repulsion_force(node1, node2, strength, anti_collision)
            j -= 1
        i += 1


@numba.njit
def apply_gravity_force(node, strength, use_strong_gravity):
    if not use_strong_gravity:
        x_dist = node.x
        y_dist = node.y
        distance = np.sqrt(x_dist * x_dist + y_dist * y_dist)

        if distance > 0:
            factor = node.mass * strength / distance
            node.dx -= x_dist * factor
            node.dy -= y_dist * factor
    else:
        x_dist = node.x
        y_dist = node.y

        factor = node.mass * strength
        node.dx -= x_dist * factor
        node.dy -= y_dist * factor


@numba.njit
def apply_gravity_force_on_nodes(nodes, strength, use_strong_gravity):
    for node in nodes:
        apply_gravity_force(node, strength, use_strong_gravity)


@numba.njit
def apply_attraction_force(node1, node2, strength, edge_weight, edge_weight_influence, distributed_attraction,
                           anti_collision):
    edge_weight = np.power(edge_weight, edge_weight_influence)
    x_dist = node1.x - node2.x
    y_dist = node1.y - node2.y

    distance = np.sqrt(x_dist * x_dist + y_dist * y_dist)
    if anti_collision:
        # Check if the nodes are colliding
        distance -=  node1.size + node2.size

    if distance > 0:
        if not distributed_attraction:
            factor = -strength * edge_weight
        else:
            factor = -strength * edge_weight / node1.mass
        node1.dx += x_dist * factor
        node1.dy += y_dist * factor
        node2.dx -= x_dist * factor
        node2.dy -= y_dist * factor


@numba.njit
def apply_attraction_force_on_nodes(nodes, edges, strength, edge_weight_influence, distributed_attraction,
                                    anti_collision):
    for edge in edges:
        apply_attraction_force(nodes[edge.node1], nodes[edge.node2],
                               strength, edge.weight, edge_weight_influence,
                               distributed_attraction, anti_collision)


@numba.njit
def adjust_speed_and_apply_forces(nodes, speed, speed_efficiency, jitter_tolerance, anti_collision):
    # Auto adjust speed.
    total_swinging = 0.0  # How much irregular movement
    total_effective_traction = 0.0  # How much useful movement

    for n in nodes:
        swinging = np.sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))
        total_swinging += n.mass * swinging
        total_effective_traction += .5 * n.mass * np.sqrt(
            (n.old_dx + n.dx) * (n.old_dx + n.dx) + (n.old_dy + n.dy) * (n.old_dy + n.dy))

    # Optimize jitter tolerance.  The 'right' jitter tolerance for
    # this network. Bigger networks need more tolerance. Denser
    # networks need less tolerance. Totally empiric.
    estimated_optimal_jitter_tolerance = .05 * np.sqrt(len(nodes))
    min_jitter = np.sqrt(estimated_optimal_jitter_tolerance)
    max_jitter = 10
    jt = jitter_tolerance * max(min_jitter,
                                min(max_jitter, estimated_optimal_jitter_tolerance * total_effective_traction / (
                                        len(nodes) * len(nodes))))

    min_speed_efficiency = 0.05

    # Protective against erratic behavior
    if total_swinging / total_effective_traction > 2.0:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .5
        jt = max(jt, jitter_tolerance)

    target_speed = jt * speed_efficiency * total_effective_traction / total_swinging

    if total_swinging > jt * total_effective_traction:
        if speed_efficiency > min_speed_efficiency:
            speed_efficiency *= .7
    elif speed < 1000:
        speed_efficiency *= 1.3

    # But the speed shoudn't rise too much too quickly, since it would
    # make the convergence drop dramatically.
    max_rise = .5
    speed = speed + min(target_speed - speed, max_rise * speed)

    for n in nodes:
        swinging = n.mass * np.sqrt((n.old_dx - n.dx) * (n.old_dx - n.dx) + (n.old_dy - n.dy) * (n.old_dy - n.dy))

        if anti_collision:
            factor = 0.1 * speed / (1.0 + np.sqrt(speed * swinging))
            df = np.sqrt(n.dx ** 2 + n.dy ** 2)
            factor = min(factor * df, 10.) / df
        else:
            factor = speed / (1.0 + np.sqrt(speed * swinging))

        n.x = n.x + (n.dx * factor)
        n.y = n.y + (n.dy * factor)

    return np.array([speed, speed_efficiency])


@numba.njit
def optimize_layout(num_iterations, nodes, edges,
                    # Repusion force params
                    repulsion_strength,
                    # Attraction force params
                    attraction_strength,
                    edge_weight_influence,
                    distributed_attraction,
                    # Gravity force params
                    gravity_strength,
                    use_strong_gravity,
                    # Adjust spedd params
                    speed,
                    speed_efficiency,
                    jitter_tolerance,
                    # Shared params
                    anti_collision
                    ):
    logs = []
    xy = np.stack((nodes.x, nodes.y)).T
    logs.append(xy)
    for i in range(num_iterations):
        for node in nodes:
            node.old_dx = node.dx
            node.old_dy = node.dy
            node.dx = 0
            node.dy = 0

        apply_repulsion_force_on_nodes(nodes, repulsion_strength, anti_collision)
        apply_gravity_force_on_nodes(nodes, gravity_strength, use_strong_gravity)
        apply_attraction_force_on_nodes(nodes, edges, attraction_strength,edge_weight_influence,
                                        distributed_attraction, anti_collision)
        values = adjust_speed_and_apply_forces(nodes, speed, speed_efficiency, jitter_tolerance, anti_collision)
        speed = values[0]
        speed_efficiency = values[1]

        xy = np.stack((nodes.x, nodes.y)).T
        logs.append(xy)

    return logs


class ForceGraph:
    def __init__(self,
                 X = None,
                 # Graph related params
                 n_neighbors=30,
                 metric='correlation',
                 local_connectivity=1,

                 # Initial layout
                 random_state=42,
                 init_mode='pca',

                 # Repusion force params
                 repulsion_strength=1.0,

                 # Attraction force params
                 attraction_strength=1.0,
                 edge_weight_influence=1.0,
                 distributed_attraction=False,

                 # Gravity force params
                 gravity_strength=1.0,
                 use_strong_gravity=False,

                 # Shared params
                 anti_collision=False,

                 # Adjust speed and speed_efficiency params
                 speed=1.0,
                 speed_efficiency=1.0,
                 jitter_tolerance=1.0,

                 # Nodes params
                 mass = 1.0,
                 size = 1.0,

                 # Tuning
                 num_iterations=100
                 ):

        #self.graph = check_array(graph, accept_sparse=True)
        self.X = X
        # Graph related params
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.local_connectivity = local_connectivity

        # Initial layout params
        self.random_state = check_random_state(random_state)
        if self.X is None:
            self.init_mode = 'random'
        else:
            self.init_mode = init_mode

        # Forces params --> Three types of forces
        self.repulsion_strength = repulsion_strength
        self.attraction_strength = attraction_strength
        self.distributed_attraction = distributed_attraction
        self.edge_weight_influence = edge_weight_influence
        self.gravity_strength = gravity_strength
        self.use_strong_gravity = use_strong_gravity
        self.anti_collision = anti_collision

        # Adjust speed and speed_efficiency params
        self.speed = speed
        self.speed_efficiency = speed_efficiency
        self.jitter_tolerance = jitter_tolerance

        # Nodes params
        self.mass = mass
        self.size = size

        self.num_iterations = num_iterations

    def fit(self, X):
        self.graph = compute_graph(X, self.n_neighbors, self.metric, None, self.local_connectivity, 1.0)
        # Initialize layout
        self.pts = init_layout(X, graph=self.graph, random_state=self.random_state, dim=2, init_mode=self.init_mode)
        # Compute nodes and edges
        print('Construct nodes...', end='')
        t0 = time()
        self.nodes = compute_nodes(self.graph, self.pts, self.mass, self.size)
        print('done in %.2fs.' % (time() - t0))

        print('Construct edges...', end='')
        t0 = time()
        self.edges = compute_edges(self.graph)
        print('done in %.2fs.' % (time() - t0))

        print('Optimize layout...')
        self.logs = optimize_layout(self.num_iterations,
                        self.nodes, self.edges,
                        # Repusion force params
                        self.repulsion_strength,
                        # Attraction force params
                        self.attraction_strength,
                        self.edge_weight_influence,
                        self.distributed_attraction,
                        # Gravity force params
                        self.gravity_strength,
                        self.use_strong_gravity,
                        # Adjust spedd params
                        self.speed,
                        self.speed_efficiency,
                        self.jitter_tolerance,
                        # Shared params
                        self.anti_collision)
        return np.array([(node['x'], node['y']) for node in self.nodes])