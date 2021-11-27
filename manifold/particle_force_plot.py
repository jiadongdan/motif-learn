import numpy as np
from scipy.integrate import ode

class Particle:
    def __init__(self, mass, position):
        self.mass = mass
        self.x = position[0]
        self.y = position[1]

    def field(self, x, y):
        a, b = (1.2620581223663445, 1.003005400225809)
        r2 = np.hypot(x - self.x, y - self.y) ** 2
        field_x = self.mass * (x - self.x) * 2 * a * b * np.power(r2, b - 1) / (np.power(r2, b) + 1)
        field_y = self.mass * (y - self.y) * 2 * a * b * np.power(r2, b - 1) / (np.power(r2, b) + 1)
        return np.array([field_x, field_y])

    def field_(self, x, y):
        r2 = np.hypot(x - self.x, y - self.y) ** 2
        field_x = self.mass * (x - self.x) / r2
        field_y = self.mass * (y - self.y) / r2
        return np.array([field_x, field_y])

    def is_close(self, point):
        R = 0.01
        x, y = point
        return np.hypot(x - self.x, y - self.y) < R


class Particles:
    def __init__(self, particles):
        self.particles = particles
        self.field = None
        self.field_norm = None

    def generate_field(self, x, y):
        self.XMIN, self.XMAX = x.min(), x.max()
        self.YMIN, self.YMAX = y.min(), y.max()
        self.field = np.zeros_like(np.array([x, y]))
        for particle in self.particles:
            self.field += particle.field(x, y)
        self.field_x = self.field[0]
        self.field_y = self.field[1]
        self.field_mag = np.hypot(self.field_x, self.field_y)
        self.field_norm = self.field / self.field_mag

    def direction(self, y):
        x0, y0 = y
        f = 0
        for particle in self.particles:
            f += particle.field(x0, y0)
        return f / np.hypot(f[0], f[1])

    def generate_line(self, x0, y0, zoom=3):
        assert (self.field_norm is not None)
        streamline = lambda t, y: list(self.direction(y))
        solver = ode(streamline).set_integrator('vode')

        # Initialize the coordinate lists
        line = [np.array([x0, y0])]

        # Integrate in both the forward and backward directions
        dt = 0.008

        # Solve in both the forward and reverse directions
        for sign in [1, -1]:

            # Set the starting coordinates and time
            solver.set_initial_value([x0, y0], 0)

            # Integrate field line over successive time steps
            while solver.successful():

                # Find the next step
                solver.integrate(solver.t + sign * dt)

                # Save the coordinates
                if sign > 0:
                    line.append(solver.y)
                else:
                    line.insert(0, solver.y)

                # Check if line connects to a charge
                flag = False
                for p in self.particles:
                    if p.is_close(solver.y):
                        flag = True
                        break

                # Terminate line at charge or if it leaves the area of interest
                if flag or not (self.XMIN * zoom < solver.y[0] < self.XMAX * zoom) or \
                        not self.YMIN * zoom < solver.y[1] < self.YMAX * zoom:
                    break
        return np.array(line)
