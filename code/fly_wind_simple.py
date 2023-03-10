
import numpy as np
from scipy import integrate


class FlyWindSimpleCal:
    def __init__(self):

        self.state_names = ['d', 'v', 'w', 'phi', 'zeta']
        self.input_names = ['u_v', 'u_phi']
        self.n = len(self.state_names)  # # of states
        self.m = len(self.input_names)  # # of inputs
        self.p = None
        self.output_mode = None
        self.output_names = []

        # To store simulation data
        self.N = []
        self.t = []
        self.x = []
        self.y = []
        self.u = []
        self.sim_data = []

    def system_ode(self, x, t, tsim, usim):
        """ Dynamical system model of a linear system.

        Inputs
            x: states (array)
            t: time (array)
            tsim: numpy array of simulation time
            usim: numpy array with each input in a column

        Outputs
            xdot: derivative of states (array)
        """

        # Find current inputs based on time in ODE solver
        time_index = np.argmin(np.abs(t - tsim))  # find the closest time to simulation
        usim = np.atleast_2d(usim)
        u = np.squeeze(usim[time_index, :])
        u_v = u[0]
        u_phi = u[1]

        # Get states
        # d, v, w, phi, zeta = x

        # Derivative of states
        xdot = np.array([0,  # d
                         u_v,  # v
                         0,  # w
                         u_phi,  # theta
                         0])   # zeta

        # Return the state derivative
        return np.squeeze(xdot)

    def simulate(self, x0, tsim, usim, output_mode=None):
        self.N = len(tsim)
        self.t = np.atleast_2d(tsim).T  # simulation time vector
        self.u = np.atleast_2d(usim)  # input(s)

        # Solve ODE
        x0 = np.array(x0)
        self.x = integrate.odeint(self.system_ode, x0, tsim, tcrit=tsim, args=(tsim, usim))

        # Get states
        d = self.x[:, 0]
        v = self.x[:, 1]
        w = self.x[:, 2]
        phi = np.unwrap(self.x[:, 3])
        zeta = np.unwrap(self.x[:, 4])

        # Compute ground velocity components
        v_x = v * np.cos(phi)
        v_y = v * np.sin(phi)

        # Ground velocity components in body frame (v always aligned with heading)
        v_x_body = v*0
        v_y_body = v

        # Compute position
        pos_x = integrate.cumtrapz(v_x, tsim, initial=0)
        pos_y = integrate.cumtrapz(v_y, tsim, initial=0)

        # Compute wind velocity components in global & body frame
        w_x = w * np.cos(zeta)
        w_y = w * np.sin(zeta)
        w_x_body = w * np.cos(phi - zeta)
        w_y_body = w * np.sin(phi - zeta)

        # Compute wind & air velocity components & angle
        a_x = -v_x + w_x
        a_y = -v_y + w_y
        gamma = np.unwrap(np.arctan2(a_x, a_y))

        # Compute wind & air velocity components & angle in body frame
        a_x_body = -v_x_body - w_x_body
        a_y_body = -v_y_body + w_y_body
        gamma_body = np.unwrap(np.arctan2(a_x_body, a_y_body))

        # Compute optic flow
        of = v / d

        # Store time, states, & outputs
        self.sim_data = {'time': tsim,
                         'x': self.x,
                         'u': self.u,
                         'd': d,
                         'v': v,
                         'w': w,
                         'phi': phi,
                         'zeta': zeta,
                         'y': self.y,
                         'gamma': gamma,
                         'of': of,
                         'v_x': v_x,
                         'v_y': v_y,
                         'w_x': w_x,
                         'w_y': w_y,
                         'a_x': a_x,
                         'a_y': a_y,
                         'v_x_body': v_x_body,
                         'v_y_body': v_y_body,
                         'w_x_body': w_x_body,
                         'w_y_body': w_y_body,
                         'a_x_body': a_x_body,
                         'a_y_body': a_y_body,
                         'gamma_body': gamma_body,
                         'pos_x': pos_x,
                         'pos_y': pos_y
                         }

        y = self.get_outputs(output_mode=output_mode)
        self.y = y
        self.sim_data['y'] = self.y

        return self.sim_data, self.y

    def set_output_mode(self, output_mode):
        self.output_mode = output_mode

    def get_outputs(self, output_mode=None):
        # Collect outputs
        if output_mode is not None:
            self.output_mode = output_mode

        if isinstance(self.output_mode, list):
            output_names = self.output_mode
        else:
            output_names = self.output_mode.split(',')

        n_output = len(output_names)
        self.p = n_output

        n_point = len(self.sim_data['time'])
        y = np.nan * np.zeros((n_point, n_output))
        self.output_names = output_names
        for n in range(n_output):
            y[:, n] = self.sim_data[output_names[n].strip()]

        return y
