
import numpy as np
from scipy import integrate, signal
import copy


class LinearSystemSimulator:
    def __init__(self, A, B=None, C=None, D=None, system_type='continuous', measurement_type='linear'):
        self.A = np.array(A)  # system matrix

        if B is None:
            self.B = np.zeros_like((len(A), 1))

        if C is None:
            C = np.eye(len(A))

        if D is None:
            D = np.zeros_like(B)

        self.B = np.atleast_2d(np.array(B))  # input matrix
        self.C = np.atleast_2d(np.array(C))  # measurement matrix
        self.D = np.atleast_2d(np.array(D))  # feedforward matrix

        self.p, self.n = C.shape  # [# of outputs, # of states]
        self.m = B.shape[0]  # of inputs

        # Error checks
        if self.p != self.D.shape[0]:
            raise '"C" & "D" matrices must have the same number of columns'

        self.system_type = system_type
        self.measurement_type = measurement_type
        if self.measurement_type != 'linear':
            self.p = 1

        # To store simulation data
        self.N = []
        self.t = []
        self.x = []
        self.y = []
        self.u = []
        self.state = []

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
        u = usim[time_index, :]
        u = np.atleast_2d(u).T

        # if u > 0:
        #     print(u)

        # Derivative of states
        x = np.atleast_2d(x)
        x = np.transpose(x)
        xdot = (self.A @ x) + (self.B @ u)

        # Return the state derivative
        return np.squeeze(xdot)

    def measurement_function(self, x, u, measurement_type='linear'):
        # Set the measurement function & compute it
        if measurement_type == 'linear':  # use given C & D matrices to compute output
            y = (self.C @ x) + (self.D @ u)
        elif measurement_type == 'divide_first_two_states':
            self.y = np.atleast_2d(self.y[:, 0]).T  # only need 1st colum
            y = x[1] / x[0]
        elif measurement_type == 'multiply_first_two_states':
            self.y = np.atleast_2d(self.y[:, 0]).T  # only need 1st column
            y = x[1] * x[0]
        else:
            raise measurement_type + ' measurement type not an option'

        return y

    def simulate(self, x0, tsim, usim, measurement_type=None, system_type=None):
        self.N = len(tsim)
        self.t = np.atleast_2d(tsim).T  # simulation time vector
        self.u = np.atleast_2d(usim)  # input(s)
        self.y = np.zeros((self.N, self.p))  # preallocate output

        # Use defaults if not specified
        if system_type is None:
            system_type = self.system_type

        if measurement_type is None:
            measurement_type = self.measurement_type

        # Solve ODE
        x0 = np.array(x0)
        if system_type == 'continuous':
            # self.x = integrate.odeint(self.system_ode, x0, tsim, tcrit=tsim, args=(tsim, usim))
            sys_scipy = signal.lti(self.A, self.B, self.C, self.D)
            _, _, self.x = signal.lsim(sys_scipy, usim, tsim, X0=x0, interp=True)
        elif system_type == 'discrete':
            self.x = self.simulate_discrete(x0, tsim, usim)

        # Compute output
        for k in range(self.N):
            self.y[k, :] = self.measurement_function(self.x[k, :], self.u[k, :].T, measurement_type)

        # Store time, states, & outputs
        self.state = {'t': tsim,
                      'u': self.u,
                      'x0': x0,
                      'x': self.x,
                      'y': self.y}

        return self.state, self.y

    def simulate_discrete(self, x0, tsim, usim):
        x = copy.copy(x0)
        x = np.atleast_2d(x).T
        for k in range(len(tsim) - 1):
            u = np.atleast_2d(usim[k, :]).T
            xnew = (self.A @ x[:, -1:]) + (self.B @ u)
            x = np.hstack((x, xnew))

        return x.T
