
import numpy as np
import scipy
import cvxpy
import control


def optimize_matrix_rows_to_states_iterative(O, states=None, alpha=1e-6, beta=1e-3, square=True, sigma='auto',
                                             include_constraints=True, norm=None):
    """ Try to reconstruct a state basis vector (ex: [1 0 0 ... 0]) by combing matrix rows via convex optimization.

        Inputs
            O: observability matrix where the # of columns equals the number of states
            alpha: optimization parameter for promoting sparsity
            beta: reconstruction error bound for optimization constraints
            include_constraints: (boolean) use constraint sin the optimization
            norm: how to normalize the input O matrix. 'mean', 'max', 'min', or None for no normalization

        Outputs:
            output: dictionary with output data
    """

    # Output dict
    out = {'O_subset_iter': [],
           'row_iter': [],
           'ejo_iter': [],
           'E_iter': [],
           'En_iter': [],
           'CN_iter': [],
           'CN_total': [],
           'CN_min': [],
           'CN_min_index': [],
           'O_total': [],
           'row_total': [],
           'iteration_index': 1,
           'states': states}

    O_subset = O.copy()
    O_subset_collection = None
    isobservable = True
    n = 0  # iteration counter
    while isobservable:
        iter_data = optimize_matrix_rows_to_states(O_subset,
                                                   states=states,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   square=square,
                                                   sigma=sigma,
                                                   include_constraints=include_constraints,
                                                   norm=norm)

        isobservable = iter_data['observable']

        # If current iteration is observable
        if isobservable:
            # Set used rows to 0's for next iteration
            rows_iter = (iter_data['row_subset'])
            O_subset[rows_iter, :] = 0

            # Store rows & add to O_subset_iter
            O_subset_iter = np.atleast_2d(O[rows_iter, :])

            # State reconstruction of current iteration
            ejo_iter = []
            E_iter = []
            for j in iter_data['states_data']:
                ejo_iter.append(j['ejo_subset'][-1])
                E_iter.append(j['E_subset'][-1])
            ejo_iter = np.vstack(ejo_iter)
            E_iter = np.vstack(E_iter)
            En_iter = np.linalg.norm(E_iter, ord=2)

            # Condition # of current iteration only
            CN_iter = iter_data['cn_subset']
            # CN_iter - calculate_condition_number(O_subset_iter, sigma=sigma, square=square)

            # Add current subset to collection of all iterations
            if O_subset_collection is not None:
                O_subset_collection = np.vstack((O_subset_collection, O_subset_iter))
            else:
                O_subset_collection = O_subset_iter

            # Compute condition # for collection
            CN_total = calculate_condition_number(O_subset_collection, sigma=sigma, square=square)

            # Collect data for output
            out['row_iter'].append(rows_iter)
            out['O_subset_iter'].append(O_subset_iter)
            out['ejo_iter'].append(ejo_iter)
            out['E_iter'].append(E_iter)
            out['En_iter'].append(En_iter)
            out['CN_iter'].append(CN_iter)
            out['CN_total'].append(CN_total)

        else:
            if n == 0:  # no observable subsets, set everything to nan
                out['row_iter'].append(np.nan)
                out['O_subset_iter'].append(np.nan)
                out['ejo_iter'].append(np.nan)
                out['E_iter'].append(np.nan)
                out['En_iter'].append(np.nan)
                out['CN_iter'].append(np.nan)
                out['CN_total'].append(np.nan)
                out['CN_min_index'] = np.nan
                out['CN_min'] = np.nan
                out['row_total'] = np.nan
                out['O_total'] = np.nan
                out['iteration_index'] = np.nan

            else:  # at least one observable iteration
                out['row_total'] = np.hstack(out['row_iter'])
                out['O_total'] = np.vstack(out['O_subset_iter'])

                # Minimum condition #
                CN_min_index = np.argmin(out['CN_total'])
                CN_min = out['CN_total'][CN_min_index]
                out['CN_min_index'] = CN_min_index
                out['CN_min'] = CN_min
                out['iteration_index'] = np.arange(1, n + 1, 1)

        n = n + 1  # next iteration

    return out


def optimize_matrix_rows_to_states(O, states=None, alpha=1e-6, beta=0.01, square=True, sigma='auto',
                                   include_constraints=True, norm=None):
    """ Try to reconstruct a state basis vector (ex: [1 0 0 ... 0]) by combing matrix rows via convex optimization.

        Inputs
            O: observability matrix where the # of columns equals the number of states
            alpha: optimization parameter for promoting sparsity
            beta: reconstruction error bound for optimization constraints
            include_constraints: (boolean) use constraint sin the optimization
            norm: how to normalize the input O matrix. 'mean', 'max', 'min', or None for no normalization

        Outputs:
            output: dictionary with output data
    """

    # State basis vectors to reconstruct
    if states is None:  # default is all the states
        states = np.arange(0, 5, 1).tolist()
    else:
        # if states is a list, use the list directly, if not then convert states to a list
        if not isinstance(states[0], list):
            # print('Warning: states should be specified as a list of list for accuracy')
            states = [states]

    # Collect outputs in dict
    out = {'states': states,
           'states_data': [],
           'row_subset': [],
           'O_subset': []}

    # Reconstruct the state basis vector for each state combination
    for j in states:
        data = optimize_matrix_rows_to_state(O,
                                             state=j,
                                             alpha=alpha,
                                             beta=beta,
                                             square=square,
                                             sigma=sigma,
                                             include_constraints=include_constraints,
                                             norm=norm)

        out['states_data'].append(data)
        out['row_subset'].append(data['row_subset'][-1])

    out['row_subset'] = np.unique(np.hstack(out['row_subset']))  # collect all the unique rows across the states

    # If any NaN's in data then at least one state is unobservable
    nanI = np.isnan(out['row_subset'])
    if np.any(nanI):
        out['observable'] = False
        out['row_subset'] = np.nan
        out['O_subset'] = np.nan
        out['cn_subset'] = np.nan
    else:  # all states are observable, so collect the rows of O
        out['observable'] = True
        out['O_subset'] = O[out['row_subset']]
        out['cn_subset'] = calculate_condition_number(out['O_subset'], sigma=sigma, square=square)

    return out


def optimize_matrix_rows_to_state(O, state=1, alpha=1e-6, beta=0.01, square=True, sigma='auto',
                                  include_constraints=True, norm=None):
    """ Try to reconstruct a state basis vector (ex: [1 0 0 ... 0]) by combing matrix rows via convex optimization.

        Inputs
            O: observability matrix where the # of columns equals the number of states
            state: state index to reconstruct. If states has more than one element, reconstruct the combination of states
            alpha: optimization parameter for promoting sparsity
            beta: reconstruction error bound for optimization constraints
            include_constraints: (boolean) use constraint sin the optimization
            norm: how to normalize the input O matrix. 'mean', 'max', 'min', or None for no normalization

        Outputs:
            output: dictionary with output data
    """

    # Number of rows
    n_row = O.shape[0]

    # Number of states
    n_state = O.shape[1]

    # Normalize O, if specified
    On = normalize_matrix(O, norm=norm)

    # State basis vector
    if isinstance(state, list):
        state = np.asarray(state)

    ej = np.zeros([1, n_state])
    ej[0, state] = 1
    # print(ej)

    # Set up loss function
    v = cvxpy.Variable((1, n_row))  # free parameter vector, same n# of rows as O
    E = ej - cvxpy.matmul(v, On)  # reconstruction error
    regularizer = cvxpy.norm1(v)  # regularizer to drive parameters to 0 >>> promotes sparsity
    Loss = cvxpy.norm2(E) + alpha * regularizer  # combined loss function

    # Set up constraints, if specified
    constraints = []
    if include_constraints:
        # Set each element of the reconstruction error to be <= beta
        for j in range(n_state):
            constraints.append(E[0, j] <= beta)

    # Minimize loss function
    objective = cvxpy.Minimize(Loss)
    prob = cvxpy.Problem(objective, constraints)
    prob.solve(solver='MOSEK', verbose=False)

    # Reconstruct the state from full O & optimized parameters in v, then calculate the error
    vo = v.value[0]  # optimized parameter values
    vo_sort_index = np.flip(np.argsort(np.abs(vo), axis=-1))  # sort the optimized variables by value
    ejo = vo @ On  # reconstructed state
    E = np.squeeze(ejo - ej)  # state reconstruction error
    E_beta = np.abs(E) <= beta  # check if the constraints were satisfied
    E_beta_min = np.max(np.abs(E))  # the maximum error (minimum beta) in any column
    observable = np.all(E_beta)  # if constraints were satisfied, the state is observable
    En = np.linalg.norm(E, ord=2)  # error norm

    # Collect outputs in dict
    output = {'On': On,
              'vo': vo,
              'vo_sort_index': vo_sort_index,
              'ej': ej,
              'ejo': ejo,
              'E': E,
              'E_beta': E_beta,
              'E_beta_min': E_beta_min,
              'En': En,
              'observable': observable,

              # For iterative subset method
              'row_subset': [],
              'O_subset': [],
              'vo_subset_all': [],
              'vo_subset': [],
              'ejo_subset': [],
              'E_subset': [],
              'E_beta_subset': [],
              'E_beta_min_subset': [],
              'observable_subset': [],
              'cn_subset': np.nan}

    # If the state is observable, find the smallest subset of rows required to reconstruct it
    if observable:
        # Iteratively add the largest rows of vo until the state becomes observable again
        observable_subset = False
        vo_subset_all = np.zeros_like(vo)  # start with all zeros
        i = 0
        while ~observable_subset:  # while unobservable
            row_subset = vo_sort_index[0:(i + 1)]  # add row corresponding to next largest vo variable
            O_subset = O[row_subset]  # subset of O for row subset
            vo_subset_all[row_subset] = vo[row_subset]  # add next largest vo variable
            # vo_subset[vo_sort_index[i]] = vo[vo_sort_index[i]]  # add next largest vo variable
            vo_subset = vo[row_subset]  # just the rows of vo used
            ejo_subset = vo_subset_all @ On  # reconstruct state with subset of vo

            # Reconstruct state with subset of vo, this might be off because of normalization
            # ejo_subset = vo_subset @ O_subset

            E_subset = np.squeeze(ejo_subset - ej)  # state reconstruction error for subset
            E_beta_subset = np.abs(E_subset) <= beta  # check if the constraints were satisfied
            E_beta_min_subset = np.max(np.abs(E_subset))  # the maximum error (minimum beta) in any column
            observable_subset = np.all(E_beta_subset)  # if constraints were satisfied, the state is observable

            # Collect subset data
            output['row_subset'].append(row_subset)
            output['O_subset'].append(O_subset)
            output['vo_subset_all'].append(vo_subset_all)
            output['vo_subset'].append(vo_subset)
            output['ejo_subset'].append(ejo_subset)
            output['E_subset'].append(E_subset)
            output['E_beta_subset'].append(E_beta_subset)
            output['E_beta_min_subset'].append(E_beta_min_subset)
            output['observable_subset'].append(observable_subset)

            i = i + 1

        # Calculate condition of subset of O for of final iteration
        output['cn_subset'] = calculate_condition_number(output['O_subset'][-1], sigma=sigma, square=square)

    else:
        # If not able to reconstruct within tolerance then no need to iterate to find subset, set everything as NaN
        output['row_subset'] = [np.nan]
        output['O_subset'] = [np.nan]
        output['vo_subset_all'] = [np.nan]
        output['vo_subset'] = [np.nan]
        output['ejo_subset'] = [np.nan]
        output['E_subset'] = [np.nan]
        output['E_beta_subset'] = [np.nan]
        output['E_beta_min_subset'] = [np.nan]
        output['observable_subset'] = [np.nan]
        output['cn_subset'] = np.nan

    return output


def normalize_matrix(O, norm=None):
    if norm is not None:
        if norm == 'mean':
            On = O / np.mean(O)
        elif norm == 'max':
            On = O / np.max(O)
        elif norm == 'min':
            On = O / np.min(O)
        else:
            On = O / norm
    else:
        On = O.copy()

    return On


def calculate_condition_number(A, svdFlag=True, square=True, sigma=None, return_subspace=False):
    if square:
        A = A.T @ A

    if svdFlag:  # use singular values
        U, E, V = np.linalg.svd(A)
    else:  # use eigenvalues
        E, V = np.linalg.eig(A)

    # Project into observable subspace before calculating CN, if specified
    if sigma is not None:
        if sigma == 'auto':
            sigma = 0.000001 * np.max(E)

        subspace = E > sigma
        E_nonzero = E[subspace]

    else:
        subspace = np.nan
        E_nonzero = E.copy()

    minE = np.min(E_nonzero)
    maxE = np.max(E_nonzero)
    CN = maxE / minE

    if return_subspace:
        return CN, subspace
    else:
        return CN


def rank_test(O, states=None, tol=None):
    """ Evaluate the observability of states based on the observability matrix O.

        Inputs
            O:              observability matrix
            states:         states of interest, default is all states

        Outputs
            observability:  observability gramian
    """

    n_state = O.shape[1]

    # Get rank of O
    O_rank = np.linalg.matrix_rank(O, tol)

    # Set states to evaluate
    if states is None:
        states = np.arange(0, n_state, 1)
    else:
        states = np.array(states)

    # Evaluate each state
    observability = np.zeros(n_state)  # to store the observability of each state
    for s in states:
        # Create state vector
        state_vector = np.zeros((1, n_state))
        state_vector[0, s] = 1

        # Augment O
        O_augmented = np.vstack((O, state_vector))

        # Get rank of augmented O
        O_augmented_rank = np.linalg.matrix_rank(O_augmented, tol)

        # If ranks are equal, then the state is observable
        if O_augmented_rank == O_rank:
            observability[s] = 1

    return observability


def analytical_observability_gramian(A, C, system_type, use_observability_matrix=False, n_derivatives=1000):
    """ Calculate the analytical observability gramian for a given system A & C matrices.
        Inputs
            A:              system transition matrix (n x n)
            C:              system measurement/output matrix (n x p)
            system_type:    'continuous' or 'discrete'
            n_derivatives:  # of derivatives to use when calculating discrete observability gramian,
                            has no effect for continuous time system
        Outputs
            Wo:             observability gramian
    """

    # Calculate observability gramian
    if use_observability_matrix:  # use observability matrix
        Oa = control.obsv(A, C)
        Wa = Oa.T @ Oa

        return Wa, Oa  # return gramian & observability matrix
    else:
        # Calculate observability gramian based on system type
        if system_type == 'continuous':
            # Solve the Lyapunov equation
            Wa = -scipy.linalg.solve_continuous_lyapunov(A.T, C.T @ C)
        elif system_type == 'discrete':
            # Discrete summation
            Wa = np.zeros_like(A)
            for t in range(0, n_derivatives):
                Wa += np.linalg.matrix_power(A.T, t) @ C.T @ C @ np.linalg.matrix_power(A, t)
        else:
            raise Exception('"system_type" must be "continuous" or "discrete"')

        return Wa  # return gramian
