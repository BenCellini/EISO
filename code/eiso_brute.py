import numpy as np
import itertools
import math
from multiprocessing import Pool


def reconstruct_state(O=None, ej=None, beta=1e-6):
    """ Reconstruct the state basis vector ej from the observability matrix O.
        Find the vector v such that v * O = ej using least-squares approach.

        Inputs
            O: observability matrix
            ej: state index to reconstruct. ex: ej = 1 >>> ej = [1, 0, 0].
                Can also set ej as vector directly: ej = [1, 0, 0].
            beta: reconstruction error bound hyperparameter for optimization constraints

        Outputs:
            isobservable: (boolean) if reconstruction was successful
            ejo: reconstructed state basis vector
            v: least-squares solution
    """

    # Get O
    O = np.atleast_2d(O)
    n_state = O.shape[1]

    # Set the vector to reconstruct
    ej = np.atleast_1d(np.squeeze(np.array(ej)))
    if ej.shape[0] == 1:  # state index given
        s = ej.copy()
        ej = np.zeros((n_state, 1)).T
        ej[0, s] = 1.0
    else:  # vector given
        ej = np.atleast_2d(ej)  # make column vector

    # Try to reconstruct the state using least-squares
    solve_data = np.linalg.lstsq(O.T, ej.T, rcond=None)  # try to find solution to reconstruct state basis vector
    v = solve_data[0].T  # solution
    ejo = v @ O  # reconstruct state basis vector using least-squares solution
    error = ej - ejo  # error between state & reconstructed state
    beta_check = np.abs(error) < beta  # check if any element is over the beta reconstruction tolerance
    isobservable = np.all(beta_check)  # use tolerance to determine if state is (approximately) observable

    return isobservable, ejo, v


def reconstruct_states(O=None, ej=None, beta=1e-6):
    """ Reconstruct the state basis vectors in the list ej from the observability matrix O.
        Find the vector v such that v * O = ej using least-squares approach.

        Inputs
            O: observability matrix
            ej: list of state indices to reconstruct. ex: ej = 1 >>> ej = [1, 0, 0].
                Can also set ej as list of vectors directly: ej = [1, 0, 0].
            beta: reconstruction error bound hyperparameter for optimization constraints

        Outputs:
            allobservable: (boolean) if reconstruction was successful for all states
            isobservable: (boolean list) if reconstruction was successful for each state
            ejo: reconstructed state basis vectors for each state
            v: least-squares solution for each state
    """

    # Make sure ej is a list
    if not isinstance(ej, list):
        ej = [ej]

    # Try to reconstruct each state
    isobservable = []
    ejo = []
    v = []
    for s in ej:  # each state
        isobservable_s, ejo_s, v_s = reconstruct_state(O=O, ej=s, beta=beta)

        # Store data
        isobservable.append(isobservable_s)
        ejo.append(ejo_s)
        v.append(v_s)

    # Test if all states are observable
    allobservable = np.all(np.array(isobservable))

    return allobservable, isobservable, ejo, v


def calculate_condition_number(O=None, ej=None, beta=1e-6, square=False):
    """ Calculate a rank-truncated condition # of the matrix O.

        Inputs
            O: observability matrix
            ej: state index to reconstruct. ex: ej = 1 >>> ej = [1, 0, 0].
                Can also set ej as vector directly: ej = [1, 0, 0].
            beta: reconstruction error bound hyperparameter for optimization constraints
            square: if True, then square the condition number (matches CN of Gramian)

        Outputs:
            cn: condition #
    """

    # Get O
    O = np.atleast_2d(O)

    # Try to reconstruct state from entire O first
    # isobservable, ejo, v = reconstruct_state(O=O, ej=ej, beta=beta)
    isobservable, _, ejo, v = reconstruct_states(O=O, ej=ej, beta=beta)

    if isobservable:  # possible to reconstruct state
        # SVD of O
        U, e, V = np.linalg.svd(O)

        # Augment sigma matrix
        E = np.diag(e)
        while E.shape[0] < U.shape[1]:
            E = np.vstack((E, [0] * E.shape[1]))

        while E.shape[1] < V.shape[0]:
            E = np.hstack((E, np.array([[0] * E.shape[0]]).T))

        # Starting from the largest singular value, try to reconstruct for the state
        # from the rank-truncated O to within beta tolerance.
        # If not successful, then add next largest singular value
        rank_trunc = None
        for n in range(1, e.shape[0] + 1):  # each singular value
            # Rank-truncated O
            Ort = U[:, 0:n] @ E[0:n, :] @ V

            # Try to reconstruct the state with rank-truncated O using least-squares
            # isobservable, ejo, v = reconstruct_state(O=Ort, ej=ej, beta=beta)
            isobservable, _, ejo, v = reconstruct_states(O=Ort, ej=ej, beta=beta)

            # If reconstruction was successful, then break the loop &  use only the
            # singular values used for the rank-truncation in the condition # computation.
            if isobservable:
                rank_trunc = n  # how many singular values to use for the rank-truncation condition #
                break

        # Rank-truncated singular values
        if rank_trunc is not None:  # rank truncation was possible with nonzero singular values
            e_trun = e[0:rank_trunc]
            # sigma0 = np.min(e_trun)

            # Calculate condition # of rank-truncated O
            if e_trun.shape[0] == 1:  # in the case where only one singular value is required, use inverse
                # cn = 1.0 / np.min(e_trun)
                cn = np.max(e_trun) / np.min(e_trun)
            else:  # ratio of max/min singular values
                cn = np.max(e_trun) / np.min(e_trun)

            # Square condition # to match with condition number from Gramian, if specified
            if square:
                cn = cn ** 2

        else:  # near-zero singular values made a difference when reconstructing O from all singular values
            cn = np.inf

    else:  # not possible to reconstruct state
        cn = np.nan

    return cn


class EISO:
    """ Wrapper class for computing the condition # of O, used for parallel processing
    """

    def __init__(self, O, ej, beta=1e-6, square=False):
        self.O = np.atleast_2d(O)
        self.ej = ej
        self.beta = beta
        self.square = square

    def task(self, row_subset):
        Ohat = self.O[row_subset]  # subset of O
        CN = calculate_condition_number(O=Ohat, ej=self.ej, beta=self.beta, square=self.square)

        return CN


def eiso_brute(O=None, ej=None, beta=1e-6, square=False, max_row=None, par=False, show_n_comb=False):
    """ Brute force EISO

        Inputs
            O: observability matrix
            ej: state index to reconstruct. ex: ej = 1 >>> ej = [1, 0, 0].
                Can also set ej as vector directly: ej = [1, 0, 0].
            beta: reconstruction error bound hyperparameter for optimization constraints
            square: if True, then square the condition number (matches CN of Gramian)
            max_row: max # of rows in collection to test
            par: (boolean) use parallel processing if True
            show_n_comb: (boolean) print # of row combinations required if True

        Outputs:
            CN_min: minimum condition # across subsets
            O_min: subset of rows of O corresponding to minimum condition #
            row_min: subset of row indices of O corresponding to minimum condition #
            CN: # condition # for every subset tested
            rows: all row subsets tested
    """

    # Get O
    O = np.atleast_2d(O)
    n_row = O.shape[0]

    # Make object for calculating the condition # of subsets of O
    func = EISO(O, ej, beta=beta, square=square)

    # Set max row collection size
    if max_row is None:
        max_row = n_row

    # Compute total number of combinations required
    calculate_number_of_combinations(n_row, max_row=max_row, show_output=show_n_comb)

    # Different size subsets to test
    row_indices = np.arange(0, n_row, 1)

    # Set max size for subset of O
    row_sizes = np.arange(0, max_row, 1) + 1

    # Cycle through all the different row combinations and compute the condition #
    CN = []
    rows = []
    for r in row_sizes:  # each subset size
        # All combinations of rows for collection size r
        row_subset = itertools.combinations(row_indices, r)  # get all the combinations for each row size

        # Make argument list for each row combination
        items = []  # argument list
        for c in row_subset:
            args = (np.array(c),)
            items.append(args)

        # Run through each combination for a fixed row size
        if par and (r < n_row):  # parallel
            # Start parallel processing
            with Pool() as pool:
                print('Start')
                result = pool.starmap(func.task, items)

        else:  # sequential
            result = []
            for i in items:
                out = func.task(i[0])
                result.append(out)

        # Add conditions #'s to list
        CN.append(np.array(result))
        rows.append(items)

    # Combine results for each row size
    CN = np.hstack(CN)
    rows = sum(rows, [])
    # rows = list(itertools.chain.from_iterable(rows))

    # Find minimum condition # & corresponding minimum subset of O
    if np.all(np.isnan(CN)):  # no observable subsets
        CN_min = np.nan
        row_min = np.nan
        O_min = np.nan

    else:  # some observable subsets
        CN_min_index = np.nanargmin(CN)
        CN_min = CN[CN_min_index]
        row_min = rows[CN_min_index][0]
        O_min = O[row_min]

    return CN_min, O_min, row_min, CN, rows


def calculate_number_of_combinations(n_row, max_row=None, show_output=False):
    # All the row indices
    row_indices = np.arange(0, n_row, 1)

    # Set maximum row collection size
    if max_row is None:
        row_sizes = row_indices + 1
    else:
        row_sizes = np.arange(0, max_row, 1)

    N = 0  # keep track of total combinations required
    for r in row_sizes:  # every different size combination
        nCr = math.comb(n_row, r)
        # nCr = int(math.factorial(n_row) / (math.factorial(r)*math.factorial(n_row - r)))
        N = N + nCr

    if show_output:
        print('Total combinations:', N)

    return N
