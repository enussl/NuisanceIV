from .identification_strategy_finder import NIVStrategyFinder

def posinv(A):
    """
    Compute the inverse of a positive-definite matrix using Cholesky decomposition.

    Inputs:
    - A: Positive-definite matrix.

    Outputs:
    - inv: Inverse of matrix A.
    """
    cholesky, info = slg.lapack.dpotrf(A)
    if info != 0:
        raise np.linalg.LinAlgError("Singular or non-pd Matrix.")
    inv, info = slg.lapack.dpotri(cholesky)
    if info != 0:
        raise np.linalg.LinAlgError("Singular or non-pd Matrix.")
    inv += np.triu(inv, k=1).T
    return inv


def col_bind(*args):
    """
    Concatenate arrays column-wise.

    Inputs:
    - args: List of arrays to concatenate.

    Outputs:
    - Concatenated array with columns bound.
    """
    return np.concatenate([get_2d(a) for a in args], axis=1)


def row_bind(*args):
    """
    Concatenate arrays row-wise.

    Inputs:
    - args: List of arrays to concatenate.

    Outputs:
    - Concatenated array with rows bound.
    """
    return np.concatenate(args, axis=0)


def get_2d(a):
    """
    Reshape a 1- or 2-d numpy-array to be 2-dimensional.

    Inputs:
    - a: Array of shape (n,) or (n, m).

    Outputs:
    - Reshaped array of shape (n, 1) or (n, m).
    """
    if len(a.shape) <= 1:
        a = a.reshape(-1, 1)
    return a


def cov(a, b=None):
    """
    Compute cross-covariance matrix between arrays a and b.
    If b is None, covariance matrix of a is returned.

    Inputs:
    - a: Array of shape n_obs OR (n_obs, dims_a)
    - b: None or array of shape n_obs OR (n_obs, dims_b)

    Outputs:
    - Covariance matrix of shape (dims_a, dims_b)
    """
    a = get_2d(a)
    b = a if b is None else get_2d(b)

    d_a = a.shape[1]

    Sigma = np.cov(col_bind(a, b).T)
    return Sigma[:d_a, d_a:]


def civ(
    X,
    Y,
    I,
    B=None,
    N=None,
    W=None,
    predict=lambda x, b: sm.OLS(x, b).fit().predict(),
):
    """
    Compute the causal effect of X,N on Y using instrument I and conditioning set B.

    Inputs:
    - X:        Regressor. numpy array [shape (n_obs,) OR (n_obs, dims_X)]
    - Y:        Response. numpy array [shape (n_obs,) OR (n_obs, dims_Y)]
    - I:        Instrument. numpy array [shape (n_obs,) OR (n_obs, dims_I)]
    - B:        Conditioning set. numpy array [shape (n_obs,) OR (n_obs, dims_B)]
    - N:        Nuisance regressor. numpy array [shape (n_obs,) OR (n_obs, dims_B)]
    - W:        Weight matrix [shape (dims_I, dims_I)]
                or a tuple (Weight matrix W, Weight matrix factor L s.t. W=LL')
    - predict:  function(X, B) that predicts X from B

    Outputs:
    - Estimated causal effect; numpy array (dims_X+dims_N, dims_Y)
    """
    if B is not None:
        r_X = get_2d(X) - get_2d(predict(get_2d(X), get_2d(B)))
        r_Y = get_2d(Y) - get_2d(predict(get_2d(Y), get_2d(B)))
        r_I = get_2d(I) - get_2d(predict(get_2d(I), get_2d(B)))
        if N is not None:
            r_N = get_2d(N) - get_2d(predict(get_2d(N), get_2d(B)))
        return civ(r_X, r_Y, r_I, B=None, W=W, N=(r_N if N is not None else None))
    else:
        if W is None:
            try:
                W = posinv(cov(I))
            except np.linalg.LinAlgError as e:
                e.args += (
                    "Instruments may have degenerate covariance matrix; "
                    "try using less instruments.",
                )
                warnings.warn(
                    e.args[0]
                    + "Instruments may have degenerate covariance matrix; "
                    + "try using less instruments.",
                    slg.LinAlgWarning,
                    stacklevel=3,
                )
                W = np.eye(I.shape[1])

        if type(W) is not tuple:
            W = (W, slg.lapack.dpotrf(W)[0].T)

        regressors = X if N is None else col_bind(X, N)
        covregI = cov(regressors, I)
        covIY = cov(I, Y)
        weights = W[0]
        cho = covregI @ W[1]

        estimates = slg.solve(
            cho @ cho.T, covregI @ weights @ covIY, assume_a="pos"
        )

        return estimates


def align(ref, lagged, tuples=False, min_lag=0):
    """
    Returns appropriately lagged values for time series data.

    Inputs:
    - ref:          Reference time series, lagged at 0. numpy array of shape (n_obs,) or (n_obs, dims)

    - lagged:       List of time series to be lagged relative to ref.
                    Provide either as [X1, lag1, X2, lag2, ...] or as list of tuples [(X1, lag1), (X2, lag2), ...],
                    where each X is numpy array of shape (n_obs,) or (n_obs, dims) and lag is integer

    - tuples:       Indicate whether lagged is provided as list of tuples or plain list

    Outputs: (ref, lagged)
    - ref:          Reference time series (numpy array, shape (n_obs-m, dims)),
                    with appropriately many entries removed in the beginning to have same length as lagged

    - lagged:       List [X1, X2, ...] of lagged time series, each of shape (n_obs, dims)
    """
    if not tuples:
        it = iter(lagged)
        lagged = list(zip(it, it))
    lagged = [(get_2d(x[:-v, ...]) if v > 0 else x) for (x, v) in lagged]
    m = min(x.shape[0] for x in lagged)
    m = min(m, ref.shape[0] - min_lag)
    lagged = [x[(x.shape[0] - m) :, ...] for x in lagged]
    ref = get_2d(ref[(ref.shape[0] - m) :, ...])
    return ref, lagged


def simulate_linear_SCM(G, n_samples):
    """
    Simulate data from a linear Structural Causal Model (SCM) compatible with a directed acyclic graph (DAG).

    Inputs:
    - G:         Directed acyclic graph (DAG, networkx) defining the SCM.
    - n_samples: Number of samples to generate.

    Outputs:
    - df: DataFrame containing the simulated data.
    - true_coefficients: Dictionary of true coefficients for each edge in the graph.
    """
    edge_weights = {edge: np.random.uniform(2, 4) for edge in G.edges()}
    true_coefficients = {}

    data = {}
    for node in nx.topological_sort(G):
        parents = list(G.predecessors(node))
        if not parents:
            data[node] = np.random.normal(0, 1, n_samples)
            true_coefficients[node] = {}
        else:
            data[node] = np.zeros(n_samples)
            true_coefficients[node] = {}
            for parent in parents:
                true_coefficients[node][parent] = edge_weights[
                    (parent, node)
                ]
                data[node] += data[parent] * edge_weights[(parent, node)]
            data[node] += np.random.normal(0, 1, n_samples)
    df = pd.DataFrame(data)
    return df, true_coefficients


def run_simulation_with_seeds_custom(
    G,
    num_samples,
    X_columns,
    Y_column,
    I_columns=None,
    N_columns=None,
    B_columns=None,
    num_seeds=10,
):
    """
    Run a simulation to estimate the causal effect of X on Y using multiple seeds for reproducibility and stability.

    Inputs:
    - G:           Directed acyclic graph (DAG) defining the SCM.
    - num_samples: Number of samples to generate.
    - X_columns:   Columns in the DataFrame corresponding to the regressor X.
    - Y_column:    Column in the DataFrame corresponding to the response Y.
    - I_columns:   Columns in the DataFrame corresponding to the instrument I.
    - N_columns:   Columns in the DataFrame corresponding to the nuisance regressor N.
    - B_columns:   Columns in the DataFrame corresponding to the conditioning set B.
    - num_seeds:   Number of seeds to use for reproducibility.

    Outputs:
    - mean_X: Mean deviation of the estimated causal effect for X.
    - std_X:  Standard deviation of the deviation for X.
    - mean_Z: Mean deviation of the estimated causal effect for Z (if applicable).
    - std_Z:  Standard deviation of the deviation for Z (if applicable).
    """
    deviations = []
    for seed in range(num_seeds):
        np.random.seed(seed)
        data, coefficients = simulate_linear_SCM(G, num_samples)
        X = data[X_columns].to_numpy()
        Y = data[Y_column].to_numpy()
        I = (
            data[I_columns].to_numpy()
            if I_columns is not None
            else None
        )
        N = (
            data[N_columns].to_numpy()
            if N_columns is not None
            else None
        )
        B = (
            data[B_columns].to_numpy()
            if B_columns is not None
            else None
        )

        res = civ(X=X, Y=Y, I=I, N=N, B=B)

        deviation_X = abs(res[0] - coefficients[Y_column][X_columns])
        deviation_Z = (
            abs(res[1] - coefficients[Y_column][N_columns[0]])
            if N_columns is not None
            else None
        )

        deviations.append((deviation_X, deviation_Z))
        values_X = np.array([item[0][0] for item in deviations])
        values_Z = (
            np.array([item[1][0] for item in deviations])
            if N_columns is not None
            else None
        )

        mean_X = values_X.mean()
        std_X = values_X.std()

        mean_Z = (
            values_Z.mean()
            if N_columns is not None
            else None
        )
        std_Z = (
            values_Z.std()
            if N_columns is not None
            else None
        )
    return mean_X, std_X, mean_Z, std_Z


def simulate_linear_VAR(
    alpha,
    beta,
    gamma,
    theta,
    delta,
    T,
    burn_in=1000,
    add_edge=False,
):
    """
    Simulate data from a linear Vector Auto-Regression (VAR).

    Inputs:
    - alpha:   Coefficient for lagged W in the W equation.
    - beta:    Coefficient for W in the P equation.
    - gamma:   Coefficient for lagged D in the D equation.
    - theta:   Coefficient for P in the D equation.
    - delta:   Coefficient for lagged P in the D equation.
    - T:       Total number of time steps to simulate.
    - burn_in: Number of initial samples to discard (for stability).
    - add_edge: Whether to add an extra edge in the graph.

    Outputs:
    - df: DataFrame containing the simulated time series data.
    """
    epsilon = np.random.normal(0, 1, T)
    eta = np.random.normal(0, 1, T)
    zeta = np.random.normal(0, 1, T)
    H = np.random.normal(0, 1, T)

    W = np.zeros(T)
    P = np.zeros(T)
    D = np.zeros(T)

    W[0] = 0.0
    D[0] = 0.0

    if add_edge == False:
        for t in range(1, T):
            W[t] = alpha * W[t - 1] + epsilon[t]
            P[t] = beta * W[t] + H[t] + eta[t]
            D[t] = gamma * D[t - 1] + theta * P[t] + H[t] + zeta[t]
    else:
        for t in range(1, T):
            W[t] = alpha * W[t - 1] + epsilon[t]
            P[t] = beta * W[t] + H[t] + eta[t]
            D[t] = gamma * D[t - 1] + delta * P[t - 1] + theta * P[t] + H[t] + zeta[t]

    data = {
        "W_t": W[burn_in:],
        "P_t": P[burn_in:],
        "P_t-1": P[burn_in - 1 : T - 1],
        "D_t": D[burn_in:],
        "D_t-1": D[burn_in - 1 : T - 1],
        "D_t-2": D[burn_in - 2 : T - 2],
    }

    for i in range(1, 27):
        data[f"W_t-{i}"] = W[burn_in - i : T - i]

    df = pd.DataFrame(data)
    return df
