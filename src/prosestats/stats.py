from collections import defaultdict, Counter
import numpy as np
import scipy.stats as scstats

def invert_dict(data):
    ret = defaultdict(dict)
    for m, dct in data.items():
        for n, v in dct.items():
            ret[n][m] = v
    return ret

def flatten_dict(data):
    ret = {}
    for k, vs in data.items():
        for k_, v in vs.items():
            ret[k,k_] = v
    return ret

def nominal_agreement(a,b):
    return int(a==b)
def ordinal_agreement(a,b, V):
    return 1 - abs(a - b)/(V-1)

def nominal_metric(a, b):
    return a != b
def interval_metric(a, b):
    return (a-b)**2
def ratio_metric(a, b):
    return ((a-b)/(a+b))**2
def ordinal_metric(N_v, a, b):
    a, b = min(a,b), max(a,b)
    return (sum(N_v[a:b+1]) - (N_v[a] + N_v[b])/2)**2

def compute_alpha(data, metric="ordinal", V=None):
    """
    Computes Krippendorff's alpha for a coding matrix V.
    V implicitly represents a M x N matrix where M is the number of
        coders and N is the number of instances.
    In reality, to represent missing values, it is a dictionary of M
    dictionaries, with each dictionary having some of N keys.
    @V is the number of elements.
    """
    data_ = invert_dict(data)
    if V is None:
        V = max({v for worker in data.values() for v in worker.values()})+1

    O = np.zeros((V, V))
    E = np.zeros((V, V))
    for _, dct in data_.items():
        if len(dct) <= 1: continue
        o = np.zeros((V,V))
        for m, v in dct.items():
            v = int(v)
            for m_, v_ in dct.items():
                v_ = int(v_)
                if m != m_:
                    o[v, v_] += 1
        M_n = len(dct)
        O += o/(M_n - 1)
    N_v = O.sum(0)
    E = (np.outer(N_v, N_v) - N_v * np.eye(V))/(sum(N_v)-1)

    if metric == "nominal":
        metric = nominal_metric
    elif metric == "interval":
        metric = lambda a, b: interval_metric(a/V, b/V)
    elif metric == "ratio":
        metric = ratio_metric
    elif metric == "ordinal":
        metric = lambda v, v_: ordinal_metric(N_v, v, v_)
    else:
        raise ValueError("Invalid metric " + metric)

    delta = np.array([[metric(v, v_) for v in range(V)] for v_ in range(V)])
    D_o = (O * delta).sum()
    D_e = (E * delta).sum()

    return 1 - D_o/D_e

def test_compute_alpha():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = {
        'A': {6:2, 7:3, 8:0, 9:1, 10:0, 11:0, 12:2, 13:2, 15:2,},
        'B': {1:0, 3:1, 4:0, 5:2, 6:2, 7:3, 8:2,},
        'C': {3:1, 4:0, 5:2, 6:3, 7:3, 9:1, 10:0, 11:0, 12:2, 13:2, 15:3,},
        }
    assert np.allclose(compute_alpha(data, "nominal", 4), 0.691, 5e-3)
    assert np.allclose(compute_alpha(data, "interval", 4), 0.811, 5e-3)
    assert np.allclose(compute_alpha(data, "ordinal", 4), 0.807, 5e-3)

def outliers_modified_z_score(ys, threshold = 3.5):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)

def outliers_modified_z_score_one_sided(ys, threshold = 3.5):
    median_y = np.median(ys)
    median_absolute_deviation_y = np.median([np.abs(y - median_y) for y in ys])
    modified_z_scores = [0.6745 * (y - median_y) / median_absolute_deviation_y
                         for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)

def compute_pearson_rho(data):
    """
    Given matrix of (task, worker_responses),
        computes pairs of (worker_response - avg) for every task
        and averages over all workers.
    """
    data_ = invert_dict(data)
    # get task means.
    pearson_data = defaultdict(list)
    for _, worker_responses in data_.items():
        # compute pairs,
        if len(worker_responses) < 2: continue
        avg = np.mean(list(worker_responses.values()))
        for worker, response in worker_responses.items():
            pearson_data[worker].append([response, avg])
    micro_data = np.array(sum(pearson_data.values(), []))
    micro, _ = scstats.pearsonr(micro_data.T[0], micro_data.T[1])
    macros = []
    for _, vs in pearson_data.items():
        if len(vs) < 3: continue
        vs = np.array(vs)
        rho, _ = scstats.pearsonr(vs.T[0], vs.T[1])
        if not np.isnan(rho): # uh, not sure how to handle this...
            macros.append(rho)
    return micro #, np.mean(macros)

def test_pearson_rho():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = {
        'A': {6:2, 7:3, 8:0, 9:1, 10:0, 11:0, 12:2, 13:2, 15:2,},
        'B': {1:0, 3:1, 4:0, 5:2, 6:2, 7:3, 8:2,},
        'C': {3:1, 4:0, 5:2, 6:3, 7:3, 9:1, 10:0, 11:0, 12:2, 13:2, 15:3,},
        }
    micro, macro = compute_pearson_rho(data)
    assert np.allclose(micro, 0.947, 5e-3)
    assert np.allclose(macro, 0.948, 5e-3)

def compute_tmean(data, mode="function", frac=0.1):
    if mode == "worker":
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.append(scstats.trim_mean(vs, frac))
        return np.mean(ret)
    else:
        data = invert_dict(data)
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.extend(vs)
        return scstats.trim_mean(ret, frac)

def compute_mean(data, mode="function"):
    if mode == "worker":
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.extend(vs)
        return np.mean(ret)
    else:
        data = invert_dict(data)
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.append(np.mean(vs))
        return np.mean(ret)

def compute_median(data):
    ret = []
    for vs in data.values():
        vs = list(vs.values())
        ret.extend(vs)
    return np.median(ret)

def compute_std(data, mode="worker"):
    data = invert_dict(data)
    if mode == "worker":
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.extend((vs - np.mean(vs)).tolist())
        return np.std(ret)
    else:
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.append(np.mean(vs))
        return np.std(ret)

def compute_nu(data):
    return compute_std(data, "worker")/compute_std(data, "function")

def compute_mad(data, mode="worker"):
    data = invert_dict(data)
    if mode == "worker":
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.extend((vs - np.median(vs)).tolist())
        return np.mean(np.abs(ret))
    else:
        ret = []
        for vs in data.values():
            vs = list(vs.values())
            ret.append(np.mean(vs))
        return np.mean(np.abs(ret - np.median(ret)))

def compute_agreement(data, mode="nominal", V=None):
    """
    Computes simple agreement by taking pairs of data and simply computing probabilty that they agree.

    @V is range of values
    """
    if V is None:
        V = len({v for worker in data.values() for v in worker})

    if mode == "nominal":
        f = nominal_agreement
    elif mode == "ordinal":
        f = lambda a,b: ordinal_agreement(a, b, V)
    else:
        raise ValueError("Invalid mode {}".format(mode))

    data_ = invert_dict(data)
    ret, i = 0., 0
    for _, worker_responses in data_.items():
        # compute pairs,
        if len(worker_responses) < 2: continue
        responses = sorted(worker_responses.values())
        # get pairs
        for j, r in enumerate(responses):
            for _, r_ in enumerate(responses[j+1:]):
                # compute probability
                ret += (f(r,r_) - ret)/(i+1)
                i += 1
    return ret

def test_compute_agreement_nominal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = {
        'A': {6:2, 7:3, 8:0, 9:1, 10:0, 11:0, 12:2, 13:2, 15:2,},
        'B': {1:0, 3:1, 4:0, 5:2, 6:2, 7:3, 8:2,},
        'C': {3:1, 4:0, 5:2, 6:3, 7:3, 9:1, 10:0, 11:0, 12:2, 13:2, 15:3,},
        }
    agreement = compute_agreement(data, "nominal")
    assert np.allclose(agreement, 0.75, 5e-3)

def test_compute_agreement_ordinal():
    # Example from http://en.wikipedia.org/wiki/Krippendorff's_Alpha
    data = {
        'A': {6:2, 7:3, 8:0, 9:1, 10:0, 11:0, 12:2, 13:2, 15:2,},
        'B': {1:0, 3:1, 4:0, 5:2, 6:2, 7:3, 8:2,},
        'C': {3:1, 4:0, 5:2, 6:3, 7:3, 9:1, 10:0, 11:0, 12:2, 13:2, 15:3,},
        }
    agreement = compute_agreement(data, "ordinal")
    assert np.allclose(agreement, 0.75, 5e-3)

def _factorize(data):
    """
    Try to learn turker and task scores as a linear model.
    """
    workers = sorted(data.keys())
    tasks = sorted({hit for hits in data.values() for hit in hits})
    n_entries = sum(len(hits) for hits in data.values())

    X = np.zeros((n_entries, len(workers)+len(tasks)))
    Y = np.zeros(n_entries)
    i = 0
    for worker, hits in data.items():
        for task, value in hits.items():
            X[i, workers.index(worker)] = 1
            X[i, len(workers) + tasks.index(task)] = 1
            Y[i] = value
            i += 1

    return X, Y

def compute_mean_agreement(data, mode="nominal", V=None):
    if V is None:
        V = len({v for worker in data.values() for v in worker})

    if mode == "nominal":
        f = nominal_agreement
    elif mode == "ordinal":
        f = lambda a,b: ordinal_agreement(a, b, V)
    else:
        raise ValueError("Invalid mode {}".format(mode))

    data_ = invert_dict(data)
    ret, i = 0., 0
    for _, worker_responses in data_.items():
        # compute pairs,
        if len(worker_responses) < 2: continue
        responses = sorted(worker_responses.values())

        if mode == "nominal":
            m = scstats.mode(responses)[0]
        elif mode == "ordinal":
            m = np.mean(responses)

        # get pairs
        for r in responses:
            # compute probability
            ret += (f(m,r) - ret)/(i+1)
            i += 1
    return ret

def compute_worker(data, mode="std"):
    data_ = invert_dict(data)
    if mode == "std":
        Ms = {hit_id: np.mean(list(ws.values())) for hit_id, ws in data_.items()}
    elif mode == "mad":
        Ms = {hit_id: np.median(list(ws.values())) for hit_id, ws in data_.items()}
    elif mode == "iaa":
        Ms = {hit_id: np.median(list(ws.values())) for hit_id, ws in data_.items()}
    else:
        raise ValueError()

    ret = {}
    for worker, responses in data.items():
        vs = [Ms[hit_id] - response for hit_id, response in responses.items()]
        if mode == "std":
            ret[worker] = np.std(vs)
        elif mode == "mad":
            ret[worker] = np.mean(np.abs(vs))
        elif mode == "iaa":
            ret[worker] = np.mean([1 if v == 0 else 0 for v in vs])
    return ret

def compute_task_worker_interactions(data, alpha=0.1):
    """
    Using a mixed-effects model: y = Wx + Za jk
    """
    data = flatten_dict(data)
    keys = sorted(data.keys())
    workers, hits = zip(*keys)
    workers, hits = sorted(set(workers)), sorted(set(hits))

    Y = np.zeros(len(keys) + 2)
    X = np.zeros((len(keys) + 2, len(workers) + len(hits))) # + len(keys)-1))

    wf = [0 for _ in workers]
    hf = [0 for _ in hits]
    for i, (worker, hit) in enumerate(keys):
        Y[i] = data[worker,hit]
        wi, hi = workers.index(worker), hits.index(hit)
        X[i, wi] = 1
        X[i, len(workers) + hi] = 1
        wf[wi] += 1
        hf[hi] += 1
    # constraint: proportional sum of workers = 0
    Y[len(keys)], X[len(keys), :len(workers)] = 0, wf
    # constraint: proportional sum of tasks = 0
    Y[len(keys)+1], X[len(keys)+1, len(workers):] = 0, hf

    model = Ridge(alpha=alpha)#, fit_intercept=False)
    model.fit(X, Y)# - Y.mean())

    mean = model.intercept_
    worker_coefs = model.coef_[:len(workers)]
    hit_coefs = model.coef_[len(workers):]

    residuals = Y - model.predict(X)

    ret = {
        "mean": mean,
        "worker-std": np.std(worker_coefs),
        "task-std": np.std(hit_coefs),
        "residual-std": np.std(residuals),
        }

    return ret
