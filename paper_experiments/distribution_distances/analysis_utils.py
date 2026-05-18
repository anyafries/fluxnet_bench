import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def split_idx_2way(n, rng):
    """Random 50/50 index split."""
    perm = rng.permutation(n)
    mid = n // 2
    return perm[:mid], perm[mid:]


def split_idx_3way(n, rng):
    """Random 1/3, 1/3, 1/3 index split."""
    perm = rng.permutation(n)
    third = n // 3
    return perm[:third], perm[third:2*third], perm[2*third:]


def fit_domain_clf(xtrain_dom, xtest_dom, max_iter=100, max_depth=6, random_state=42):
    """
    Train HGBC to distinguish train (0) from test (1).

    AUC is computed on a held-out 50 % split (no OOB in HGBC); the returned
    classifier is then refit on all data for use in density ratio estimation.
    Returns (clf, heldout_auc).
    """
    x_adv = np.vstack([xtrain_dom, xtest_dom])
    y_adv = np.concatenate([np.zeros(len(xtrain_dom)), np.ones(len(xtest_dom))])

    # Held-out split for an unbiased AUC estimate
    split_rng = np.random.default_rng(random_state)
    perm = split_rng.permutation(len(x_adv))
    mid  = len(x_adv) // 2
    tr_idx, va_idx = perm[:mid], perm[mid:]

    clf_val = HistGradientBoostingClassifier(
        max_iter=max_iter, max_depth=max_depth, random_state=random_state
    )
    clf_val.fit(x_adv[tr_idx], y_adv[tr_idx])
    y_va_pred = clf_val.predict(x_adv[va_idx])
    auc     = roc_auc_score(y_adv[va_idx], clf_val.predict_proba(x_adv[va_idx])[:, 1])
    bal_acc = balanced_accuracy_score(y_adv[va_idx], y_va_pred)

    # Refit on all data so the returned clf is maximally informative for weights
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, max_depth=max_depth, random_state=random_state
    )
    clf.fit(x_adv, y_adv)
    return clf, auc, bal_acc


def density_ratio(clf, x, source, n_train_dom, n_test_dom, clip_quantile=0.99, eps=1e-6):
    """
    Importance weights from a domain classifier.
      source='test_to_train': weight test points to look like train.
      source='train_to_test': weight train points to look like test.
    Returns (w, ess_frac).
    """
    s = clf.predict_proba(x)[:, 1]  # P(test | x)
    s = np.clip(s, eps, 1 - eps)
    if source == "test_to_train":
        w = (n_test_dom / n_train_dom) * (1 - s) / s
    elif source == "train_to_test":
        w = (n_train_dom / n_test_dom) * s / (1 - s)
    else:
        raise ValueError("source must be 'test_to_train' or 'train_to_test'")
    if clip_quantile is not None:
        print(f"Clipping weights at {clip_quantile} quantile (value={np.quantile(w, clip_quantile):.2f})")
        w = np.minimum(w, np.quantile(w, clip_quantile))
    ess = w.sum() ** 2 / (w ** 2).sum()
    return w, ess / len(w)


def shared_support_weights(clf, xtrain_eval, xtest_eval, n_dom_tr, n_dom_te, clip_q=0.99):
    pi_train = clf.predict_proba(xtrain_eval)[:, 1]  # P(test | x)
    pi_test = clf.predict_proba(xtest_eval)[:, 1]

    alpha = n_dom_te / (n_dom_tr + n_dom_te)

    w_train_shared = pi_train / alpha
    w_test_shared = (1 - pi_test) / (1 - alpha)

    if clip_q is not None and clip_q < 1:
        w_train_shared = np.minimum(w_train_shared, np.quantile(w_train_shared, clip_q))
        w_test_shared = np.minimum(w_test_shared, np.quantile(w_test_shared, clip_q))

    return w_train_shared, w_test_shared


def weighted_rmse(y, yhat, w):
    return np.sqrt(np.sum(w * (y - yhat) ** 2) / np.sum(w))
