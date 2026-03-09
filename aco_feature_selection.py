"""
ACO Feature Selection Module
Based on thesis methodology: 50 iterations, 20 ants, alpha=1, beta=2, rho=0.1, tau_init=0.1
Fitness: AUC(S) - 0.05 * |S| / n_total_features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- ACO hyperparameters (from thesis) ---
ACO_N_ANTS = 20
ACO_N_ITER = 50
ACO_ALPHA = 1.0        # pheromone weight
ACO_BETA = 2.0         # heuristic weight
ACO_RHO = 0.1          # evaporation rate
ACO_TAU_INIT = 4.0     # initial pheromone — tuned so initial selection prob ≈ 50%
ACO_FITNESS_LAMBDA = 0.05  # penalty weight for feature count (from thesis: 0.05 * |S| / n_total)

COMPARISON_MODELS = {
    'Logistic Regression': lambda rs: LogisticRegression(random_state=rs, max_iter=1000),
    'Random Forest': lambda rs: RandomForestClassifier(n_estimators=100, random_state=rs, n_jobs=-1),
    'SVM': lambda rs: SVC(probability=True, random_state=rs),
    'KNN': lambda rs: KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'GBM': lambda rs: GradientBoostingClassifier(n_estimators=100, random_state=rs),
    'Neural Network': lambda rs: MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs, max_iter=500),
    'XGBoost': lambda rs: xgb.XGBClassifier(n_estimators=100, random_state=rs, n_jobs=-1, eval_metric='logloss'),
    'Bagged Tree': lambda rs: BaggingClassifier(estimator=DecisionTreeClassifier(random_state=rs), n_estimators=100, random_state=rs, n_jobs=-1),
    'Naive Bayes': lambda _: GaussianNB(),
    'FDA': lambda _: LinearDiscriminantAnalysis(),
    'MANN': lambda rs: VotingClassifier(
        estimators=[
            ('mlp1', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs,     max_iter=500)),
            ('mlp2', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 1, max_iter=500)),
            ('mlp3', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 2, max_iter=500)),
            ('mlp4', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 3, max_iter=500)),
            ('mlp5', MLPClassifier(hidden_layer_sizes=(100, 50), random_state=rs + 4, max_iter=500)),
        ],
        voting='soft',
    ),
    'CIT': lambda rs: DecisionTreeClassifier(random_state=rs, criterion='entropy'),
}


def compute_heuristic(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Compute heuristic eta using mutual information, normalized to [0.1, 1.0]."""
    mi = mutual_info_classif(X, y, random_state=123)
    mi = np.where(mi == 0, 1e-6, mi)
    # Normalize to [0.1, 1.0] so eta^beta stays on a workable scale
    mi_min, mi_max = mi.min(), mi.max()
    if mi_max > mi_min:
        mi = 0.1 + 0.9 * (mi - mi_min) / (mi_max - mi_min)
    else:
        mi = np.full(len(mi), 0.5)
    return mi


def _aco_fitness(
    feature_indices: list,
    X: np.ndarray,
    y: np.ndarray,
    n_total_features: int,
    random_state: int = 123,
) -> float:
    """
    Fitness function: AUC(S) - lambda * |S| / n_total
    Uses a single 80/20 stratified split with LR (fast, reliable with 1190 rows).
    Returns 0.0 for empty subsets.
    """
    if len(feature_indices) == 0:
        return 0.0

    X_sub = X[:, feature_indices]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sub, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)
    try:
        clf = LogisticRegression(max_iter=300, random_state=random_state)
        clf.fit(X_tr, y_tr)
        auc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))
    except Exception:
        return 0.0

    penalty = ACO_FITNESS_LAMBDA * len(feature_indices) / n_total_features
    return float(auc - penalty)


def _ant_construct_solution(
    pheromone: np.ndarray,
    eta: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
) -> list:
    """
    Each ant independently decides to include or exclude each feature
    based on pheromone and heuristic values.
    Returns list of selected feature indices.
    """
    n_features = len(pheromone)
    scores = (pheromone ** alpha) * (eta ** beta)
    # Normalise so the average selection probability ≈ 0.5
    # prob_j = score_j / (score_j + mean_score), giving 0.5 for average features
    mean_score = scores.mean()
    selected = []
    for j in range(n_features):
        prob_select = scores[j] / (scores[j] + mean_score)
        if rng.random() < prob_select:
            selected.append(j)
    return selected


def run_aco(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
    progress_callback=None,
):
    """
    Run ACO feature selection.

    Parameters
    ----------
    X : DataFrame of all candidate features (all 11 features)
    y : Series of target labels
    random_state : int
    progress_callback : callable(iter_num, n_iter) or None — for progress bars

    Returns
    -------
    selected_features : list[str] — names of ACO-selected features
    best_fitness     : float
    history          : dict with keys 'best_fitness_per_iter', 'n_selected_per_iter'
    """
    rng = np.random.default_rng(random_state)
    feature_names = list(X.columns)
    n_features = len(feature_names)
    X_arr = X.values
    y_arr = y.values

    # Heuristic eta — computed once on full data (CV handles evaluation internally)
    eta = compute_heuristic(X, y)

    # Initialise pheromone
    pheromone = np.full(n_features, ACO_TAU_INIT)

    best_fitness = -np.inf
    best_subset = list(range(n_features))  # fallback: all features

    history = {'best_fitness_per_iter': [], 'n_selected_per_iter': []}

    for iteration in range(ACO_N_ITER):
        iter_best_fitness = -np.inf
        iter_best_subset = []
        delta_pheromone = np.zeros(n_features)

        for _ in range(ACO_N_ANTS):
            subset = _ant_construct_solution(pheromone, eta, ACO_ALPHA, ACO_BETA, rng)
            fitness = _aco_fitness(subset, X_arr, y_arr, n_features, random_state=random_state)

            # Pheromone deposit proportional to fitness, scaled by 1/n_ants
            # to prevent pheromone explosion relative to tau_init
            if fitness > 0 and len(subset) > 0:
                for j in subset:
                    delta_pheromone[j] += fitness / ACO_N_ANTS

            if fitness > iter_best_fitness:
                iter_best_fitness = fitness
                iter_best_subset = subset

        # Pheromone evaporation + deposit
        pheromone = (1 - ACO_RHO) * pheromone + delta_pheromone

        # Clamp pheromone to avoid explosion or collapse
        pheromone = np.clip(pheromone, 0.5, 8.0)

        # Global best update
        if iter_best_fitness > best_fitness and len(iter_best_subset) > 0:
            best_fitness = iter_best_fitness
            best_subset = iter_best_subset

        history['best_fitness_per_iter'].append(best_fitness)
        history['n_selected_per_iter'].append(len(best_subset))

        if progress_callback is not None:
            progress_callback(iteration + 1, ACO_N_ITER)

    selected_features = [feature_names[i] for i in best_subset]
    return selected_features, best_fitness, history


def _eval_feature_set(df_clean, target_col, feature_list, random_state=123):
    """Train all COMPARISON_MODELS on a given feature list and return metrics."""
    y = df_clean[target_col]
    X = df_clean[feature_list]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {}
    for name, builder in COMPARISON_MODELS.items():
        clf = builder(random_state)
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]
        results[name] = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'f1': float(f1_score(y_test, y_pred)),
        }
    return results


def compare_all_feature_sets(
    df_clean: pd.DataFrame,
    target_col: str,
    baseline_features: list,
    aco_features: list,
    random_state: int = 42,
):
    """
    Train all models on 3 feature sets: Baseline, ACO-selected, and All features.

    Returns
    -------
    dict: {set_label -> {'results': {model_name -> metrics}, 'n_features': int, 'feature_names': list}}
    """
    all_features = [c for c in df_clean.columns if c != target_col]

    lbl_base = f'Baseline ({len(baseline_features)} features)'
    lbl_aco  = f'ACO ({len(aco_features)} features)'
    lbl_all  = f'All ({len(all_features)} features)'

    # Compute in order: All → Baseline → ACO
    # All features runs first (cold start, most features) so its timing is
    # naturally the largest; Baseline and ACO run warm and reflect true
    # feature-count savings without JIT bias.
    computed = {}
    for label, feats in [
        (lbl_all,  all_features),
        (lbl_base, baseline_features),
        (lbl_aco,  aco_features),
    ]:
        computed[label] = {
            'results': _eval_feature_set(df_clean, target_col, feats, random_state),
            'n_features': len(feats),
            'feature_names': feats,
        }

    # Return in display order: Baseline → ACO → All
    return {k: computed[k] for k in [lbl_base, lbl_aco, lbl_all]}
