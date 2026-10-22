# Understanding `LassoLarsCV` — Notes from a scikit-learn Walkthrough

> Environment: Python 3.10, scikit-learn 1.1.3, using the Titanic dataset from OpenML.

## 1. The Original Snippet (and its bugs)

```python
KFold(n_splits=5, random_states=seed)
reg = LassoLarsCV(cv=folder, normalize=True).fit(X.y)
```

Three issues:

| Issue | Problem | Fix |
|---|---|---|
| `random_states` | Wrong parameter name; also has no effect without `shuffle=True` | `KFold(n_splits=5, shuffle=True, random_state=seed)` |
| `.fit(X.y)` | `X.y` is attribute access, not two arguments | `.fit(X, y)` |
| `normalize=True` | Deprecated in sklearn 1.1, removed in 1.4 | `make_pipeline(StandardScaler(), LassoLarsCV(cv=folder))` |

## 2. What the Code Actually Does

- **`KFold(n_splits=5, shuffle=True, random_state=...)`** creates a cross-validation splitter: data is divided into 5 folds; each fold takes a turn as held-out validation while the other 4 train the model. Every sample is validated exactly once, and the 5 scores are averaged.
- **`LassoLarsCV(cv=folder)`** fits a Lasso (L1-regularized linear regression) where the regularization strength `alpha` is chosen automatically: the LARS algorithm generates a path of candidate alphas, each is scored with the 5-fold procedure, and the alpha with the lowest mean held-out MSE wins. The final model is then refit on *all* the data with that alpha.

In one line: **KFold = the evaluation protocol, LARS = the alpha generator, CV = the judge that picks the winner.**

## 3. scikit-learn Estimator Lifecycle

```python
regx = LassoLarsCV(cv=folder)   # 1. configure — just a container of hyperparameters
reg = regx.fit(X, y)            # 2. learn — fit() returns self, so reg IS regx
reg.alpha_                      # 3. inspect learned attributes (trailing underscore)
reg.predict(X_new)              # 4. inference
```

Key conventions:

- Constructor arguments (e.g. `cv`, `normalize`) are stored verbatim; printing an unfitted estimator just echoes them.
- Anything **learned from data** has a **trailing underscore** (`alpha_`, `coef_`, `intercept_`) and raises `AttributeError` before `.fit()`.
- `predict()` is literally the linear equation: `intercept_ + X @ coef_`. Alpha is "used up" during fitting — it's baked into the coefficients.

## 4. Alpha, L1 and L2

`alpha` is neither L1 nor L2 — it is the **strength knob** applied to a penalty term:

    minimize:  error_on_data + alpha * penalty(coefficients)

| | L1 — Lasso | L2 — Ridge |
|---|---|---|
| Penalty | `alpha * sum(abs(beta))` | `alpha * sum(beta**2)` |
| Effect | Shrinks some coefficients to **exactly zero** | Shrinks all toward zero, never exactly |
| Consequence | Automatic feature selection | Keeps all features, dampened |
| sklearn | `Lasso`, `LassoLarsCV` | `Ridge`, `RidgeCV` |

Hybrid: `ElasticNet` mixes both via `l1_ratio`.

The CV error vs. alpha curve is **U-shaped**: too small → overfitting, too large → underfitting. `alpha_` is the bottom of the U.

## 5. Feature Selection: Shrunk vs. Removed

`coef_` always has **one slot per feature, in the same order as the columns of X**. Eliminated features don't disappear — their slot holds exactly `0.0`.

- **Shrunk** = volume knob turned down. e.g. `parch`: `-0.00498` (no penalty) → `-0.00237` (Lasso). Still used in predictions.
- **Removed** = mute button. Coefficient is exactly `0.0`; the feature contributes nothing and could be deleted from X.

Check programmatically:

```python
np.count_nonzero(reg.coef_)                          # how many features kept
[n for n, c in zip(feature_names, m.coef_) if c != 0]  # which ones
X_reduced = X[:, m.coef_ != 0]                       # reduced dataset
```

### Caveat: zero ≠ ground truth

On the synthetic example (8 features, only 3 real), some pure-noise features got small *nonzero* coefficients (e.g. `-0.017`) because with only 100 samples, noise accidentally correlates with the target and CV chose a weak alpha. Which noise features survive changes with the random seed. **Use both signals:** exact zeros for elimination, plus skepticism of tiny magnitudes. Prediction accuracy and model parsimony are different goals; the "1-standard-error rule" (pick the strongest alpha within one SE of the CV minimum) gives leaner models.

## 6. Titanic Example Results

Predicting survival from `[pclass, sex=female, age, sibsp, parch, fare]`:

| Feature | Coefficient | Interpretation |
|---|---|---|
| `sex=female` | **+0.494** | Strongest positive effect on survival |
| `pclass` | **−0.160** | Higher class number (3rd) hurt survival |
| `sibsp` | −0.043 | Slightly negative |
| `age` | −0.005 | Slightly negative |
| `parch`, `fare` | ~0 (but nonzero) | Weak, kept but nearly silenced |

- CV chose `alpha ≈ 5e-05` (very light penalty → all 6 features kept).
- Inference: 1st-class woman, 25 → predicted survival **0.98**; 3rd-class man, 40, with family → **0.01**.
- Lasso is a linear regression on a 0/1 target, so scores aren't guaranteed to stay in [0, 1]. For real probabilities use `LogisticRegressionCV`.

Data prep gotchas encountered: `pandas` needed installing; one passenger had a missing `fare` (filled with median); `LassoLarsCV` rejects NaN input.

## 7. Migration Path: `normalize=True` → Pipeline

Deprecation timeline:

| sklearn version | `normalize=True` | Pipeline + `StandardScaler` |
|---|---|---|
| ≤ 1.0 | works (default True) | works |
| 1.1 | works, `FutureWarning` | works |
| 1.2–1.3 | still works, default flips to False | works |
| ≥ 1.4 | **TypeError: unexpected keyword argument** | works |

```python
# BEFORE
reg = LassoLarsCV(cv=folder, normalize=True).fit(X, y)

# AFTER (works on 1.1 already — switch before upgrading)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

reg = make_pipeline(StandardScaler(), LassoLarsCV(cv=folder)).fit(X, y)
```

What changes with the pipeline:

1. **Attribute access moves one level down**: `reg.alpha_` → `reg.named_steps["lassolarscv"].alpha_` (or `reg[-1].alpha_`).
2. **`coef_` is in scaled units** — divide by `reg[0].scale_` for raw-unit interpretation. `predict`/`score` are unaffected (inputs are scaled automatically).

### Is `StandardScaler` numerically identical to old `normalize=True`?

Yes (up to floating-point rounding) under standard settings — and it's provable, not luck. Old `normalize=True` **centered each column first, then** scaled by the L2 norm of the centered column, `sqrt(sum((x - mean)**2))`. `StandardScaler` scales by the standard deviation, `sqrt(mean((x - mean)**2))`. Since both center, the two scalings differ by the *same global factor* `sqrt(n_samples)` for every column, and rescaling all features by a global constant is exactly cancelled by rescaling alpha. Algebra: fitting `Lasso(alpha')` on `X_std = sqrt(n) * X_l2scaled` and substituting `beta_std = beta / sqrt(n)` yields the old objective iff `alpha' = alpha * sqrt(n_samples)`. Same objective → same minimizer.

Verified empirically with a fixed, selection-active penalty (`Lasso(alpha=0.001)`, 3 of 6 features zeroed): the compensated pipeline `make_pipeline(StandardScaler(), Lasso(alpha=0.001 * sqrt(n)))` reproduced the old `normalize=True` coefficients with max difference 1.67e-16 (machine epsilon) and the identical selection boundary.

With CV-selected alpha (e.g. `LassoLarsCV`) you don't even need the compensation: CV finds the equivalent alpha automatically in the new scaling. The compensation only matters when porting a manually-fixed alpha.

Cases where exactness genuinely breaks:

- `StandardScaler(with_mean=False)` (the recipe printed in sklearn's warning): without centering, scaling uses the uncentered second moment, which differs *per feature* — only approximate.
- `fit_intercept=False`, or `sample_weight` (also flagged in the warning).
- Different sklearn versions/solvers may differ in the last few digits.

Simplest valid option: since normalization made no difference on this data anyway, plain `LassoLarsCV(cv=folder).fit(X, y)` (no scaling at all) is also future-proof.

## 8. Useful Fitted Attributes for Inspection

```python
reg.alpha_       # CV-chosen regularization strength
reg.coef_        # learned coefficients (0.0 = feature eliminated)
reg.intercept_   # bias term
reg.cv_alphas_   # the alpha candidates LARS produced
reg.mse_path_    # held-out MSE per (fold, alpha); shape (n_folds, n_alphas)
```

## 9. Experiments Worth Trying

- Change `random_state` in `KFold` → fold composition changes, `alpha_` shifts slightly.
- `n_splits=10` → less variance in the CV estimate, more compute.
- Synthetic data: increase noise → CV picks a *larger* alpha (more regularization needed).
- Add `embarked` (one-hot encoded) as a Titanic feature.
- Swap `LassoLarsCV` for `RidgeCV` → no coefficient hits exactly zero.
- Compare `LogisticRegressionCV(cv=folder).fit(X, y).predict_proba(...)` against Lasso's raw scores.
