# Adding a new conflict model

This guide walks through adding a new conflict-style model (target + distractor
inputs driving a single decision variable) end to end, from the drift math to a
name you can pass to `get_model_config()` and the simulators.

It uses the existing `conflict_dsstimflex_weight` model as the worked reference.
Everything here generalizes to any drift-based model, but conflict models touch
the most pieces (a dynamical drift, a flexible stimulus envelope, and an optional
decision-variable weight), so they make the best template.

## The one rule that explains everything: parameters bind by name

There is no positional parameter array threaded through these models. At
simulation time, `ssms/basic_simulators/simulator.py` builds each function's
keyword arguments by **name-matching** the trial's `theta` against the `params`
list registered for that function:

```python
# simulator.make_drift_dict (paraphrased)
drift_params = {
    name: value
    for name, value in theta.items()
    if name in drift_config[drift_name]["params"]
}
drift_fun(**drift_params)
```

The same pattern runs for the boundary and the (optional) weight function. So a
parameter flows to a function **iff its name appears in that function's `params`
list and matches one of the function's keyword arguments.** Everything below is
just keeping four things in agreement:

1. the Python function's signature (the kwargs),
2. the function's `params` list in `base.py`,
3. the model's `param_dict` in `conflict.py`,
4. the C simulator's expectation of the core DDM params (`a`, `z`, `t`).

Mismatch #1↔#2 and a parameter silently never reaches the function. Mismatch
#2↔#3 and you either pass an unused parameter or omit a required one.

## The four layers

```
ssms/basic_simulators/drift_functions.py   (1) the math: drift / weight / support helpers
ssms/config/_modelconfig/base.py           (2) registry: drift_config / weight_config / boundary_config
ssms/config/_modelconfig/conflict.py       (3) the model: param_dict + which fns + which C simulator
ssms/config/_modelconfig/__init__.py       (4) registration: import + add to get_model_config()
```

### Layer 1 — drift / weight functions (`drift_functions.py`)

A drift function takes `t` (timepoints) plus named scalar parameters and returns
a drift timecourse `np.ndarray` the same length as `t`. Build it from the shared
helpers so behavior stays consistent across models:

- **`ds_support_analytic(t, init_p, fix_point, slope)`** — saturating-exponential
  weight dynamics, `x(t) = (init_p - fix_point) e^{-slope·t} + fix_point`. Use one
  per input to model an evolving drift weight (the `ds` in `dsstimflex`).
- **`stimflex_support(t, onset, offset, coh, tau_rise, tau_fall)`** — a coherence
  envelope that rises after `onset`, holds at `coh`, and decays after `offset`,
  with causal exponential rise/fall. This is the single stimulus-envelope helper;
  all conflict drifts build their target/distractor inputs from it.

`conflict_dsstimflex_drift` is the canonical combination — independent `ds`
dynamics per input, each gated by a flexible stimulus envelope:

```python
def conflict_dsstimflex_drift(
    t, tinit=0, dinit=0, tslope=1, dslope=1, tfixedp=1, dfixedp=0,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, toffset=None, doffset=None,
    vtaurise=0.05, vtaufall=0.1, sum_drifts=True,
):
    if toffset is None: toffset = np.max(t)
    if doffset is None: doffset = np.max(t)
    tcohs = stimflex_support(t, tonset, toffset, tcoh, vtaurise, vtaufall)
    dcohs = stimflex_support(t, donset, doffset, dcoh, vtaurise, vtaufall)
    w_t = ds_support_analytic(t, init_p=tinit, fix_point=tfixedp, slope=tslope)
    w_d = ds_support_analytic(t, init_p=dinit, fix_point=dfixedp, slope=dslope)
    return w_t * tcohs + w_d * dcohs   # or column_stack when sum_drifts=False
```

A **weight function** has the same shape — `t` + named params → a timecourse —
but multiplies the decision variable rather than driving it. `weight_window`
reuses `stimflex_support` to build a `[wmin, 1]` gate:

```python
def weight_window(t, tonset=0., toffset=1., donset=0., doffset=1.,
                  wmin=0., wtaurise=0.05, wtaufall=0.1, onoff_method="target"):
    ...
    return stimflex_support(t, onset, offset, 1 - wmin, wtaurise, wtaufall) + wmin
```

Notes:

- **Always handle `t is None`** (default to `np.arange(0, 20, 0.1)`) — the registry
  and some callers probe the function with no `t`.
- If your model is a small tweak of an existing one (only different defaults),
  prefer a `functools.partial` over a new function, as the codebase already does:
  `conflict_dsstimflex_dual_drift = partial(conflict_dsstimflex_drift, sum_drifts=False)`.
- At the bottom of the file, add the `DriftFunction` type alias line for any new
  top-level function (mirrors the existing `name: DriftFunction = name` block).

### Layer 2 — register the function and its parameters (`base.py`)

Add an entry whose `params` list is **exactly the kwargs that should be supplied
from `theta`** (omit `t`, and omit any kwarg you want left at its Python default,
e.g. `sum_drifts`, `onoff_method`):

```python
drift_config = {
    ...
    "conflict_dsstimflex_drift": {
        "fun": df.conflict_dsstimflex_drift,
        "params": ["tinit", "dinit", "tslope", "dslope", "tfixedp", "dfixedp",
                   "tcoh", "dcoh", "tonset", "donset", "toffset", "doffset",
                   "vtaurise", "vtaufall"],
    },
}

weight_config = {
    "weight_window": {
        "fun": df.weight_window,
        "params": ["tonset", "toffset", "donset", "doffset",
                   "wmin", "wtaurise", "wtaufall"],
    },
}
```

A parameter can legitimately appear in **both** the drift `params` and the weight
`params` (e.g. `tonset`/`toffset` drive the envelope *and* the weight window) — it
is routed to both functions by name. That's intended, not a duplicate.

### Layer 3 — define the model (`conflict.py`)

A model config is assembled by `_new_config`, with each parameter declared via
`_new_param(default, lower, upper)`. The `param_dict` is the model's **full**
parameter set: core DDM params (`a`, `z`, `t`) **plus** every drift param **plus**
every weight param **plus** any boundary param. Order is the public parameter
order (used for `params`, `default_params`, `param_bounds`).

```python
def _dsstimflex_weight_param_dict():
    return dict(
        a=_new_param(2.0, 0.3, 3.0),       # core DDM: boundary separation
        z=_new_param(0.5, 0.1, 0.9),       # core DDM: starting point
        t=_new_param(1.0, 1e-3, 2.0),      # core DDM: non-decision time
        tinit=_new_param(2.0, 0.0, 5.0),   # drift params follow...
        dinit=_new_param(2.0, 0.0, 5.0),
        tslope=_new_param(2.0, 0.01, 5.0),
        dslope=_new_param(2.0, 0.01, 5.0),
        tfixedp=_new_param(3.0, 0.0, 5.0),
        dfixedp=_new_param(0.0, 0.0, 5.0),
        tcoh=_new_param(0.5, -1.0, 1.0),
        dcoh=_new_param(-0.5, -1.0, 1.0),
        tonset=_new_param(0.0, 0.0, 1.0),
        donset=_new_param(0.0, 0.0, 1.0),
        vtaurise=_new_param(0.05, 1e-3, 0.5),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
        wmin=_new_param(0.5, 0.0, 1.0),    # weight params follow...
        wtaurise=_new_param(0.05, 1e-3, 0.5),
        wtaufall=_new_param(0.1, 1e-3, 0.5),
    )

def get_conflict_dsstimflex_weight_config():
    return _new_config(
        name="conflict_dsstimflex_weight",
        param_dict=_dsstimflex_weight_param_dict(),
        boundary_name="constant", boundary=bf.constant,
        drift_name="conflict_dsstimflex_drift", drift_fun=df.conflict_dsstimflex_drift,
        weight_name="weight_window", weight_fun=df.weight_window,
        choices=[-1, 1], n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )
```

`weight_name`/`weight_fun` are **optional** — pass them only for models with a
decision-variable weight; `_new_config` omits the keys otherwise.

Notice `toffset`/`doffset` are **not** in this `param_dict` even though they are in
the drift's `params` list. That's fine: a name in a function's `params` that is
absent from `theta` simply falls back to the function's default (here `None` →
"lasts to end of trial"). List a parameter in `param_dict` only when you want it
free/sampled; otherwise rely on the function default.

#### Picking the C simulator

`simulator=` selects the compiled integrator from `cssm`. The relevant ones for
conflict models:

| `cssm.` simulator | Use for                                                        |
|-------------------|----------------------------------------------------------------|
| `ddm_flex`        | flexible time-varying drift, no leak, no weight                |
| `ddm_flex_weight` | flexible drift **+** a multiplicative decision-variable weight |
| `ddm_flex_leak`   | flexible drift with a single leak term (`g`)                   |
| `ddm_flex_leak2`  | flexible drift with separate target/distractor leak (`gt`, `gd`) |

A model has a weight **iff** it uses `ddm_flex_weight` *and* sets
`weight_name`/`weight_fun`. Boundary choice is independent of this.

#### Angle (or other boundary) variants

Add a sibling builder that reuses the same `param_dict`, appends the boundary's
parameter, and swaps the boundary function. The codebase does exactly this:

```python
def get_conflict_dsstimflex_weight_angle_config():
    return _new_config(
        name="conflict_dsstimflex_weight_angle",
        param_dict=_dsstimflex_weight_param_dict() | dict(theta=_new_param(0.0, 0.0, 1.3)),
        boundary_name="angle", boundary=bf.angle,
        drift_name="conflict_dsstimflex_drift", drift_fun=df.conflict_dsstimflex_drift,
        weight_name="weight_window", weight_fun=df.weight_window,
        choices=[-1, 1], n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )
```

### Layer 4 — register the model (`__init__.py`)

Two edits:

1. Add the builder to the `from .conflict import (...)` block.
2. Add a `"<name>": get_<name>_config()` entry to the dict returned by
   `get_model_config()`.

The module-level validation block at the bottom of `__init__.py` runs
`get_model_config()` on import, so a misnamed parameter or a builder that raises
will fail loudly at import time with the offending model name.

## Parameter-naming rules

`validation.py` checks every parameter name against
`^[a-zA-Z][a-zA-Z0-9]{0,30}$`:

- letters/digits only, **no underscores or hyphens**, must start with a letter,
  ≤ 31 chars (hence `vtaurise`, `wtaufall`, `dfixedp` rather than
  `v_tau_rise` etc.);
- names must be unique within a model's `param_dict`.

Duplicates or invalid names raise at import.

## Checklist

- [ ] Drift (and weight) function in `drift_functions.py`; handles `t is None`;
      `DriftFunction` alias line added for new top-level functions.
- [ ] `drift_config` (and `weight_config`) entry in `base.py` whose `params` list
      matches the function kwargs you want sourced from `theta`.
- [ ] `get_<name>_config()` builder in `conflict.py`: full `param_dict`
      (core + drift + weight + boundary), correct `drift_name`/`drift_fun`,
      optional `weight_name`/`weight_fun`, correct `cssm` simulator.
- [ ] Angle/other-boundary sibling if needed (reuse the param_dict, add the
      boundary param, swap `boundary_name`/`boundary`).
- [ ] Import + `get_model_config()` dict entry in `__init__.py`.
- [ ] Names pass the alphanumeric validation rule.

## Verify (cheap, no training)

```python
# 1. Registration + validation pass, params as expected:
from ssms.config._modelconfig import get_model_config
cfg = get_model_config()["conflict_dsstimflex_weight"]
print(cfg["params"])          # full ordered param list
print(cfg["param_bounds"])    # [[lowers...], [uppers...]]

# 2. Tiny simulation through the public API (single condition, few samples):
from ssms.basic_simulators.simulator import simulator
out = simulator(model="conflict_dsstimflex_weight",
                theta={k: v for k, v in zip(cfg["params"], cfg["default_params"])},
                n_samples=50)
print(out["rts"].shape, out["choices"].shape)
```

If `get_model_config()` imports without raising and the tiny simulation returns
`rts`/`choices` with no missing-parameter error, the four layers are in
agreement.
