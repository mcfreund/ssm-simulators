"""Configuration for conflict models with dynamical drift.

The stimflex family shares three ingredients, combined in each ``get_*_config`` below:

* a **drift** base -- ``conflict_stimflex_drift`` (static boxcar), ``conflict_dsstimflex_drift``
  (exponential cue-locked dynamics), or ``conflict_dsstimflexlin_drift`` (linear cue-locked
  dynamics); ``conflict_stimflex_dual_drift`` is the two-column variant for dual-particle sims.
* a **gate** on the decision variable -- ``logit_gate`` (discrete onset selection) or
  ``hazard_gate`` (continuous detection hazard). Onset-time-dependent (lin) variants are
  expressed pan-side as per-trial effective gate coefficients, so need no separate gate here.
* a **simulator** -- ``ddm_flex_weight`` (single particle + gate) or ``ddm_flex_weight_dualleak``
  (two leaky accumulators + gate).

Param dicts are assembled from the small ``_*_params()`` helpers so a drift/gate base is
specified once. ``conflict_ds``/``conflict_ds_angle`` are unrelated (no gate; ``ddm_flex``).
"""

import cssm
from ssms.basic_simulators import boundary_functions as bf, drift_functions as df
from ssms.config._modelconfig.utils import _new_config, _new_param


def get_conflict_ds_config():
    return _new_config(
        name="conflict_ds",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_ds_drift",
        drift_fun=df.conflict_ds_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


def get_conflict_ds_angle_config():
    return _new_config(
        name="conflict_ds_angle",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            tinit=_new_param(2.0, 0.0, 5.0),
            dinit=_new_param(2.0, 0.0, 5.0),
            tslope=_new_param(2.0, 0.01, 5.0),
            dslope=_new_param(2.0, 0.01, 5.0),
            tfixedp=_new_param(3.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="conflict_ds_drift",
        drift_fun=df.conflict_ds_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex,
    )


# ---------------------------------------------------------------------------
# Shared param-dict building blocks for the stimflex family
# ---------------------------------------------------------------------------

def _core_params():
    """Boundary separation, starting point, non-decision time."""
    return dict(
        a=_new_param(2.0, 0.3, 3.0),
        z=_new_param(0.5, 0.1, 0.9),
        t=_new_param(1.0, 1e-3, 2.0),
    )


def _stimflex_drift_params():
    """Static (boxcar) drift rates + stimulus onset/offset/coherence + low-pass taus."""
    return dict(
        vt=_new_param(2.0, 0.0, 5.0),
        vd=_new_param(2.0, 0.0, 5.0),
        tcoh=_new_param(0.5, -1.0, 1.0),
        dcoh=_new_param(-0.5, -1.0, 1.0),
        tonset=_new_param(0.0, 0.0, 1.0),
        donset=_new_param(0.0, 0.0, 1.0),
        toffset=_new_param(0.2, 0.0, 1.0),
        doffset=_new_param(0.2, 0.0, 1.0),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
    )


def _dsstimflex_drift_params():
    """Exponential cue-locked drift dynamics (init/slope/fixed-point) + stimulus + taus."""
    return dict(
        tinit=_new_param(2.0, 0.0, 5.0),
        dinit=_new_param(2.0, 0.0, 5.0),
        tslope=_new_param(2.0, 0.01, 5.0),
        dslope=_new_param(2.0, 0.01, 5.0),
        tfixedp=_new_param(3.0, 0.0, 5.0),
        dfixedp=_new_param(0.0, 0.0, 5.0),
        tcoh=_new_param(0.5, -1.0, 1.0),
        dcoh=_new_param(-0.5, -1.0, 1.0),
        tonset=_new_param(0.0, 0.0, 1.0),
        donset=_new_param(0.0, 0.0, 1.0),
        toffset=_new_param(0.2, 0.0, 1.0),
        doffset=_new_param(0.2, 0.0, 1.0),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
    )


def _dsstimflexlin_drift_params():
    """Linear cue-locked drift dynamics in (level, tilt); ``maxstimoffset`` is supplied."""
    return dict(
        tlevel=_new_param(2.0, 0.0, 20.0),
        dlevel=_new_param(2.0, 0.0, 20.0),
        ttilt=_new_param(0.0, -1.0, 1.0),
        dtilt=_new_param(0.0, -1.0, 1.0),
        tcoh=_new_param(0.5, -1.0, 1.0),
        dcoh=_new_param(-0.5, -1.0, 1.0),
        tonset=_new_param(0.0, 0.0, 1.0),
        donset=_new_param(0.0, 0.0, 1.0),
        toffset=_new_param(0.2, 0.0, 1.0),
        doffset=_new_param(0.2, 0.0, 1.0),
        maxstimoffset=_new_param(1.0 + 16 / 60, 0.0, 20.0),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
    )


def _dsstimflexpwlin_drift_params():
    """Piecewise-linear (ramp-then-plateau) cue-locked drift; ``maxstimoffset`` is supplied.

    Evaluator params: per-input ramp start / plateau value + knot (fraction of the span). The
    cognitive (level, tilt) parameterization is applied model-side (pan).
    """
    return dict(
        tstart=_new_param(2.0, 0.0, 20.0),
        dstart=_new_param(2.0, 0.0, 20.0),
        tplateau=_new_param(2.0, 0.0, 20.0),
        dplateau=_new_param(2.0, 0.0, 20.0),
        ttau=_new_param(0.5, 1e-2, 1.0),
        dtau=_new_param(0.5, 1e-2, 1.0),
        tcoh=_new_param(0.5, -1.0, 1.0),
        dcoh=_new_param(-0.5, -1.0, 1.0),
        tonset=_new_param(0.0, 0.0, 1.0),
        donset=_new_param(0.0, 0.0, 1.0),
        toffset=_new_param(0.2, 0.0, 1.0),
        doffset=_new_param(0.2, 0.0, 1.0),
        maxstimoffset=_new_param(1.0 + 16 / 60, 0.0, 20.0),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
    )


def _logit_gate_params():
    """Discrete onset-selection gate: leak floor, rise time, target-vs-distractor logit."""
    return dict(
        wmin=_new_param(0.5, 0.0, 1.0),
        wtaurise=_new_param(0.05, 1e-3, 0.5),
        wtarget=_new_param(0.0, -10.0, 10.0),
    )


def _softmax_gate_params():
    """3-class onset selection: leak floor, rise time, target & distractor selection log-odds."""
    return dict(
        wmin=_new_param(0.5, 0.0, 1.0),
        wtaurise=_new_param(0.05, 1e-3, 0.5),
        wtarget=_new_param(0.0, -10.0, 10.0),
        wdistractor=_new_param(0.0, -10.0, 10.0),
    )


def _hazard_gate_params():
    """Continuous detection hazard: log baseline + per-stimulus log-hazard-ratios + gate shape."""
    return dict(
        wbaseline=_new_param(0.0, -10.0, 10.0),
        wtarget=_new_param(0.0, -5.0, 5.0),
        wdistractor=_new_param(0.0, -5.0, 5.0),
        wmin=_new_param(0.5, 0.0, 1.0),
        wtaurise=_new_param(0.05, 1e-3, 0.5),
    )


def _hazard2_gate_params():
    """hazard_gate + a focal post-onset energy window of length wtauonset (default 50 ms)."""
    return _hazard_gate_params() | dict(
        wtauonset=_new_param(0.05, 1e-3, 0.5),
    )


# ---------------------------------------------------------------------------
# ddm_flex_weight models (single particle + decision-variable gate)
# ---------------------------------------------------------------------------

def get_conflict_stimflex_logit_config():
    return _new_config(
        name="conflict_stimflex_logit",
        param_dict=_core_params() | _stimflex_drift_params() | _logit_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        weight_name="logit_gate",
        weight_fun=df.logit_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflex_logit_config():
    return _new_config(
        name="conflict_dsstimflex_logit",
        param_dict=_core_params() | _dsstimflex_drift_params() | _logit_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
        weight_name="logit_gate",
        weight_fun=df.logit_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflexlin_logit_config():
    return _new_config(
        name="conflict_dsstimflexlin_logit",
        param_dict=_core_params() | _dsstimflexlin_drift_params() | _logit_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflexlin_drift",
        drift_fun=df.conflict_dsstimflexlin_drift,
        weight_name="logit_gate",
        weight_fun=df.logit_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflexpwlin_logit_config():
    return _new_config(
        name="conflict_dsstimflexpwlin_logit",
        param_dict=_core_params() | _dsstimflexpwlin_drift_params() | _logit_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflexpwlin_drift",
        drift_fun=df.conflict_dsstimflexpwlin_drift,
        weight_name="logit_gate",
        weight_fun=df.logit_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_stimflex_softmax_config():
    return _new_config(
        name="conflict_stimflex_softmax",
        param_dict=_core_params() | _stimflex_drift_params() | _softmax_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        weight_name="softmax_gate",
        weight_fun=df.softmax_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_stimflex_hazard_config():
    return _new_config(
        name="conflict_stimflex_hazard",
        param_dict=_core_params() | _stimflex_drift_params() | _hazard_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        weight_name="hazard_gate",
        weight_fun=df.hazard_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_stimflex_hazard2_config():
    return _new_config(
        name="conflict_stimflex_hazard2",
        param_dict=_core_params() | _stimflex_drift_params() | _hazard2_gate_params(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_drift",
        drift_fun=df.conflict_stimflex_drift,
        weight_name="hazard2",
        weight_fun=df.hazard2,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


# ---------------------------------------------------------------------------
# ddm_flex_weight_dualleak models (two leaky accumulators + gate)
# ---------------------------------------------------------------------------

def get_conflict_stimflex_logit_dualleak_config():
    return _new_config(
        name="conflict_stimflex_logit_dualleak",
        param_dict=(
            _core_params()
            | dict(gt=_new_param(0.0, 0.0, 1.0), gd=_new_param(0.0, 0.0, 1.0))
            | _stimflex_drift_params()
            | _logit_gate_params()
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflex_dual_drift",
        drift_fun=df.conflict_stimflex_dual_drift,
        weight_name="logit_gate",
        weight_fun=df.logit_gate,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight_dualleak,
    )
