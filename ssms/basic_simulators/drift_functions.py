"""Define a collection of drift functions for the simulators in the package."""

# External
from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm
from scipy.special import expit, softmax


# TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
def constant(t: np.ndarray = np.arange(0, 20, 0.1)) -> np.ndarray:  # noqa: B008
    """Constant drift function.

    Arguments
    ---------
        t: np.ndarray, optional
            Timepoints at which to evaluate the drift. Defaults to
            np.arange(0, 20, 0.1).

    Returns
    -------
        np.ndarray: Array of drift values, same length as t
    """
    return np.zeros(t.shape[0])


def gamma_drift(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    shape: float = 2,
    scale: float = 0.01,
    c: float = 1.5,
) -> np.ndarray:
    """Drift function that follows a scaled gamma distribution.

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        shape: float
            Shape parameter of the gamma distribution
        scale: float
            Scale parameter of the gamma distribution
        c: float
            Scalar parameter that scales the peak of
            the gamma distribution.
            (Note this function follows a gamma distribution
            but does not integrate to 1)

    Return
    ------
        np.ndarray
            The gamma drift evaluated at the supplied timepoints t.
    """
    num_ = np.power(t, shape - 1) * np.exp(np.divide(-t, scale))
    div_ = (
        np.power(shape - 1, shape - 1)
        * np.power(scale, shape - 1)
        * np.exp(-(shape - 1))
    )
    return c * np.divide(num_, div_)


def ds_support_analytic(
    t: np.ndarray | None,
    init_p: float = 0,
    fix_point: float = 1,
    slope: float = 2,
) -> np.ndarray:
    """Solve DE.

    DE is of the form:
       x' = slope*(fix_point - x),
       with initial condition init_p.
       The solution takes the form:
       (init_p - fix_point) * exp(-slope * t) + fix_point

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
        init_p: float
            Initial condition of dynamical system
        fix_point: float
            Fixed point of dynamical system
        slope: float
            Coefficient in exponent of the solution.
    Return
    ------
    np.ndarray
         The gamma drift evaluated at the supplied timepoints t.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)

    return (init_p - fix_point) * np.exp(-(slope * t)) + fix_point


def conflict_ds_drift(
    t: np.ndarray | None,
    tinit: float = 0,
    dinit: float = 0,
    tslope: float = 1,
    dslope: float = 1,
    tfixedp: float = 1,
    tcoh: float = 1.5,
    dcoh: float = 1.5,
) -> np.ndarray:
    """This drift is inspired by a conflict task which
       involves a target and a distractor stimuli both presented
       simultaneously.

       Two drift timecourses are linearly combined weighted
       by the coherence in the respective target and distractor stimuli.
       Each timecourse follows a dynamical system as described
       in the ds_support_analytic() function.

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        tinit: float
            Initial condition of target drift timecourse
        dinit: float
            Initial condition of distractor drift timecourse
        tslope: float
            Slope parameter for target drift timecourse
        dslope: float
            Slope parameter for distractor drift timecourse
        tfixedp: float
            Fixed point for target drift timecourse
        tcoh: float
            Coefficient for the target drift timecourse
        dcoh: float
            Coefficient for the distractor drift timecourse
    Return
    ------
    np.ndarray
         The full drift timecourse evaluated at the supplied timepoints t.
    """  # noqa: D205, D401, D404
    if t is None:
        t = np.arange(0, 20, 0.1)
    w_t = ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope)

    w_d = ds_support_analytic(t=t, init_p=dinit, fix_point=0, slope=dslope)

    v_t = (w_t * tcoh) + (w_d * dcoh)

    return v_t  # , w_t, w_d


def attend_drift(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    ptarget: float = -0.3,
    pouter: float = -0.3,
    pinner: float = 0.3,
    r: float = 0.5,
    sda: float = 2,
) -> np.ndarray:
    """Shrink spotlight model, which involves a time varying
    function dependent on a linearly decreasing standard deviation of attention.

    Arguments
    --------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        pouter: float
            perceptual input for outer flankers
        pinner: float
            perceptual input for inner flankers
        ptarget: float
            perceptual input for target flanker
        r: float
            rate parameter for sda decrease
        sda: float
            width of attentional spotlight
    Return
    ------
    np.ndarray
        Drift evaluated at timepoints t
    """  # noqa: D205
    new_sda = np.maximum(sda - r * t, 0.001)

    a_outer = norm.sf(1.5, loc=0, scale=new_sda)
    a_inner = norm.cdf(1.5, loc=0, scale=new_sda) - norm.cdf(0.5, loc=0, scale=new_sda)
    a_target = norm.cdf(0.5, loc=0, scale=new_sda) - norm.cdf(
        -0.5, loc=0, scale=new_sda
    )

    v_t = (2 * pouter * a_outer) + (2 * pinner * a_inner) + (ptarget * a_target)

    return v_t


def attend_drift_simple(
    t: np.ndarray = np.arange(  # noqa: B008
        0, 20, 0.1
    ),  # TODO: #81 B008 Do not perform function call `np.arange` in argument defaults; instead, perform the call within the function, or read the default from a module-level singleton variable  # noqa: B008, FIX002
    ptarget: float = -0.3,
    pouter: float = -0.3,
    r: float = 0.5,
    sda: float = 2,
) -> np.ndarray:
    """Drift function for shrinking spotlight model, which involves a time varying
    function dependent on a linearly decreasing standard deviation of attention.

    Arguments
    --------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        pouter: float
            perceptual input for outer flankers
        ptarget: float
            perceptual input for target flanker
        r: float
            rate parameter for sda decrease
        sda: float
            width of attentional spotlight
    Return
    ------
    np.ndarray
        Drift evaluated at timepoints t
    """  # noqa: D205
    new_sda = np.maximum(sda - r * t, 0.001)
    a_outer = 1.0 - norm.cdf(
        0.5, loc=0, scale=new_sda
    )  # equivalent to norm.sf(0.5, loc=0, scale=new_sda)
    a_target = norm.cdf(0.5, loc=0, scale=new_sda) - 0.5

    v_t = (2 * pouter * a_outer) + (2 * ptarget * a_target)

    return v_t


def boxcar(
    t: np.ndarray,
    onset: float,
    offset: float,
    height: float = 1.0,
) -> np.ndarray:
    """Rectangular pulse: ``height`` on ``[onset, offset]``, 0 elsewhere."""
    pulse = np.zeros_like(t)
    pulse[(t >= onset) & (t <= offset)] = height
    return pulse


def piecewise_lowpass(
    t: np.ndarray,
    raw: np.ndarray,
    offset: float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """Causal first-order low-pass of ``raw`` with region-dependent time constants.

    ``tau_rise`` governs the response while the stimulus is on (``t <= offset``);
    ``tau_fall`` governs the post-offset tail (``t > offset``), where ``raw`` has
    dropped to 0 so the response relaxes exponentially from its offset value. Assumes
    a uniform time grid.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniformly spaced).
        raw: np.ndarray
            Signal to filter (same length as ``t``).
        offset: float
            Boundary between the rise region (<=) and the fall region (>).
        tau_rise, tau_fall: float
            Time constants for the two regions.
    Returns
    -------
        np.ndarray: filtered signal, same length as ``t``.
    """
    dt = t[1] - t[0] if t.size > 1 else 0.1
    on = t <= offset
    y = np.zeros_like(raw)
    a_r = 1.0 - np.exp(-dt / tau_rise)
    y[on] = lfilter([a_r], [1.0, -(1.0 - a_r)], raw[on])
    off = ~on
    if off.any():
        k = np.arange(1, off.sum() + 1)
        y[off] = y[on][-1] * np.exp(-k * dt / tau_fall)
    return y


def linear_scale(
    t: np.ndarray,
    level: float,
    tilt: float,
    maxstimoffset: float,
) -> np.ndarray:
    """Cue-relative linear drift-weight in (level, tilt).

    A line in absolute (cue-locked) time ``t`` -- so different stimulus onsets sample
    different segments of the *same* latent trajectory (cf. :func:`ds_support_analytic`,
    which is likewise cue-locked)::

        u = t / maxstimoffset            # fraction of the expressible span elapsed
        v(t) = level * (1 + tilt*(2u - 1))

    ``maxstimoffset`` is the latest a stimulus can turn off across the design (max onset
    + stimulus duration). Normalizing by it puts ``g = 2u - 1`` in [-1, 1] over the whole
    expressible range ``t in [0, maxstimoffset]``, so ``|tilt| <= 1`` keeps ``v``
    non-negative there. ``level`` is the drift at the reference midpoint
    (``t = maxstimoffset/2``) and, with onset ~ U and symmetric ``tilt``, equals the
    onset-marginal mean expressed drift. ``tilt = 0`` is static drift. Evaluated over all
    ``t`` (the line extrapolates); pair with :func:`boxcar` to gate it to the pulse.
    """
    u = t / maxstimoffset
    return level * (1.0 + tilt * (2.0 * u - 1.0))


def filtered_pulse(
    t: np.ndarray,
    coh: float,
    weight: np.ndarray | float,
    onset: float,
    offset: float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """One stimulus channel's drift: ``lowpass(boxcar(coh) * weight)``.

    The coherence boxcar on ``[onset, offset]`` gates a within-trial drift ``weight`` --
    a scalar for static drift (see :func:`stimflex_support`) or a per-timepoint array for
    dynamic drift (:func:`linear_scale`, :func:`ds_support_analytic`). The product is then
    passed through :func:`piecewise_lowpass`, smoothing the onset (``tau_rise``) and the
    post-offset tail (``tau_fall``).
    """
    raw = boxcar(t, onset, offset, coh) * weight
    return piecewise_lowpass(t, raw, offset, tau_rise, tau_fall)


def stimflex_support(
    t: np.ndarray,
    onset: float,
    offset: float,
    coh: float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """Construct a coherence timecourse with flexible onset/offset and
    exponential rise/fall (causal low-pass filter).

    Thin wrapper for :func:`filtered_pulse` with unit (static) weight: builds the boxcar
    (``coh`` on ``[onset, offset]``, else 0) and passes it through the causal IIR filter.

    Arguments
    ---------
        t: np.ndarray
            Timepoints within trial (uniformly spaced).
        onset: float
            Time at which the stimulus begins rising.
        offset: float
            Time at which the stimulus begins falling.
        coh: float
            Coherence of the stimulus when fully 'on'.
        tau_rise: float
            Time constant for the exponential rise.
        tau_fall: float
            Time constant for the exponential fall.
    Returns
    -------
        np.ndarray: Array of coherence values, same length as t.
    """
    return filtered_pulse(t, coh, 1.0, onset, offset, tau_rise, tau_fall)


def conflict_dsstimflex_drift(
    t: np.ndarray | None,
    tinit: float = 0,
    dinit: float = 0,
    tslope: float = 1,
    dslope: float = 1,
    tfixedp: float = 1,
    dfixedp: float = 0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    vtaurise: float = 0.05,
    vtaufall: float = 0.1,
    sum_drifts: bool = True
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset.

    Each input's drift is ``lowpass(boxcar(coh) * weight)`` (see :func:`filtered_pulse`),
    where the within-trial ``weight`` follows an independent saturating-exponential
    dynamic (``ds_support_analytic``). The boxcar gates the weight to the on-window; the
    causal filter (``vtaurise``/``vtaufall``) smooths onset and the post-offset tail.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        tcoh: float
            Coherence of the target stimulus when 'on'.
        dcoh: float
            Coherence of the distractor stimulus when 'on'.
        tinit: float
            Initial condition of target drift timecourse.
        dinit: float
            Initial condition of distractor drift timecourse.
        tslope: float
            Slope parameter for target drift timecourse.
        dslope: float
            Slope parameter for distractor drift timecourse.
        tfixedp: float
            Fixed point for target drift timecourse.
        dfixedp: float
            Fixed point for distractor drift timecourse.
        tonset: float
            Onset time of the target stimulus coherence.
        donset: float
            Onset time of the distractor stimulus coherence.
        vtaurise, vtaufall: float
            Time constants for the exponential rise/fall of the stimulus envelope.
        rel_first: bool
            If True, the first stimulus to appear (target or distractor)
            is treated as appearing at time 0, and the other stimulus
            is adjusted accordingly. If False, the onsets are treated
            as absolute times.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(
        t, tcoh, ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope),
        tonset, toffset, vtaurise, vtaufall)
    ddrift = filtered_pulse(
        t, dcoh, ds_support_analytic(t=t, init_p=dinit, fix_point=dfixedp, slope=dslope),
        donset, doffset, vtaurise, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def conflict_dsstimflexlin_drift(
    t: np.ndarray | None,
    tlevel: float = 1.0,
    dlevel: float = 1.0,
    ttilt: float = 0.0,
    dtilt: float = 0.0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    maxstimoffset: float = 1.0,
    vtaurise: float = 0.05,
    vtaufall: float = 0.1,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Conflict drift with *linear*, cue-locked within-trial drift dynamics (level/tilt).

    Each input's drift is ``lowpass(boxcar(coh) * linear_scale(level, tilt))`` (see
    :func:`filtered_pulse`, :func:`linear_scale`): a boxcar for the stimulus pulse gating
    a linear latent trajectory in (``level``, ``tilt``), then a causal piecewise low-pass
    filter (``vtaurise``/``vtaufall``). The line is in *absolute (cue-locked) time*, like
    the exponential ``ds_support_analytic``, so variable onsets sample different segments
    of one latent trajectory -- tilt shows up mostly as an onset-graded (across-trial)
    effect, plus a small within-window slope. ``tilt = 0`` recovers a static (boxcar)
    drift, matching :func:`stimflex_support` scaled by ``level``.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
        tlevel, dlevel: float
            Target/distractor drift level = value at the reference midpoint
            (``t = maxstimoffset/2``); the onset-marginal mean expressed drift.
        ttilt, dtilt: float
            Target/distractor cue-relative linear tilt (fractional): ``v`` runs from
            ``level*(1-tilt)`` at the cue to ``level*(1+tilt)`` at ``maxstimoffset``.
            ``|tilt| <= 1`` keeps the signal non-negative (a prior/bounds concern here).
        tcoh, dcoh: float
            Coherence of the target/distractor stimulus when 'on'.
        tonset, donset: float
            Onset time of the target/distractor stimulus.
        toffset, doffset: float
            Offset time of the target/distractor stimulus.
        maxstimoffset: float
            Latest a stimulus can turn off across the design (max onset + duration); the
            reference span normalizing the cue-locked line. Supplied by the model, not
            inferred (cf. tonset/donset).
        vtaurise, vtaufall: float
            Time constants for the low-pass rise (on-window) / fall (post-offset tail).
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(
        t, tcoh, linear_scale(t, tlevel, ttilt, maxstimoffset),
        tonset, toffset, vtaurise, vtaufall)
    ddrift = filtered_pulse(
        t, dcoh, linear_scale(t, dlevel, dtilt, maxstimoffset),
        donset, doffset, vtaurise, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def conflict_stimflex_drift(
    t: np.ndarray | None,
    vt: float = 0,
    vd: float = 0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    vtaurise: float = 0.05,
    vtaufall: float = 0.1,
    rel_first: bool = False,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Drift function for conflict task with static drift rates and stimuli with
    potentially variable onset/duration and exponential rise/fall.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
            Usually np.arange() of some sort.
        tcoh: float
            Coherence of the target stimulus when 'on'.
        dcoh: float
            Coherence of the distractor stimulus when 'on'.
        vt: float
            Static drift-rate of target stimulus, when 'on'.
        vd: float
            Static drift-rate of distractor stimulus, when 'on'.
        tonset: float
            Onset time of the target stimulus coherence.
        donset: float
            Onset time of the distractor stimulus coherence.
        toffset, doffset: float or None
            Offset time of the stimulus coherence pulse. If None, the pulse
            lasts until the end of the trial.
        vtaurise, vtaufall: float
            Time constants for the exponential rise/fall of the stimulus envelope.
        rel_first: bool
            If True, the first stimulus to appear (target or distractor)
            is treated as appearing at time 0, and the other stimulus
            is adjusted accordingly. If False, the onsets are treated
            as absolute times.
        sum_drifts: bool
            If True, the drift contributions from target and distractor
            are summed to produce a single drift timecourse. If False,
            a 2D array is returned with separate columns for target
            and distractor drift timecourses.
    Returns
    -------
        np.ndarray: Array of drift values, same length as t. If sum_drifts
            is False, the array has shape (len(t), 2)
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if rel_first:
        first = min(tonset, donset)
        tonset -= first
        donset -= first
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tcohs = stimflex_support(t, tonset, toffset, tcoh, vtaurise, vtaufall)
    dcohs = stimflex_support(t, donset, doffset, dcoh, vtaurise, vtaufall)
    if sum_drifts:
        return vt * tcohs + vd * dcohs
    else:
        return np.column_stack((vt * tcohs, vd * dcohs))


# Re-use drift fun but set different default args
conflict_stimflexrel1_drift = partial(conflict_stimflex_drift, rel_first=True)
conflict_stimflexrel1_dual_drift = partial(
    conflict_stimflex_drift, rel_first=True, sum_drifts=False
)
conflict_dsstimflex_dual_drift = partial(
    conflict_dsstimflex_drift, sum_drifts=False
)


def weight_window(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    donset: float = 0.0,
    doffset: float = 1.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtaufall: float = 0.1,
    onoff_method: str = "target",
    rng=None,  # deterministic; accepted so cssm can inject uniformly
) -> np.ndarray:
    if t is None:
        t = np.arange(0, 20, 0.1)
    if onoff_method == "target":
        onset = tonset
        offset = toffset
    elif onoff_method == "first-last":
        onset = min(tonset, donset)
        offset = max(toffset, doffset)
    return stimflex_support(t, onset, offset, 1 - wmin, wtaurise, wtaufall) + wmin


def weight_window_ds(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtaufall: float = 0.1,
    winit: float = 0,
    wslope: float = 0.2,
    rng=None,  # deterministic; accepted so cssm can inject uniformly
) -> np.ndarray:
    if t is None:
        t = np.arange(0, 20, 0.1)
    ## Stimulus-driven weight:
    stim_weight = stimflex_support(t, tonset, toffset, 1 - wmin, wtaurise, wtaufall)
    ## Cue-locked weight (scale amplitude of stim-driven):
    cue_weight = ds_support_analytic(t=t, init_p=winit, fix_point=1, slope=wslope)
    return stim_weight * cue_weight + wmin


def sample_hazard_onset(t, rate, rng):
    """Sample the gate opening time from a time-varying hazard (inf if it never opens)."""
    dt = t[1] - t[0]
    cum_hazard = np.cumsum(rate) * dt
    idx = np.searchsorted(cum_hazard, rng.exponential())
    return t[idx] if idx < t.shape[0] else np.inf


def prob_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    tcoh: float = 1.0,
    donset: float = 0.0,
    doffset: float = 1.0,
    dcoh: float = 1.0,
    vtaurise: float = 0.05,
    vtaufall: float = 0.1,
    wbaseline: float = 0,
    wtarget: float = 1,
    wdistractor: float = 1,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    if t is None:
        t = np.arange(0, 20, 0.1)
    target_energy = stimflex_support(t, tonset, toffset, np.abs(tcoh), vtaurise, vtaufall)
    distractor_energy = stimflex_support(t, donset, doffset, np.abs(dcoh), vtaurise, vtaufall)
    rate = np.exp(target_energy * wtarget + distractor_energy * wdistractor + wbaseline)
    if rng is None:
        rng = np.random.default_rng()
    return (t > sample_hazard_onset(t, rate, rng)).astype(float)


def parametric_weight(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    vtaurise: float = 0.05,
    wbaseline: float = 0,
    wtarget: float = 1,
    wdistractor: float = 1,
    rng=None,  # deterministic; accepted so cssm can inject uniformly
) -> np.ndarray:
    if t is None:
        t = np.arange(0, 20, 0.1)
    target_energy = stimflex_support(t, tonset, np.max(t), np.abs(tcoh), vtaurise, 1e12)
    distractor_energy = stimflex_support(t, donset, np.max(t), np.abs(dcoh), vtaurise, 1e12)
    return expit(target_energy * wtarget + distractor_energy * wdistractor + wbaseline)


def parametric_mixture(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtarget: float = 0,
    wdistractor: float = 0,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    if t is None:
        t = np.arange(0, 20, 0.005)
    if rng is None:
        rng = np.random.default_rng()
    onset = rng.choice([min(tonset, donset), tonset, donset], p = softmax([0, wtarget, wdistractor]))
    return stimflex_support(t, onset, t.max(), 1 - wmin, wtaurise, 1e12) + wmin


def parametric_mixture_det(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtarget: float = 0,
    wdistractor: float = 0,
    rng=None,  # accepted for a uniform cssm signature; deterministic, so unused
) -> np.ndarray:
    """Deterministic counterpart to ``parametric_mixture``.

    Instead of sampling a single onset per trial, return the softmax-weighted *average*
    of the three candidate weight windows (baseline=earlier stimulus, target onset,
    distractor onset), with mixing weights ``softmax([0, wtarget, wdistractor])``. Every
    trial with the same parameters/design yields the identical weight timecourse, which
    removes the trial-level mixture noise and sharpens the likelihood -- making
    ``wtarget``/``wdistractor`` more identifiable than the stochastic version.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    onsets = (min(tonset, donset), tonset, donset)
    p = softmax([0.0, wtarget, wdistractor])
    windows = np.stack(
        [stimflex_support(t, onset, t.max(), 1 - wmin, wtaurise, 1e12) for onset in onsets])
    # sum_k p_k * (stim_k + wmin) = (sum_k p_k stim_k) + wmin, since sum_k p_k = 1.
    return wmin + p @ windows


# Type alias for drift functions
DriftFunction = Callable[..., np.ndarray]

attend_drift: DriftFunction = attend_drift  # noqa: PLW0127
constant: DriftFunction = constant  # noqa: PLW0127
gamma_drift: DriftFunction = gamma_drift  # noqa: PLW0127
ds_support_analytic: DriftFunction = ds_support_analytic  # noqa: PLW0127
conflict_ds_drift: DriftFunction = conflict_ds_drift  # noqa: PLW0127
conflict_dsstimflex_drift: DriftFunction = conflict_dsstimflex_drift  # noqa: PLW0127
conflict_dsstimflexlin_drift: DriftFunction = conflict_dsstimflexlin_drift  # noqa: PLW0127
conflict_stimflex_drift: DriftFunction = conflict_stimflex_drift  # noqa: PLW0127
conflict_dsstimflex_dual_drift: DriftFunction = conflict_dsstimflex_dual_drift  # noqa: PLW0127
weight_window: DriftFunction = weight_window  # noqa: PLW0127
weight_window_ds: DriftFunction = weight_window_ds  # noqa: PLW0127
prob_gate: DriftFunction = prob_gate  # noqa: PLW0127
parametric_weight: DriftFunction = parametric_weight  # noqa: PLW0127
parametric_mixture: DriftFunction = parametric_mixture  # noqa: PLW0127
parametric_mixture_det: DriftFunction = parametric_mixture_det  # noqa: PLW0127