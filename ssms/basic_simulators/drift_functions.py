"""Define a collection of drift functions for the simulators in the package."""

# External
from collections.abc import Callable

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


def causal_lowpass(t: np.ndarray, raw: np.ndarray, tau: float) -> np.ndarray:
    """Causal first-order (EMA) low-pass of ``raw`` with time constant ``tau``.

    ``y[n] = (1 - a)*y[n-1] + a*x[n]`` with ``a = 1 - exp(-dt/tau)``, on a uniform time
    grid. ``tau <= 0`` is the identity (no smoothing, sharp onset) -- the ``tau = 0``
    limit, handled explicitly to avoid a divide-by-zero.
    """
    if tau <= 0:
        return np.array(raw, dtype=float)
    dt = t[1] - t[0] if t.size > 1 else 0.1
    a = 1.0 - np.exp(-dt / tau)
    return lfilter([a], [1.0, -(1.0 - a)], raw)


def piecewise_lowpass(
    t: np.ndarray,
    raw: np.ndarray,
    offset: float,
    tau_rise: float,
    tau_fall: float,
) -> np.ndarray:
    """Causal low-pass of ``raw`` with region-dependent time constants.

    The on-window (``t <= offset``) is a causal EMA with time constant ``tau_rise`` (see
    :func:`causal_lowpass`; ``tau_rise = 0`` gives a sharp, unsmoothed onset). The
    post-offset tail (``t > offset``), where ``raw`` has dropped to 0, relaxes exponentially
    from the offset value with time constant ``tau_fall``. Assumes a uniform time grid.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniformly spaced).
        raw: np.ndarray
            Signal to filter (same length as ``t``).
        offset: float
            Boundary between the rise region (<=) and the fall region (>).
        tau_rise, tau_fall: float
            Time constants for the two regions (``tau_rise = 0`` -> no onset smoothing).
    Returns
    -------
        np.ndarray: filtered signal, same length as ``t``.
    """
    on = t <= offset
    y = np.zeros_like(raw, dtype=float)
    y[on] = causal_lowpass(t, raw[on], tau_rise)
    off = ~on
    if off.any():
        dt = t[1] - t[0] if t.size > 1 else 0.1
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


def pwlin_scale(
    t: np.ndarray,
    vstart: float,
    vplateau: float,
    tau: float,
    maxstimoffset: float,
) -> np.ndarray:
    """Cue-relative ramp-then-plateau drift-weight (piecewise-linear analogue of ``linear_scale``).

    A two-segment line in absolute (cue-locked) time ``t``: it ramps linearly from ``vstart`` at
    the cue to ``vplateau`` at the knot, then holds ``vplateau`` (a coarse saturating
    growth/decay). With ``u = t / maxstimoffset`` the fraction of the expressible span elapsed and
    the knot ``tau`` in that same normalized time::

        u    = t / maxstimoffset                          # fraction of the expressible span elapsed
        v(t) = vstart + (vplateau - vstart) * min(u, tau) / tau

    Like :func:`linear_scale`, the line is cue-locked, so different stimulus onsets sample
    different segments of the *same* latent trajectory (steep during the ramp ``u < tau``, flat
    after). ``vstart == vplateau`` is static drift. This is a design-agnostic *evaluator*: the
    (level, tilt) -> (vstart, vplateau) reparameterization that makes the onset-marginal mean
    drift equal ``level`` for any tilt/knot depends on the onset design and is done model-side
    (pan), not here. Pair with :func:`boxcar`/:func:`filtered_pulse` to gate it to the pulse.
    """
    u = t / maxstimoffset
    return vstart + (vplateau - vstart) * np.minimum(u, tau) / tau


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
    a scalar for static drift or a per-timepoint array for dynamic drift 
    (:func:`linear_scale`, :func:`ds_support_analytic`). The product is then
    passed through :func:`piecewise_lowpass`, smoothing the onset (``tau_rise``) and the
    post-offset tail (``tau_fall``).
    """
    raw = boxcar(t, onset, offset, coh) * weight
    return piecewise_lowpass(t, raw, offset, tau_rise, tau_fall)


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
    vtaufall: float = 0.1,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Drift function for conflict task with static drift rates and stimuli with
    potentially variable onset/duration and an exponential post-offset tail.

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
        vtaufall: float
            Time constant for the exponential post-offset tail of the stimulus envelope.
            (Onset is sharp: the low-pass rise time is fixed to 0.)
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
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(t, tcoh, vt, tonset, toffset, tau_rise=0.0, tau_fall=vtaufall)
    ddrift = filtered_pulse(t, dcoh, vd, donset, doffset, tau_rise=0.0, tau_fall=vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def conflict_stimflex_dual_drift(t: np.ndarray | None, **kwargs) -> np.ndarray:
    """Two-column (target, distractor) drift for dual-particle simulators.

    Thin wrapper around :func:`conflict_stimflex_drift` with ``sum_drifts=False``, so it
    returns the target and distractor drift timecourses column-stacked (shape ``(len(t), 2)``)
    rather than summed. Consumed by :func:`cssm.ddm_flex_weight_dualleak`, whose two leaky
    accumulators need the drifts kept separate. Defined as a named function (not a
    ``functools.partial``) so the simulator's ``drift_fun.__name__`` metadata read works.
    """
    return conflict_stimflex_drift(t, sum_drifts=False, **kwargs)


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
    vtaufall: float = 0.1,
    sum_drifts: bool = True
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset.

    Each input's drift is ``lowpass(boxcar(coh) * weight)`` (see :func:`filtered_pulse`),
    where the within-trial ``weight`` follows an independent saturating-exponential
    dynamic (``ds_support_analytic``). The boxcar gates the weight to the on-window; the
    onset is sharp (rise time fixed to 0) and the post-offset tail relaxes with ``vtaufall``.

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
        vtaufall: float
            Time constant for the exponential post-offset tail (onset is sharp: rise = 0).
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(
        t, tcoh, ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope),
        tonset, toffset, tau_rise=0.0, tau_fall=vtaufall)
    ddrift = filtered_pulse(
        t, dcoh, ds_support_analytic(t=t, init_p=dinit, fix_point=dfixedp, slope=dslope),
        donset, doffset, tau_rise=0.0, tau_fall=vtaufall)
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
    vtaufall: float = 0.1,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Conflict drift with *linear*, cue-locked within-trial drift dynamics (level/tilt).

    Each input's drift is ``lowpass(boxcar(coh) * linear_scale(level, tilt))`` (see
    :func:`filtered_pulse`, :func:`linear_scale`): a boxcar for the stimulus pulse gating
    a linear latent trajectory in (``level``, ``tilt``), then a causal low-pass with a
    sharp onset (rise = 0) and a ``vtaufall`` post-offset tail. The line is in *absolute (cue-locked) time*, like
    the exponential ``ds_support_analytic``, so variable onsets sample different segments
    of one latent trajectory -- tilt shows up mostly as an onset-graded (across-trial)
    effect, plus a small within-window slope. ``tilt = 0`` recovers a static (boxcar)
    drift, matching :func:`conflict_stimflex_drift` with the drift rate set to ``level``.

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
        vtaufall: float
            Time constant for the post-offset tail (onset is sharp: rise = 0).
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(
        t, tcoh, linear_scale(t, tlevel, ttilt, maxstimoffset),
        tonset, toffset, tau_rise=0.0, tau_fall=vtaufall)
    ddrift = filtered_pulse(
        t, dcoh, linear_scale(t, dlevel, dtilt, maxstimoffset),
        donset, doffset, tau_rise=0.0, tau_fall=vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def conflict_dsstimflexpwlin_drift(
    t: np.ndarray | None,
    tstart: float = 1.0,
    dstart: float = 1.0,
    tplateau: float = 1.0,
    dplateau: float = 1.0,
    ttau: float = 0.5,
    dtau: float = 0.5,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    maxstimoffset: float = 1.0,
    vtaufall: float = 0.1,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Conflict drift with *piecewise-linear* (ramp-then-plateau) cue-locked within-trial dynamics.

    The within-trial analogue of :func:`conflict_dsstimflexlin_drift`, with the linear line
    replaced by a ramp-then-plateau (a coarse saturating trajectory). Each input's drift is
    ``lowpass(boxcar(coh) * pwlin_scale(vstart, vplateau, tau))`` (see :func:`pwlin_scale`,
    :func:`filtered_pulse`): a boxcar for the stimulus pulse gating the ramp-plateau latent
    trajectory in absolute (cue-locked) time, then a causal low-pass with a sharp onset (rise = 0)
    and a ``vtaufall`` post-offset tail. The line is cue-locked, so variable onsets sample
    different segments of one saturating trajectory (steep during the ramp, flat after the knot).
    ``vstart == vplateau`` recovers a static (boxcar) drift, matching
    :func:`conflict_stimflex_drift` with the drift rate set to that common value.

    The endpoints ``*start``/``*plateau`` and knot ``*tau`` are the raw *evaluator* parameters;
    the cognitive (level, tilt) parameterization -- onset-marginal mean drift = ``level`` for any
    tilt/knot -- is applied model-side (pan ``make_theta``), since the level/tilt -> endpoint map
    depends on the onset design. See ``src/ssm/scratch/drift-dynamics-parameterization.md``.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift. Usually np.arange() of some sort.
        tstart, dstart: float
            Target/distractor drift at the cue (``u = 0``, the ramp start).
        tplateau, dplateau: float
            Target/distractor drift on the plateau (``u >= tau``, after saturation).
        ttau, dtau: float
            Target/distractor knot: fraction of the expressible span
            (``u = t / maxstimoffset``) at which the ramp switches to the plateau.
        tcoh, dcoh: float
            Coherence of the target/distractor stimulus when 'on'.
        tonset, donset: float
            Onset time of the target/distractor stimulus.
        toffset, doffset: float
            Offset time of the target/distractor stimulus.
        maxstimoffset: float
            Latest a stimulus can turn off across the design (max onset + duration); the
            reference span normalizing the cue-locked line. Supplied by the model, not inferred.
        vtaufall: float
            Time constant for the post-offset tail (onset is sharp: rise = 0).
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if toffset is None:
        toffset = np.max(t)
    if doffset is None:
        doffset = np.max(t)
    tdrift = filtered_pulse(
        t, tcoh, pwlin_scale(t, tstart, tplateau, ttau, maxstimoffset),
        tonset, toffset, tau_rise=0.0, tau_fall=vtaufall)
    ddrift = filtered_pulse(
        t, dcoh, pwlin_scale(t, dstart, dplateau, dtau, maxstimoffset),
        donset, doffset, tau_rise=0.0, tau_fall=vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def conflict_bufferlin_drift(
    t: np.ndarray | None,
    tlevel: float = 1.0,
    dlevel: float = 1.0,
    ttilt: float = 0.0,
    dtilt: float = 0.0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    maxstimoffset: float = 1.0,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Conflict drift from ideal (noiseless, non-decaying) sensory buffers with linear dynamics.

    Each input's buffer is a step at its onset, signed by coherence, and holds for the rest of the
    trial (no stimulus offset, no post-offset tail). Its drift is that step times a cue-locked line
    in (``level``, ``tilt``) clipped at zero::

        u    = t / maxstimoffset
        v(t) = coh * 1[t >= onset] * max(0, level * (1 + tilt*(2u - 1)))

    ``level`` is the drift at ``t = maxstimoffset/2``. The line is evaluated past ``maxstimoffset``
    for as long as the buffer holds, so the clip keeps ``tilt < 0`` from reversing the drift sign
    late in the trial. ``tilt = 0`` is a static step of height ``coh * level``.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
        tlevel, dlevel: float
            Target/distractor drift at the reference midpoint ``t = maxstimoffset/2``.
        ttilt, dtilt: float
            Target/distractor fractional tilt of the cue-locked line.
        tcoh, dcoh: float
            Signed coherence of the target/distractor stimulus.
        tonset, donset: float
            Onset time of the target/distractor stimulus (the buffer fills here).
        maxstimoffset: float
            Reference span normalizing the cue-locked line. Supplied by the model, not inferred.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    tdrift = (t >= tonset) * tcoh * np.maximum(linear_scale(t, tlevel, ttilt, maxstimoffset), 0.0)
    ddrift = (t >= donset) * dcoh * np.maximum(linear_scale(t, dlevel, dtilt, maxstimoffset), 0.0)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def soft_gate(
    t: np.ndarray,
    onset: float,
    tau_rise: float,
    wmin: float
) -> np.ndarray:
    """Gate that latches open at ``onset`` and stays open.

    Before ``onset`` the gate sits at its leaky-closed floor ``wmin``; at ``onset`` it
    transitions toward 1 with rise time ``tau_rise`` (a causal low-pass of a step to
    ``1 - wmin``) and never falls back -- so this is a pure :func:`causal_lowpass` of the
    step, with no fall region (no ``tau_fall``). If ``onset`` is beyond the time grid
    (e.g. ``np.inf`` from :func:`sample_hazard_onset`, i.e. the gate never opens) the
    boxcar is all-zero and the gate stays at ``wmin`` throughout. Shared by
    :func:`hazard_gate` and :func:`logit_gate` as the open-transition.
    """
    raw = boxcar(t, onset, t.max(), 1 - wmin)
    return causal_lowpass(t, raw, tau_rise) + wmin


def sample_hazard_onset(t, rate, rng):
    """Sample the gate opening time from a time-varying hazard (inf if it never opens)."""
    dt = t[1] - t[0]
    cum_hazard = np.cumsum(rate) * dt
    idx = np.searchsorted(cum_hazard, rng.exponential())
    return t[idx] if idx < t.shape[0] else np.inf


def hazard_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    tcoh: float = 1.0,
    donset: float = 0.0,
    doffset: float = 1.0,
    dcoh: float = 1.0,
    wbaseline: float = 0,
    wtarget: float = 1,
    wdistractor: float = 1,
    wmin: float = 0,
    wtaurise: float = 0.05,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Continuous-time detection gate: opening time drawn from a stimulus-driven hazard.

    A *memoryless* gate. The instantaneous opening rate is a log-linear (proportional-
    hazards) function of the momentary stimulus energy -- the unfiltered coherence boxcars::

        rate(t) = exp( |tcoh|*wtarget + |dcoh|*wdistractor + wbaseline )

    ``wbaseline`` is the log baseline rate (spontaneous opening with no stimulus); ``wtarget``
    /``wdistractor`` are log-hazard-ratios per unit coherence (sign-free -- presence may raise
    or lower the rate). A single opening time is drawn via :func:`sample_hazard_onset`
    (dt-consistent survival sampling; may be ``inf`` = never opens), then the gate latches
    open from ``wmin`` toward 1 with rise time ``wtaurise`` (see :func:`soft_gate`).

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, toffset, donset, doffset: float
            Onset/offset of the target/distractor coherence boxcars.
        tcoh, dcoh: float
            Target/distractor coherence (only ``|coh|`` enters the evidence).
        wbaseline: float
            Log baseline opening rate (coherence-independent).
        wtarget, wdistractor: float
            Log-hazard-ratio contributed by target/distractor presence, per unit coherence.
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
        rng: np.random.Generator
            Injected by cssm; unseeded fallback only when called standalone.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    tenergy = boxcar(t, tonset, toffset, np.abs(tcoh))
    denergy = boxcar(t, donset, doffset, np.abs(dcoh))
    rate = np.exp(tenergy * wtarget + denergy * wdistractor + wbaseline)
    if rng is None:
        rng = np.random.default_rng()
    t_open = sample_hazard_onset(t, rate, rng)
    return soft_gate(t, t_open, wtaurise, wmin)


def hazard2(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    tcoh: float = 1.0,
    donset: float = 0.0,
    doffset: float = 1.0,
    dcoh: float = 1.0,
    wbaseline: float = 0,
    wtarget: float = 1,
    wdistractor: float = 1,
    wtauonset: float = 0.05,
    wmin: float = 0,
    wtaurise: float = 0.05,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Focal-onset variant of :func:`hazard_gate`.

    Same log-linear (proportional-hazards) opening process, but each stimulus's coherence
    energy is a **short box of length ``wtauonset`` right after its onset** rather than the
    full ``[onset, offset]`` boxcar::

        rate(t) = exp( |tcoh|*wtarget * 1{tonset <= t <= tonset + wtauonset}
                     + |dcoh|*wdistractor * 1{donset <= t <= donset + wtauonset}
                     + wbaseline )

    So a stimulus can only drive gate opening in the first ``wtauonset`` seconds after it
    appears (a phasic onset response); past that window it no longer raises the rate. This
    concentrates stimulus-driven opening near onset, so onset-locked opening must be carried by
    a high ``wtarget`` / short latency rather than by diffuse within-window capture -- pair with
    a low ``wbaseline`` so late opening is not simply reabsorbed as spontaneous. ``toffset`` /
    ``doffset`` are accepted for signature parity with :func:`hazard_gate` but unused here (the
    energy window is set by ``wtauonset``, not the stimulus offset). Reduces to
    :func:`hazard_gate` in the limit ``wtauonset -> stimulus duration``.

    Arguments
    ---------
        t, tonset, tcoh, donset, dcoh, wbaseline, wtarget, wdistractor, wmin, wtaurise, rng:
            As in :func:`hazard_gate`.
        toffset, doffset: float
            Accepted for parity with :func:`hazard_gate`; not used (window set by ``wtauonset``).
        wtauonset: float
            Length (s) of the post-onset window over which each stimulus raises the rate.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    tenergy = boxcar(t, tonset, tonset + wtauonset, np.abs(tcoh))
    denergy = boxcar(t, donset, donset + wtauonset, np.abs(dcoh))
    rate = np.exp(tenergy * wtarget + denergy * wdistractor + wbaseline)
    if rng is None:
        rng = np.random.default_rng()
    t_open = sample_hazard_onset(t, rate, rng)
    return soft_gate(t, t_open, wtaurise, wmin)


def logit_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtarget: float = 0,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Discrete onset-selection gate: opens at the target or distractor onset.

    A one-shot attentional-capture gate: the gate always opens, at one of the two stimulus
    onsets, chosen by a 2-class softmax (target vs distractor, distractor = reference)::

        p(target) = softmax([wtarget, 0]) -> expit(wtarget)

    ``wtarget`` is the log-odds of selecting the target onset over the distractor onset. The
    chosen onset is then passed to :func:`soft_gate` (latch open from ``wmin`` toward 1 with
    rise time ``wtaurise``). With a large ``wtarget`` the target onset is always selected,
    recovering a deterministic window gate (and for simultaneous onsets ``tonset == donset``
    the choice is a no-op). Needs asynchronous onsets to be identifiable.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, donset: float
            Candidate opening times (target/distractor onsets).
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
        wtarget: float
            Log-odds of selecting the target onset (vs the distractor reference).
        rng: np.random.Generator
            Injected by cssm; unseeded fallback only when called standalone.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    if rng is None:
        rng = np.random.default_rng()
    onset = rng.choice([tonset, donset], p = softmax([wtarget, 0]))
    return soft_gate(t, onset, wtaurise, wmin)


def softmax_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    wtarget: float = 0,
    wdistractor: float = 0,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Discrete onset-selection gate over 3 candidate opening times (generalizes :func:`logit_gate`).

    A one-shot attentional-capture gate that always opens, at one of *three* candidate times,
    chosen by a 3-class softmax::

        first = min(tonset, donset)                                  # earliest onset (reference)
        p([first, target, distractor]) = softmax([0, wtarget, wdistractor])

    The reference class ``first`` opens as soon as *any* stimulus appears (identity-agnostic
    capture); ``wtarget`` / ``wdistractor`` are the log-odds of instead holding out to open
    specifically at the target / distractor onset, relative to opening at the first onset. The
    chosen onset is passed to :func:`soft_gate` (latch open from ``wmin`` toward 1 with rise time
    ``wtaurise``). Reduces to opening at the shared onset when ``tonset == donset`` (all three
    candidates coincide, a no-op); needs asynchronous onsets to be identifiable.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, donset: float
            Target/distractor onsets. ``min(tonset, donset)`` is the reference candidate.
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
        wtarget: float
            Log-odds of opening at the target onset (vs the first-onset reference).
        wdistractor: float
            Log-odds of opening at the distractor onset (vs the first-onset reference).
        rng: np.random.Generator
            Injected by cssm; unseeded fallback only when called standalone.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    if rng is None:
        rng = np.random.default_rng()
    onset = rng.choice([min(tonset, donset), tonset, donset],
                       p = softmax([0, wtarget, wdistractor]))
    return soft_gate(t, onset, wtaurise, wmin)


def delay_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    wdelay: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    rng=None,  # injected by cssm; unused (the gate is deterministic)
) -> np.ndarray:
    """Deterministic latching gate that opens at the first onset, but no earlier than ``wdelay``::

        open = max(wdelay, min(tonset, donset))

    passed to :func:`soft_gate` (latch open from ``wmin`` toward 1 with rise time ``wtaurise``).
    ``wdelay`` is a cue-locked lower bound on when integration can begin; ``wdelay = 0`` opens at
    the first onset.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, donset: float
            Target/distractor onsets.
        wdelay: float
            Earliest opening time (s, cue-locked).
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    return soft_gate(t, max(wdelay, min(tonset, donset)), wtaurise, wmin)


def wait_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    ptonset: float = 0.5,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Gate that opens at the target onset with probability ``ptonset``, else at the second onset::

        open = tonset               with probability ptonset
             = max(tonset, donset)  otherwise

    The opening never precedes the target. On distractor-first trials both candidates are the
    target onset, so ``ptonset`` acts only on target-first trials. One uniform is drawn per trial
    regardless of onset order. The chosen onset is passed to :func:`soft_gate`.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, donset: float
            Target/distractor onsets.
        ptonset: float
            Probability of opening at the target onset rather than the second onset.
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
        rng: np.random.Generator
            Injected by cssm; unseeded fallback only when called standalone.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    if rng is None:
        rng = np.random.default_rng()
    onset = tonset if rng.random() < ptonset else max(tonset, donset)
    return soft_gate(t, onset, wtaurise, wmin)


def _buffer_strength(tau, kernelh, kernelrise, kerneladapt):
    """Buffer strength ``tau`` s after a stimulus's onset: the step response of a unit with
    adaptation, 1 − (1 + h)·e^(−τ/rise) + h·e^(−τ/adapt) where ``kernelrise > 0`` (rises from 0,
    overshoots when h > 0, settles at 1), and the ideal buffer's 1 where ``kernelrise <= 0``."""
    x = np.maximum(tau, 0.0)
    rise = np.where(kernelrise > 0, kernelrise, 1.0)
    m = 1 - (1 + kernelh) * np.exp(-x / rise) + kernelh * np.exp(-x / kerneladapt)
    return np.where(kernelrise > 0, m, 1.0)


def _onset_pulse(tau, kernelrise, kerneladapt):
    """Onset transient ``tau`` s after a stimulus's onset, e^(−τ/adapt) − e^(−τ/rise) scaled to a
    peak of 1 where ``kernelrise > 0`` (needs adapt > rise), and 0 where ``kernelrise <= 0``."""
    x = np.maximum(tau, 0.0)
    on = kernelrise > 0
    rise = np.where(on, kernelrise, 0.5)
    adapt = np.where(on, kerneladapt, 1.0)
    q = rise / adapt
    peak = q ** (q / (1 - q)) - q ** (1 / (1 - q))
    return np.where(on, (np.exp(-x / adapt) - np.exp(-x / rise)) / peak, 0.0)


def _selectivity(vt, vd):
    """ρ = Δ/Σ from the role rates, 0 where Σ = 0."""
    sig = vt + vd
    return np.where(sig > 0, (vt - vd) / np.where(sig > 0, sig, 1.0), 0.0)


def _two_ramps(t, start, mid, end, knotmid, knot, hold):
    """A mode's line from ``start`` at the cue to ``mid`` at ``knotmid``, then to ``end`` at
    ``knot``, held at ``end`` after ``knot`` where ``hold > 0``; needs 0 < knotmid < knot."""
    t = np.where(hold > 0, np.minimum(t, knot), t)
    early = mid - (mid - start) * (1.0 - t / np.where(knotmid > 0, knotmid, 1.0))
    late = end - (end - mid) * (knot - t) / np.where(knot > knotmid, knot - knotmid, 1.0)
    return np.where(t < knotmid, early, late)


def _buffermode_rates(t, sumstart, sumend, diffstart, diffend, knot, hold,
                      sumboost, diffboost, boostfwhm, tonset, donset, rhopre=1.0, tresolve=0.0,
                      swappre=0.0, knotmid=0.0, summid=0.0, diffmid=0.0):
    """Target/distractor rates ``(v_T, v_D)`` from the sum/difference modes; broadcasts over params."""
    knot_safe = np.where(knot > 0, knot, 1.0)
    r = np.where(knot > 0, t / knot_safe, 1.0)
    r = np.where(hold > 0, np.minimum(r, 1.0), r)
    sig = sumend - (sumend - sumstart) * (1.0 - r)
    dif = diffend - (diffend - diffstart) * (1.0 - r)

    # A second knot at knotmid > 0 replaces the single line with two
    if np.any(knotmid > 0):
        sig = np.where(knotmid > 0, _two_ramps(t, sumstart, summid, sumend, knotmid, knot, hold),
                       sig)
        dif = np.where(knotmid > 0, _two_ramps(t, diffstart, diffmid, diffend, knotmid, knot, hold),
                       dif)
    gap = tonset - donset
    g = np.exp(-4.0 * np.log(2.0) * gap**2 / boostfwhm**2) * (t >= np.maximum(tonset, donset))
    sig = np.maximum(sig + sumboost * g, 0.0)
    dif = np.clip(dif + diffboost * g, -sig, sig)
    dif = np.where(t < tresolve, np.where(swappre > 0, -dif, rhopre * sig), dif)
    return (sig + dif) / 2, (sig - dif) / 2


def conflict_buffermode_drift(
    t: np.ndarray | None,
    sumstart: float = 1.0,
    sumend: float = 1.0,
    diffstart: float = 0.0,
    diffend: float = 0.0,
    knot: float = 0.0,
    hold: float = 1.0,
    sumboost: float = 0.0,
    diffboost: float = 0.0,
    boostfwhm: float = 16 / 60,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    swap: float = 0.0,
    kernelh: float = 0.0,
    kernelrise: float = 0.0,
    kerneladapt: float = 1.0,
    sumpulse: float = 0.0,
    rhopre: float = 1.0,
    tresolve: float = 0.0,
    pulsesel: float = 0.0,
    swappre: float = 0.0,
    tpulse: float = -1.0,
    dpulse: float = -1.0,
    tpulsegain: float = 1.0,
    dpulsegain: float = 1.0,
    toffset: float = 100.0,
    doffset: float = 100.0,
    bufferleak: float = 0.0,
    tleakgain: float = 1.0,
    dleakgain: float = 1.0,
    knotmid: float = 0.0,
    summid: float = 0.0,
    diffmid: float = 0.0,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Conflict drift from sensory buffers, written in sum (Σ = v_T + v_D) and difference
    (Δ = v_T − v_D) modes::

        r(t)   = t / knot, capped at 1 if hold      (knot <= 0: r = 1, static)
        Σ(t)   = sumend  - (sumend  - sumstart)  * (1 - r(t))   + sumboost  * g
        Δ(t)   = diffend - (diffend - diffstart) * (1 - r(t))   + diffboost * g
                 (0 < knotmid < knot: each line runs start -> mid at knotmid -> end at knot)
        g      = exp(-4 ln2 * (tonset - donset)^2 / boostfwhm^2) * 1[t >= max(tonset, donset)]
        Σ >= 0, Δ clipped to [-Σ, Σ]
        Δ(t)   = rhopre * Σ(t)   for t < tresolve   (-Δ(t) if swappre > 0)
        v_T    = (Σ + Δ)/2,  v_D = (Σ - Δ)/2
        m(τ)   = 1 - (1 + kernelh) e^(-τ/kernelrise) + kernelh e^(-τ/kerneladapt)   (kernelrise > 0)
               = 1                                                                (kernelrise <= 0)
        p(τ)   = (e^(-τ/kerneladapt) - e^(-τ/kernelrise)) / its peak             (kernelrise > 0)
               = 0                                                                (kernelrise <= 0)
        drift  = tcoh * (1[t >= tonset] m(t - tonset) L_T v_T + 1[t >= t_p] g_T sumpulse (1 + pulsesel ρ) p(t - t_p) / 2)
               + dcoh * (1[t >= donset] m(t - donset) L_D v_D + 1[t >= d_p] g_D sumpulse (1 - pulsesel ρ) p(t - d_p) / 2)
        L_T    = exp(-tleakgain * bufferleak * max(0, t - toffset)),  L_D likewise with dleakgain
                 from doffset
        t_p    = tpulse (tonset if tpulse < 0),  d_p = dpulse (donset if dpulse < 0)
        g_T    = tpulsegain,  g_D = dpulsegain
        ρ      = Δ / Σ   (0 where Σ = 0)

    ``swap > 0`` exchanges the two stimuli's roles: the distractor is read at v_T and the target at v_D;
    each stimulus keeps its own coherence, onset, offset, transient start, transient gain and leak
    gain.
    ``L`` is each buffer's leak after its stimulus's offset (``bufferleak`` 0: none).
    ``r`` is cue-locked: ``hold = 1`` ramps to the end values at ``knot`` and holds them,
    ``hold = 0`` continues the same line past ``knot``. A second knot ``knotmid`` (0 < ``knotmid`` <
    ``knot``; ``<= 0``: none) splits the ramp: Σ and Δ run linearly from their start values at the
    cue to ``summid`` and ``diffmid`` at ``knotmid``, then to their end values at ``knot``, held or
    continued after ``knot`` as above. The clip keeps ``v_T, v_D >= 0``.
    ``g`` is the simultaneity boost, switched on at the second onset. ``m`` is each buffer's
    strength after its stimulus's onset; the defaults give ideal buffers. ``p`` is an onset
    transient of peak ``sumpulse`` in the sum mode and of peak ``pulsesel`` · ρ(t) · ``sumpulse``
    in the difference mode (``pulsesel`` 0: the same for both roles). Before ``tresolve`` the
    selectivity Δ/Σ is ``rhopre``, or with ``swappre > 0`` the roles are exchanged (Δ → −Δ, as
    ``swap``). Each transient starts at its stimulus's onset unless ``tpulse``/``dpulse`` place it.
    The defaults give no transient and no pre-resolution period.

    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the drift.
        sumstart, sumend: float
            Sum mode at the cue and at ``knot``.
        diffstart, diffend: float
            Difference mode at the cue and at ``knot``.
        knot: float
            Cue-locked time (s) of the end values; ``<= 0`` is static at the end values.
        hold: float
            ``> 0`` holds the end values after ``knot``; ``0`` extrapolates.
        sumboost, diffboost: float
            Boost amplitude added to the sum/difference mode at zero onset gap.
        boostfwhm: float
            Full width at half maximum (s) of the boost over the onset gap.
        tcoh, dcoh: float
            Signed coherence of the target/distractor stimulus.
        tonset, donset: float
            Onset time of the target/distractor stimulus (the buffer fills here).
        swap: float
            ``> 0`` reads the distractor at the target's rate and the target at the distractor's.
        kernelh: float
            Amplitude of the buffer's adapting overshoot (0: a plain rise).
        kernelrise, kerneladapt: float
            Rise and adaptation time constants (s) of the buffer; ``kernelrise <= 0`` is the
            ideal buffer. They also shape the onset transient.
        sumpulse: float
            Peak of the onset transient in the sum mode (0: none).
        rhopre: float
            Selectivity Δ/Σ before ``tresolve``.
        tresolve: float
            Cue-locked time (s) from which Δ follows its modes; ``<= 0`` is from the cue.
        pulsesel: float
            Selectivity of the onset transient as a fraction of the current ρ = Δ/Σ.
        swappre: float
            ``> 0`` negates Δ before ``tresolve`` in place of ``rhopre``.
        tpulse, dpulse: float
            Start time (s) of the target's/distractor's onset transient; ``< 0`` is its stimulus's
            onset.
        tpulsegain, dpulsegain: float
            Gain on the target's/distractor's onset transient (1: the shared peak).
        toffset, doffset: float
            Offset time of the target/distractor stimulus (the buffer starts to leak here).
        bufferleak: float
            Leak rate (1/s) of each buffer after its stimulus's offset.
        tleakgain, dleakgain: float
            Gain on the target's/distractor's leak rate (1: the shared rate).
        knotmid: float
            Cue-locked time (s) of the second knot, between the cue and ``knot``; ``<= 0`` is none.
        summid, diffmid: float
            Sum and difference mode at ``knotmid``.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    if swap > 0:
        tcoh, dcoh, tonset, donset, tpulse, dpulse = dcoh, tcoh, donset, tonset, dpulse, tpulse
        tpulsegain, dpulsegain, toffset, doffset = dpulsegain, tpulsegain, doffset, toffset
        tleakgain, dleakgain = dleakgain, tleakgain
    tp = np.where(np.asarray(tpulse) < 0, tonset, tpulse)
    dp = np.where(np.asarray(dpulse) < 0, donset, dpulse)
    vt, vd = _buffermode_rates(t, sumstart, sumend, diffstart, diffend, knot, hold,
                               sumboost, diffboost, boostfwhm, tonset, donset, rhopre, tresolve,
                               swappre, knotmid, summid, diffmid)
    tdrift = (t >= tonset) * tcoh * vt
    ddrift = (t >= donset) * dcoh * vd
    if kernelrise > 0:
        tdrift = tdrift * _buffer_strength(t - tonset, kernelh, kernelrise, kerneladapt)
        ddrift = ddrift * _buffer_strength(t - donset, kernelh, kernelrise, kerneladapt)
    if bufferleak > 0:
        tdrift = tdrift * np.exp(-tleakgain * bufferleak * np.maximum(t - toffset, 0.0))
        ddrift = ddrift * np.exp(-dleakgain * bufferleak * np.maximum(t - doffset, 0.0))
    if sumpulse != 0:
        sel = pulsesel * _selectivity(vt, vd)
        tdrift = tdrift + (t >= tp) * tcoh * tpulsegain * sumpulse * (1 + sel) / 2 * _onset_pulse(
            t - tp, kernelrise, kerneladapt)
        ddrift = ddrift + (t >= dp) * dcoh * dpulsegain * sumpulse * (1 - sel) / 2 * _onset_pulse(
            t - dp, kernelrise, kerneladapt)
    if sum_drifts:
        return tdrift + ddrift
    else:
        return np.column_stack((tdrift, ddrift))


def floor_gate(
    t: np.ndarray | None,
    tonset: float = 0.0,
    donset: float = 0.0,
    fhard: float = 0.0,
    ftarget: float = 0.0,
    wfirst: float = 0.0,
    wsecond: float = 0.0,
    wmin: float = 0.0,
    wtaurise: float = 0.05,
    swap: float = 0.0,
    wdist: float = 0.0,
    rng=None,  # injected by cssm; unseeded fallback only when called standalone
) -> np.ndarray:
    """Gate that opens at the first onset, the second onset, the distractor or the target, behind a
    cue-locked floor::

        open = max(fhard, min(tonset, donset))          with probability wfirst
             = max(fhard, max(tonset, donset))          with probability wsecond
             = max(fhard + ftarget, donset)             with probability wdist
             = max(fhard + ftarget, tonset)             otherwise

    ``fhard`` is a floor on every trial; ``ftarget`` is an additional floor on trials opened by a
    stimulus's identity (a per-trial value, drawn by the caller when it varies). ``swap > 0`` exchanges the onsets, as
    the drift does, so the target-opened gate waits for the distractor. One uniform is drawn per
    trial. The chosen onset is passed to :func:`soft_gate`.

    Arguments
    ---------
        t: np.ndarray
            Timepoints (uniform grid).
        tonset, donset: float
            Target/distractor onsets.
        fhard: float
            Cue-locked floor (s) on every opening.
        ftarget: float
            Additional floor (s) when the gate opens at the target.
        wfirst, wsecond: float
            Probability of opening at the first / second onset regardless of identity.
        wdist: float
            Probability of opening at the distractor behind both floors.
        wmin: float
            Leaky-closed floor before opening.
        wtaurise: float
            Rise time of the open transition.
        swap: float
            ``> 0`` exchanges the target's and the distractor's onsets.
        rng: np.random.Generator
            Injected by cssm; unseeded fallback only when called standalone.
    """
    if t is None:
        t = np.arange(0, 20, 0.005)
    if rng is None:
        rng = np.random.default_rng()
    if swap > 0:
        tonset, donset = donset, tonset
    u = rng.random()
    if u < wfirst:
        onset = max(fhard, min(tonset, donset))
    elif u < wfirst + wsecond:
        onset = max(fhard, max(tonset, donset))
    elif u < wfirst + wsecond + wdist:
        onset = max(fhard + ftarget, donset)
    else:
        onset = max(fhard + ftarget, tonset)
    return soft_gate(t, onset, wtaurise, wmin)


# ===========================================================================
# Batched (vectorized-over-trials) siblings
#
# The cssm flex_weight sampler can precompute drift/weight for ALL trials in one
# call instead of a per-trial Python loop. A scalar function opts in by carrying
# a ``.batched`` attribute pointing at its batched sibling, which accepts
# ``(n_trials,)`` parameter arrays and returns ``(n_trials, len(t))`` (drift
# ``sum_drifts=False`` -> ``(n_trials, len(t), 2)``). The sampler falls back to
# the scalar per-trial loop for any function without a ``.batched`` sibling, so
# the two can be mixed freely and A/B-tested against each other.
#
# The deterministic construction is reproduced in CLOSED FORM (no ``lfilter``):
#   * every stimflex drift uses ``tau_rise=0`` -> the on-window is unfiltered and
#     the post-offset tail is an analytic exponential (matches ``piecewise_lowpass``
#     exactly);
#   * ``soft_gate`` is a causal EMA of a step, i.e. an exponential rise
#     ``H*(1-exp(-m*dt/tau))`` (matches ``causal_lowpass`` of the step exactly).
# So the only departure from the scalar functions is the RNG draw *order* (one
# batched draw vs n_trials scalar draws) -> distributionally equal, not bitwise.
# ===========================================================================


def _b1d(x: np.ndarray | float, n: int) -> np.ndarray:
    """Coerce a scalar/1d param to a length-``n`` float64 array (broadcasting length-1)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.shape[0] == 1 and n > 1:
        x = np.broadcast_to(x, (n,))
    return x


def _filtered_pulse_b(
    t: np.ndarray,
    coh: np.ndarray,
    weight: np.ndarray,
    onset: np.ndarray,
    offset: np.ndarray,
    tau_fall: np.ndarray,
) -> np.ndarray:
    """Batched ``filtered_pulse`` for the ``tau_rise=0`` case (all stimflex drifts).

    ``coh``/``onset``/``offset``/``tau_fall`` are ``(n,)``; ``weight`` is ``(n, 1)``
    (static drift) or ``(n, len(t))`` (dynamic drift). Returns ``(n, len(t))``.
    Reproduces ``piecewise_lowpass(boxcar(coh)*weight, offset, tau_rise=0, tau_fall)``:
    the on-window (``t <= offset``) is unfiltered, the tail (``t > offset``) decays
    analytically from the last on-window value.
    """
    N = t.shape[0]
    dt = t[1] - t[0] if N > 1 else 0.1
    tt = t[None, :]
    box = (tt >= onset[:, None]) & (tt <= offset[:, None])          # coherence support
    raw = box * (coh[:, None] * weight)                            # (n, N)
    on_region = tt <= offset[:, None]                              # piecewise_lowpass `on`
    n_on = on_region.sum(axis=1)                                   # (n,)
    idx_last = np.clip(n_on - 1, 0, N - 1)
    val_last = raw[np.arange(raw.shape[0]), idx_last]              # value at offset index
    j = np.arange(N)[None, :]
    k = np.maximum(j - n_on[:, None] + 1, 0)                       # 1,2,.. into the tail
    decay = val_last[:, None] * np.exp(-k * dt / tau_fall[:, None])
    return np.where(on_region, raw, decay)


def _soft_gate_b(
    t: np.ndarray,
    onset: np.ndarray,
    tau_rise: np.ndarray,
    wmin: np.ndarray,
) -> np.ndarray:
    """Batched ``soft_gate``: analytic exponential rise from ``wmin`` toward 1 at ``onset``.

    ``onset``/``tau_rise``/``wmin`` are ``(n,)`` (``onset=inf`` -> gate never opens,
    stays at ``wmin``). Returns ``(n, len(t))``. The causal EMA of a step of height
    ``1-wmin`` is ``(1-wmin)*(1-exp(-m*dt/tau_rise))``; ``tau_rise=0`` is the sharp step.
    """
    N = t.shape[0]
    dt = t[1] - t[0] if N > 1 else 0.1
    onset = np.asarray(onset, dtype=float)
    tau_rise = np.asarray(tau_rise, dtype=float)
    onset_idx = np.searchsorted(t, onset)                          # inf -> N (never opens)
    j = np.arange(N)[None, :]
    open_mask = j >= onset_idx[:, None]
    m = np.maximum(j - onset_idx[:, None] + 1, 0)                  # steps since opening
    tau0 = (tau_rise <= 0)[:, None]
    tau_safe = np.where(tau_rise > 0, tau_rise, 1.0)[:, None]
    frac = np.where(open_mask, np.where(tau0, 1.0, 1.0 - np.exp(-m * dt / tau_safe)), 0.0)
    wmin_c = np.asarray(wmin, dtype=float)[:, None]
    return wmin_c + (1.0 - wmin_c) * frac


def conflict_stimflex_drift_b(
    t: np.ndarray,
    vt=0, vd=0, tcoh=1.0, dcoh=1.0, tonset=0, donset=0,
    toffset=None, doffset=None, vtaufall=0.1, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_stimflex_drift` (see that function). Params ``(n,)`` -> ``(n, len(t))``."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (vt, vd, tcoh, dcoh, tonset, donset, vtaufall))
    if toffset is None:
        toffset = t.max()
    if doffset is None:
        doffset = t.max()
    vt, vd, tcoh, dcoh, tonset, donset, toffset, doffset, vtaufall = (
        _b1d(x, n) for x in (vt, vd, tcoh, dcoh, tonset, donset, toffset, doffset, vtaufall)
    )
    tdrift = _filtered_pulse_b(t, tcoh, vt[:, None], tonset, toffset, vtaufall)
    ddrift = _filtered_pulse_b(t, dcoh, vd[:, None], donset, doffset, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def logit_gate_b(
    t: np.ndarray,
    tonset=0.0, donset=0.0, wmin=0.0, wtaurise=0.05, wtarget=0, rng=None,
) -> np.ndarray:
    """Batched :func:`logit_gate` (see that function). Params ``(n,)`` -> ``(n, len(t))``.

    Draws one target-vs-distractor selection per trial from a single ``rng.random(n)``
    call (``p(target) = expit(wtarget) = softmax([wtarget, 0])[0]``).
    """
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = max(np.size(x) for x in (tonset, donset, wmin, wtaurise, wtarget))
    tonset, donset, wmin, wtaurise, wtarget = (
        _b1d(x, n) for x in (tonset, donset, wmin, wtaurise, wtarget)
    )
    onset = np.where(rng.random(n) < expit(wtarget), tonset, donset)
    return _soft_gate_b(t, onset, wtaurise, wmin)


def _boxcar_b(t: np.ndarray, onset: np.ndarray, offset: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Batched ``boxcar``: ``height`` on ``[onset, offset]``. Params ``(n,)`` -> ``(n, len(t))``."""
    tt = t[None, :]
    return ((tt >= onset[:, None]) & (tt <= offset[:, None])) * height[:, None]


def _ds_support_b(t: np.ndarray, init_p: np.ndarray, fix_point: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Batched ``ds_support_analytic``: ``(init_p-fix_point)*exp(-slope*t)+fix_point``."""
    return (init_p - fix_point)[:, None] * np.exp(-(slope[:, None] * t[None, :])) + fix_point[:, None]


def _linear_scale_b(t: np.ndarray, level: np.ndarray, tilt: np.ndarray, maxstimoffset: np.ndarray) -> np.ndarray:
    """Batched ``linear_scale``: cue-locked line ``level*(1 + tilt*(2u-1))``, ``u=t/maxstimoffset``."""
    u = t[None, :] / maxstimoffset[:, None]
    return level[:, None] * (1.0 + tilt[:, None] * (2.0 * u - 1.0))


def _pwlin_scale_b(t: np.ndarray, vstart: np.ndarray, vplateau: np.ndarray, tau: np.ndarray,
                   maxstimoffset: np.ndarray) -> np.ndarray:
    """Batched ``pwlin_scale``: ramp ``vstart``->``vplateau`` until knot ``tau``, then plateau."""
    u = t[None, :] / maxstimoffset[:, None]
    tau_c = tau[:, None]
    return vstart[:, None] + (vplateau - vstart)[:, None] * np.minimum(u, tau_c) / tau_c


def _sample_hazard_onset_b(t: np.ndarray, rate: np.ndarray, rng) -> np.ndarray:
    """Batched ``sample_hazard_onset``: inverse-CDF of the cumulative hazard, per row.

    ``rate`` is ``(n, len(t))``; returns ``(n,)`` opening times (``inf`` if never opens).
    ``cum`` is monotone per row (cumsum of a nonnegative rate) so the per-row
    ``searchsorted(cum, E)`` is the vectorized ``(cum < E).sum(axis=1)``.
    """
    N = t.shape[0]
    dt = t[1] - t[0] if N > 1 else 0.1
    cum = np.cumsum(rate, axis=1) * dt
    e = rng.exponential(size=rate.shape[0])
    idx = (cum < e[:, None]).sum(axis=1)
    return np.where(idx < N, t[np.clip(idx, 0, N - 1)], np.inf)


def conflict_stimflex_dual_drift_b(t: np.ndarray, **kwargs) -> np.ndarray:
    """Batched :func:`conflict_stimflex_dual_drift` -> ``(n, len(t), 2)`` (target, distractor)."""
    return conflict_stimflex_drift_b(t, sum_drifts=False, **kwargs)


def conflict_dsstimflex_drift_b(
    t: np.ndarray,
    tinit=0, dinit=0, tslope=1, dslope=1, tfixedp=1, dfixedp=0,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, toffset=None, doffset=None,
    vtaufall=0.1, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_dsstimflex_drift` (exponential cue-locked weight)."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (tinit, dinit, tslope, dslope, tfixedp, dfixedp,
                                 tcoh, dcoh, tonset, donset, vtaufall))
    if toffset is None:
        toffset = t.max()
    if doffset is None:
        doffset = t.max()
    (tinit, dinit, tslope, dslope, tfixedp, dfixedp, tcoh, dcoh,
     tonset, donset, toffset, doffset, vtaufall) = (
        _b1d(x, n) for x in (tinit, dinit, tslope, dslope, tfixedp, dfixedp, tcoh, dcoh,
                             tonset, donset, toffset, doffset, vtaufall)
    )
    tdrift = _filtered_pulse_b(t, tcoh, _ds_support_b(t, tinit, tfixedp, tslope), tonset, toffset, vtaufall)
    ddrift = _filtered_pulse_b(t, dcoh, _ds_support_b(t, dinit, dfixedp, dslope), donset, doffset, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def conflict_dsstimflexlin_drift_b(
    t: np.ndarray,
    tlevel=1.0, dlevel=1.0, ttilt=0.0, dtilt=0.0,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, toffset=None, doffset=None,
    maxstimoffset=1.0, vtaufall=0.1, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_dsstimflexlin_drift` (linear cue-locked weight)."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (tlevel, dlevel, ttilt, dtilt, tcoh, dcoh,
                                 tonset, donset, maxstimoffset, vtaufall))
    if toffset is None:
        toffset = t.max()
    if doffset is None:
        doffset = t.max()
    (tlevel, dlevel, ttilt, dtilt, tcoh, dcoh, tonset, donset, toffset, doffset,
     maxstimoffset, vtaufall) = (
        _b1d(x, n) for x in (tlevel, dlevel, ttilt, dtilt, tcoh, dcoh, tonset, donset,
                             toffset, doffset, maxstimoffset, vtaufall)
    )
    tdrift = _filtered_pulse_b(t, tcoh, _linear_scale_b(t, tlevel, ttilt, maxstimoffset), tonset, toffset, vtaufall)
    ddrift = _filtered_pulse_b(t, dcoh, _linear_scale_b(t, dlevel, dtilt, maxstimoffset), donset, doffset, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def conflict_dsstimflexpwlin_drift_b(
    t: np.ndarray,
    tstart=1.0, dstart=1.0, tplateau=1.0, dplateau=1.0, ttau=0.5, dtau=0.5,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, toffset=None, doffset=None,
    maxstimoffset=1.0, vtaufall=0.1, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_dsstimflexpwlin_drift` (ramp-then-plateau cue-locked weight)."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (tstart, dstart, tplateau, dplateau, ttau, dtau,
                                 tcoh, dcoh, tonset, donset, maxstimoffset, vtaufall))
    if toffset is None:
        toffset = t.max()
    if doffset is None:
        doffset = t.max()
    (tstart, dstart, tplateau, dplateau, ttau, dtau, tcoh, dcoh, tonset, donset,
     toffset, doffset, maxstimoffset, vtaufall) = (
        _b1d(x, n) for x in (tstart, dstart, tplateau, dplateau, ttau, dtau, tcoh, dcoh,
                             tonset, donset, toffset, doffset, maxstimoffset, vtaufall)
    )
    tdrift = _filtered_pulse_b(t, tcoh, _pwlin_scale_b(t, tstart, tplateau, ttau, maxstimoffset), tonset, toffset, vtaufall)
    ddrift = _filtered_pulse_b(t, dcoh, _pwlin_scale_b(t, dstart, dplateau, dtau, maxstimoffset), donset, doffset, vtaufall)
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def softmax_gate_b(
    t: np.ndarray,
    tonset=0.0, donset=0.0, wmin=0.0, wtaurise=0.05, wtarget=0, wdistractor=0, rng=None,
) -> np.ndarray:
    """Batched :func:`softmax_gate`: 3-class onset selection, one categorical draw per trial."""
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = max(np.size(x) for x in (tonset, donset, wmin, wtaurise, wtarget, wdistractor))
    tonset, donset, wmin, wtaurise, wtarget, wdistractor = (
        _b1d(x, n) for x in (tonset, donset, wmin, wtaurise, wtarget, wdistractor)
    )
    cands = np.stack([np.minimum(tonset, donset), tonset, donset], axis=1)     # (n, 3)
    p = softmax(np.stack([np.zeros(n), wtarget, wdistractor], axis=1), axis=1)  # (n, 3)
    idx = np.clip((rng.random(n)[:, None] >= np.cumsum(p, axis=1)).sum(axis=1), 0, 2)
    onset = cands[np.arange(n), idx]
    return _soft_gate_b(t, onset, wtaurise, wmin)


def conflict_bufferlin_drift_b(
    t: np.ndarray,
    tlevel=1.0, dlevel=1.0, ttilt=0.0, dtilt=0.0,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, maxstimoffset=1.0, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_bufferlin_drift` (ideal buffers, clipped linear cue-locked weight)."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (tlevel, dlevel, ttilt, dtilt, tcoh, dcoh,
                                 tonset, donset, maxstimoffset))
    tlevel, dlevel, ttilt, dtilt, tcoh, dcoh, tonset, donset, maxstimoffset = (
        _b1d(x, n) for x in (tlevel, dlevel, ttilt, dtilt, tcoh, dcoh, tonset, donset,
                             maxstimoffset)
    )
    tt = t[None, :]
    tdrift = ((tt >= tonset[:, None]) * tcoh[:, None]
              * np.maximum(_linear_scale_b(t, tlevel, ttilt, maxstimoffset), 0.0))
    ddrift = ((tt >= donset[:, None]) * dcoh[:, None]
              * np.maximum(_linear_scale_b(t, dlevel, dtilt, maxstimoffset), 0.0))
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def delay_gate_b(
    t: np.ndarray,
    tonset=0.0, donset=0.0, wdelay=0.0, wmin=0.0, wtaurise=0.05, rng=None,
) -> np.ndarray:
    """Batched :func:`delay_gate`: opens at ``max(wdelay, min(tonset, donset))``."""
    t = np.asarray(t, dtype=float)
    n = max(np.size(x) for x in (tonset, donset, wdelay, wmin, wtaurise))
    tonset, donset, wdelay, wmin, wtaurise = (
        _b1d(x, n) for x in (tonset, donset, wdelay, wmin, wtaurise)
    )
    return _soft_gate_b(t, np.maximum(wdelay, np.minimum(tonset, donset)), wtaurise, wmin)


def wait_gate_b(
    t: np.ndarray,
    tonset=0.0, donset=0.0, ptonset=0.5, wmin=0.0, wtaurise=0.05, rng=None,
) -> np.ndarray:
    """Batched :func:`wait_gate`: one uniform per trial picks target onset vs second onset."""
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = max(np.size(x) for x in (tonset, donset, ptonset, wmin, wtaurise))
    tonset, donset, ptonset, wmin, wtaurise = (
        _b1d(x, n) for x in (tonset, donset, ptonset, wmin, wtaurise)
    )
    onset = np.where(rng.random(n) < ptonset, tonset, np.maximum(tonset, donset))
    return _soft_gate_b(t, onset, wtaurise, wmin)


def conflict_buffermode_drift_b(
    t: np.ndarray,
    sumstart=1.0, sumend=1.0, diffstart=0.0, diffend=0.0, knot=0.0, hold=1.0,
    sumboost=0.0, diffboost=0.0, boostfwhm=16 / 60,
    tcoh=1.0, dcoh=1.0, tonset=0, donset=0, swap=0.0,
    kernelh=0.0, kernelrise=0.0, kerneladapt=1.0, sumpulse=0.0, rhopre=1.0, tresolve=0.0,
    pulsesel=0.0, swappre=0.0, tpulse=-1.0, dpulse=-1.0, tpulsegain=1.0, dpulsegain=1.0,
    toffset=100.0, doffset=100.0, bufferleak=0.0, tleakgain=1.0, dleakgain=1.0,
    knotmid=0.0, summid=0.0, diffmid=0.0, sum_drifts=True,
) -> np.ndarray:
    """Batched :func:`conflict_buffermode_drift` (sensory buffers, sum/difference modes)."""
    t = np.asarray(t, dtype=float)
    args = (sumstart, sumend, diffstart, diffend, knot, hold, sumboost, diffboost, boostfwhm,
            tcoh, dcoh, tonset, donset, swap, kernelh, kernelrise, kerneladapt,
            sumpulse, rhopre, tresolve, pulsesel, swappre, tpulse, dpulse, tpulsegain, dpulsegain,
            toffset, doffset, bufferleak, tleakgain, dleakgain, knotmid, summid, diffmid)
    n = max(np.size(x) for x in args)
    (sumstart, sumend, diffstart, diffend, knot, hold, sumboost, diffboost, boostfwhm,
     tcoh, dcoh, tonset, donset, swap, kernelh, kernelrise, kerneladapt,
     sumpulse, rhopre, tresolve, pulsesel, swappre, tpulse, dpulse, tpulsegain,
     dpulsegain, toffset, doffset, bufferleak, tleakgain, dleakgain, knotmid, summid, diffmid) = (
        _b1d(x, n)[:, None] for x in args)
    sw = swap > 0
    tcoh, dcoh = np.where(sw, dcoh, tcoh), np.where(sw, tcoh, dcoh)
    tonset, donset = np.where(sw, donset, tonset), np.where(sw, tonset, donset)
    toffset, doffset = np.where(sw, doffset, toffset), np.where(sw, toffset, doffset)
    tpulse, dpulse = np.where(sw, dpulse, tpulse), np.where(sw, tpulse, dpulse)
    tpulsegain, dpulsegain = np.where(sw, dpulsegain, tpulsegain), np.where(sw, tpulsegain, dpulsegain)
    tleakgain, dleakgain = np.where(sw, dleakgain, tleakgain), np.where(sw, tleakgain, dleakgain)
    tp, dp = np.where(tpulse < 0, tonset, tpulse), np.where(dpulse < 0, donset, dpulse)
    tt = t[None, :]
    static = not (np.any(knot > 0) or np.any(knotmid > 0) or np.any(sumboost != 0)
                  or np.any(diffboost != 0) or np.any(tresolve > 0))
    vt, vd = _buffermode_rates(tt[:, :1] if static else tt, sumstart, sumend, diffstart, diffend,
                               knot, hold, sumboost, diffboost, boostfwhm, tonset, donset,
                               rhopre, tresolve, swappre, knotmid, summid, diffmid)
    tdrift = (tt >= tonset) * tcoh * vt
    ddrift = (tt >= donset) * dcoh * vd
    if np.any(kernelrise > 0):
        tdrift = tdrift * _buffer_strength(tt - tonset, kernelh, kernelrise, kerneladapt)
        ddrift = ddrift * _buffer_strength(tt - donset, kernelh, kernelrise, kerneladapt)
    if np.any(bufferleak > 0):
        tdrift = tdrift * np.exp(-tleakgain * bufferleak * np.maximum(tt - toffset, 0.0))
        ddrift = ddrift * np.exp(-dleakgain * bufferleak * np.maximum(tt - doffset, 0.0))
    if np.any(sumpulse != 0):
        sel = pulsesel * _selectivity(vt, vd)
        tdrift = tdrift + (tt >= tp) * tcoh * tpulsegain * sumpulse * (1 + sel) / 2 * _onset_pulse(
            tt - tp, kernelrise, kerneladapt)
        ddrift = ddrift + (tt >= dp) * dcoh * dpulsegain * sumpulse * (1 - sel) / 2 * _onset_pulse(
            tt - dp, kernelrise, kerneladapt)
    if sum_drifts:
        return tdrift + ddrift
    return np.stack((tdrift, ddrift), axis=-1)


def floor_gate_b(
    t: np.ndarray,
    tonset=0.0, donset=0.0, fhard=0.0, ftarget=0.0, wfirst=0.0, wsecond=0.0,
    wmin=0.0, wtaurise=0.05, swap=0.0, wdist=0.0, rng=None,
) -> np.ndarray:
    """Batched :func:`floor_gate`: one uniform per trial picks first / second / distractor / target
    opening."""
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    args = (tonset, donset, fhard, ftarget, wfirst, wsecond, wmin, wtaurise, swap, wdist)
    n = max(np.size(x) for x in args)
    tonset, donset, fhard, ftarget, wfirst, wsecond, wmin, wtaurise, swap, wdist = (
        _b1d(x, n) for x in args
    )
    sw = swap > 0
    tonset, donset = np.where(sw, donset, tonset), np.where(sw, tonset, donset)
    u = rng.random(n)
    onset = np.where(
        u < wfirst, np.maximum(fhard, np.minimum(tonset, donset)),
        np.where(u < wfirst + wsecond, np.maximum(fhard, np.maximum(tonset, donset)),
                 np.where(u < wfirst + wsecond + wdist, np.maximum(fhard + ftarget, donset),
                          np.maximum(fhard + ftarget, tonset))))
    return _soft_gate_b(t, onset, wtaurise, wmin)


def hazard_gate_b(
    t: np.ndarray,
    tonset=0.0, toffset=1.0, tcoh=1.0, donset=0.0, doffset=1.0, dcoh=1.0,
    wbaseline=0, wtarget=1, wdistractor=1, wmin=0, wtaurise=0.05, rng=None,
) -> np.ndarray:
    """Batched :func:`hazard_gate`: log-linear hazard over the full coherence boxcars."""
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = max(np.size(x) for x in (tonset, toffset, tcoh, donset, doffset, dcoh,
                                 wbaseline, wtarget, wdistractor, wmin, wtaurise))
    (tonset, toffset, tcoh, donset, doffset, dcoh, wbaseline, wtarget, wdistractor,
     wmin, wtaurise) = (
        _b1d(x, n) for x in (tonset, toffset, tcoh, donset, doffset, dcoh,
                             wbaseline, wtarget, wdistractor, wmin, wtaurise)
    )
    tenergy = _boxcar_b(t, tonset, toffset, np.abs(tcoh))
    denergy = _boxcar_b(t, donset, doffset, np.abs(dcoh))
    rate = np.exp(tenergy * wtarget[:, None] + denergy * wdistractor[:, None] + wbaseline[:, None])
    onset = _sample_hazard_onset_b(t, rate, rng)
    return _soft_gate_b(t, onset, wtaurise, wmin)


def hazard2_b(
    t: np.ndarray,
    tonset=0.0, toffset=1.0, tcoh=1.0, donset=0.0, doffset=1.0, dcoh=1.0,
    wbaseline=0, wtarget=1, wdistractor=1, wtauonset=0.05, wmin=0, wtaurise=0.05, rng=None,
) -> np.ndarray:
    """Batched :func:`hazard2`: log-linear hazard over short post-onset energy windows."""
    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    n = max(np.size(x) for x in (tonset, tcoh, donset, dcoh, wbaseline, wtarget,
                                 wdistractor, wtauonset, wmin, wtaurise))
    (tonset, tcoh, donset, dcoh, wbaseline, wtarget, wdistractor,
     wtauonset, wmin, wtaurise) = (
        _b1d(x, n) for x in (tonset, tcoh, donset, dcoh, wbaseline, wtarget,
                             wdistractor, wtauonset, wmin, wtaurise)
    )
    tenergy = _boxcar_b(t, tonset, tonset + wtauonset, np.abs(tcoh))
    denergy = _boxcar_b(t, donset, donset + wtauonset, np.abs(dcoh))
    rate = np.exp(tenergy * wtarget[:, None] + denergy * wdistractor[:, None] + wbaseline[:, None])
    onset = _sample_hazard_onset_b(t, rate, rng)
    return _soft_gate_b(t, onset, wtaurise, wmin)


# Advertise the batched siblings (consumed by the cssm flex_weight precompute).
conflict_stimflex_drift.batched = conflict_stimflex_drift_b
conflict_stimflex_dual_drift.batched = conflict_stimflex_dual_drift_b
conflict_dsstimflex_drift.batched = conflict_dsstimflex_drift_b
conflict_dsstimflexlin_drift.batched = conflict_dsstimflexlin_drift_b
conflict_dsstimflexpwlin_drift.batched = conflict_dsstimflexpwlin_drift_b
logit_gate.batched = logit_gate_b
softmax_gate.batched = softmax_gate_b
conflict_bufferlin_drift.batched = conflict_bufferlin_drift_b
delay_gate.batched = delay_gate_b
wait_gate.batched = wait_gate_b
conflict_buffermode_drift.batched = conflict_buffermode_drift_b
floor_gate.batched = floor_gate_b
hazard_gate.batched = hazard_gate_b
hazard2.batched = hazard2_b


# Type alias for drift functions
DriftFunction = Callable[..., np.ndarray]

attend_drift: DriftFunction = attend_drift  # noqa: PLW0127
constant: DriftFunction = constant  # noqa: PLW0127
gamma_drift: DriftFunction = gamma_drift  # noqa: PLW0127
ds_support_analytic: DriftFunction = ds_support_analytic  # noqa: PLW0127
conflict_ds_drift: DriftFunction = conflict_ds_drift  # noqa: PLW0127
conflict_stimflex_drift: DriftFunction = conflict_stimflex_drift  # noqa: PLW0127
conflict_stimflex_dual_drift: DriftFunction = conflict_stimflex_dual_drift  # noqa: PLW0127
conflict_dsstimflex_drift: DriftFunction = conflict_dsstimflex_drift  # noqa: PLW0127
conflict_dsstimflexlin_drift: DriftFunction = conflict_dsstimflexlin_drift  # noqa: PLW0127
conflict_dsstimflexpwlin_drift: DriftFunction = conflict_dsstimflexpwlin_drift  # noqa: PLW0127
hazard_gate: DriftFunction = hazard_gate  # noqa: PLW0127
hazard2: DriftFunction = hazard2  # noqa: PLW0127
logit_gate: DriftFunction = logit_gate  # noqa: PLW0127
softmax_gate: DriftFunction = softmax_gate  # noqa: PLW0127