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