"""Define a collection of drift functions for the simulators in the package."""

# External
from collections.abc import Callable
from functools import partial

import numpy as np
from scipy.stats import norm


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


def stimflex_support(
    t: np.ndarray, onset: float, offset: float, coh: float
) -> np.ndarray:
    """
    Construct a rectangular coherence timecourse, with discrete and
    potentially variable onsets and offsets of stimulus evidence.

    Arguments
    ---------
        t: np.ndarray
            Timepoints within trial.
        onset: float
            Onset time of the coherence pulse.
        offset: float
            Offset time of the coherence pulse.
        coh: float
            Coherence of the stimulus when 'on'.
    Returns
    -------
        np.ndarray: Array of coherence values, same length as t.
    """
    cohs = np.zeros_like(t)
    cohs[(t >= onset) & (t <= offset)] = coh
    return cohs


def conflict_dsstimflex_drift(
    t: np.ndarray | None,
    tinit: float = 0,
    dinit: float = 0,
    tslope: float = 1,
    dslope: float = 1,
    tfixedp: float = 1,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    rel_first: bool = True,
    sum_drifts: bool = True
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset.

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
        tonset: float
            Onset time of the target stimulus coherence.
        donset: float
            Onset time of the distractor stimulus coherence.
        rel_first: bool
            If True, the first stimulus to appear (target or distractor)
            is treated as appearing at time 0, and the other stimulus
            is adjusted accordingly. If False, the onsets are treated
            as absolute times.
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
    tcohs = stimflex_support(t, tonset, toffset, tcoh)
    dcohs = stimflex_support(t, donset, doffset, dcoh)
    w_t = ds_support_analytic(t=t, init_p=tinit, fix_point=tfixedp, slope=tslope)
    w_d = ds_support_analytic(t=t, init_p=dinit, fix_point=0, slope=dslope)    
    if sum_drifts:
        return w_t * tcohs + w_d * dcohs
    else:
        return np.column_stack((w_t * tcohs, w_d * dcohs))


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
    rel_first: bool = False,
    sum_drifts: bool = True,
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset and duration.

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
            Duration of the stimulus coherence pulse. If None, the pulse
            lasts until the end of the trial.
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
    tcohs = stimflex_support(t, tonset, toffset, tcoh)
    dcohs = stimflex_support(t, donset, doffset, dcoh)
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
    conflict_dsstimflex_drift, rel_first=False, sum_drifts=False
)


def stimflexfilt_support(
    t: np.ndarray,
    onset: float,
    offset: float,
    coh: float,
    tau_rise: float,
    tau_fall: float) -> np.ndarray:
    """Generates a stimulus with exponential rise and fall.
    
    Arguments:
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the stimulus.
        onset: float
            Time at which the stimulus begins rising.
        offset: float
            Time at which the stimulus begins falling.
        tau_rise: float
            Time constant for the exponential rise.
        tau_fall: float
            Time constant for the exponential fall.
    
    Returns
    -------
        np.ndarray: Array of stimulus values, same length as t.
    """
    stim = np.zeros_like(t)
    on = (t >= onset) & (t <= offset)
    stim[on] = coh * (1 - np.exp(-(t[on] - onset) / tau_rise))
    
    off = t > offset
    stim_at_offset = coh * (1 - np.exp(-(offset - onset) / tau_rise))
    stim[off] = stim_at_offset * np.exp(-(t[off] - onset) / tau_fall)
    return stim


def conflict_stimflexfilt_drift(
    t: np.ndarray | None,
    vt: float = 0,
    vd: float = 0,
    tcoh: float = 1.0,
    dcoh: float = 1.0,
    tonset: float = 0,
    donset: float = 0,
    toffset: float | None = None,
    doffset: float | None = None,
    taurise: float = 0.05,
    taufall: float = 0.1,
) -> np.ndarray:
    """Drift function for conflict task with stimuli with potentially variable onset and duration.
    Filters stimulus inputs with causal low-pass filter specified by tau_rise and tau_fall.

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
            Duration of the stimulus coherence pulse. If None, the pulse
            lasts until the end of the trial.
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
    tcohs = stimflexfilt_support(t, tonset, toffset, tcoh, taurise, taufall)
    dcohs = stimflexfilt_support(t, donset, doffset, dcoh, taurise, taufall)
    return vt * tcohs + vd * dcohs


def weight_window(
    t: np.ndarray | None,
    wonset: float = 0.0,
    woffset: float = 1.0,
    wmin: float = 0.0,
    wtau: float = 0.1,
) -> np.ndarray:
    """Windowed integration weight: boxcar convolved with an exponential-decay kernel.

    The boxcar equals 1 within the window [wonset, woffset] and wmin outside it. It is
    smoothed by a causal exponential kernel exp(-tau/wtau) normalized to sum 1, so the
    output is a weighted average of boxcar values and stays within [wmin, 1] ⊆ [0, 1].

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the weight.
        wonset: float
            Onset time of the integration window.
        woffset: float
            Offset time of the integration window.
        wmin: float
            Weight value outside the window (>= 0).
        wtau: float
            Time constant of the exponential decay kernel.
    Returns
    -------
        np.ndarray: Weight values in [0, 1], same length as t.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    dt = t[1] - t[0]

    boxcar = np.zeros_like(t)
    boxcar[(t >= wonset) & (t <= woffset)] = 1.0 - wmin

    # Causal exponential decay kernel, normalized to sum 1
    kernel_t = np.arange(0, 5 * wtau + dt, dt)
    kernel = np.exp(-kernel_t / wtau)
    kernel /= kernel.sum()

    # Convolve and truncate to the input length; clip guards float rounding
    smoothed = np.clip(np.convolve(boxcar, kernel, mode="full")[: len(t)], 0.0, 1.0)
    
    # Bring max inside window to 1
    max_smoothed = np.max(smoothed)
    if max_smoothed > 0:
        smoothed = (smoothed / max_smoothed) * (1.0 - wmin)
    else:
        smoothed += wmin
    return smoothed


def weight_combined(
    t: np.ndarray | None,
    wonset: float = 0.0,
    woffset: float = 1.0,
    wmin: float = 0.0,
    wtau: float = 0.1,
) -> np.ndarray:
    """Combined decision-variable weight: default ramp times windowed integration.

    Element-wise product of ds_support_analytic (init_p=0, fix_point=1, slope=2), which
    ramps from 0 to 1, and weight_window. Both factors lie in [0, 1], so does the product.
    The default ramp and windowed weights are special cases (set the other factor to 1).

    Arguments
    ---------
        t: np.ndarray
            Timepoints at which to evaluate the weight.
        wonset, woffset, wmin, wtau: float
            Parameters of the windowed integration factor (see weight_window).
    Returns
    -------
        np.ndarray: Weight values in [0, 1], same length as t.
    """
    if t is None:
        t = np.arange(0, 20, 0.1)
    w_window = weight_window(t, wonset=wonset, woffset=woffset, wmin=wmin, wtau=wtau)
    return w_window


def weight_window_ds1d(
    t: np.ndarray | None,
    tonset: float = 0.0,
    toffset: float = 1.0,
    tcoh: float = 1.0,
    donset: float = 0.0,
    doffset: float = 1.0,
    dcoh: float = 1.0,
    taurise: float = 0.05,
    taufall: float = 0.1,
    init: float = 0.5,
    asymp: float = 1.0,
    rate: float = 0.2,
    kappa: float = 1.0,
    wmin: float = 0.5,
    divnorm_method: str = "state-weight"
) -> np.ndarray:
    
    if t is None:
        t = np.arange(0, 20, 0.1)
    #target_state = asymp + (init - asymp) * np.exp(-t/rate)
    # target_state = (t > 0.4)*1
    # target_state[target_state == 0] = 1/3
    #weights = (t >= tonset) & (t <= toffset) * 1
    weights = np.ones_like(t)
    return weights
    #distractor_state = asymp - target_state
    
    #target_stim = np.abs(stimflexfilt_support(t, tonset, toffset, np.abs(tcoh) - wmin, taurise, taufall)) + wmin
    #distractor_stim = np.abs(stimflexfilt_support(t, donset, doffset, np.abs(dcoh) - wmin, taurise, taufall)) + wmin
    
    #weights = target_stim# * target_state + distractor_stim * distractor_state
    
    #unresolved_state = asymp - np.abs(target_state - distractor_state) / (2*asymp)
    # unresolved_state = asymp - target_state
    # if divnorm_method == "prod":
    #     scale = 1/(1 + kappa * target_stim * distractor_stim * unresolved_state)
    # elif divnorm_method == "sum(stim)^2*state":
    #     scale = 1/(1 + kappa * (target_stim + distractor_stim)**2 * unresolved_state)
    # elif divnorm_method == "state":
    #     scale = 1/(1 + kappa * unresolved_state)
    # elif divnorm_method == "state-weight":
    #     scale = target_state
    # else:
    #     scale = 1
    
    # return weights * scale

# Re-use drift fun but set different default args
#weight_window_ds1d_sumstim = partial(weight_window_ds1d, divnorm_method="sumstim")
#weight_window_ds1d_sumall = partial(weight_window_ds1d, divnorm_method="sumall")


# Type alias for drift functions
DriftFunction = Callable[..., np.ndarray]

attend_drift: DriftFunction = attend_drift  # noqa: PLW0127
constant: DriftFunction = constant  # noqa: PLW0127
gamma_drift: DriftFunction = gamma_drift  # noqa: PLW0127
ds_support_analytic: DriftFunction = ds_support_analytic  # noqa: PLW0127
conflict_ds_drift: DriftFunction = conflict_ds_drift  # noqa: PLW0127
conflict_dsstimflex_drift: DriftFunction = conflict_dsstimflex_drift  # noqa: PLW0127
conflict_stimflex_drift: DriftFunction = conflict_stimflex_drift  # noqa: PLW0127
conflict_stimflexfilt_drift: DriftFunction = conflict_stimflexfilt_drift  # noqa: PLW0127
conflict_dsstimflex_dual_drift: DriftFunction = conflict_dsstimflex_dual_drift  # noqa: PLW0127
weight_window: DriftFunction = weight_window  # noqa: PLW0127
weight_combined: DriftFunction = weight_combined  # noqa: PLW0127
weight_window_ds1d: DriftFunction = weight_window_ds1d  # noqa: PLW0127