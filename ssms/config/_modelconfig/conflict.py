"""Configuration for conflict models with dynamical drift."""

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


def get_conflict_stimflexrel1_leak_config():
    return _new_config(
        name="conflict_stimflexrel1_leak",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            g=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_drift",
        drift_fun=df.conflict_stimflexrel1_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak,
    )


def get_conflict_stimflexrel1_leak2_config():
    return _new_config(
        name="conflict_stimflexrel1_leak2",
        param_dict=dict(
            a=_new_param(2.0, 0.3, 3.0),
            z=_new_param(0.5, 0.1, 0.9),
            t=_new_param(1.0, 1e-3, 2.0),
            vt=_new_param(2.0, 0.0, 5.0),
            vd=_new_param(2.0, 0.0, 5.0),
            tcoh=_new_param(0.5, -1.0, 1.0),
            dcoh=_new_param(-0.5, -1.0, 1.0),
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            gt=_new_param(0.0, 0.0, 1.0),
            gd=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_stimflexrel1_dual_drift",
        drift_fun=df.conflict_stimflexrel1_dual_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak2,
    )


def _dsstimflex_weight_param_dict():
    """Shared param_dict for the master conflict_dsstimflex_weight model and its
    angle variant."""
    return dict(
        a=_new_param(2.0, 0.3, 3.0),
        z=_new_param(0.5, 0.1, 0.9),
        t=_new_param(1.0, 1e-3, 2.0),
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
        vtaurise=_new_param(0.05, 1e-3, 0.5),
        vtaufall=_new_param(0.1, 1e-3, 0.5),
        wmin=_new_param(0.5, 0.0, 1.0),
        wtaurise=_new_param(0.05, 1e-3, 0.5),
        wtaufall=_new_param(0.1, 1e-3, 0.5),
    )


def get_conflict_dsstimflex_dsweight_config():
    return _new_config(
        name="conflict_dsstimflex_dsweight",
        param_dict=_dsstimflex_weight_param_dict() | dict(
            winit=_new_param(0.0, 0.0, 1.0),
            wslope=_new_param(0.15, 0.01, 1.0)),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
        weight_name="weight_window_ds",
        weight_fun=df.weight_window_ds,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflex_weight_config():
    return _new_config(
        name="conflict_dsstimflex_weight",
        param_dict=_dsstimflex_weight_param_dict(),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
        weight_name="weight_window",
        weight_fun=df.weight_window,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflex_weight_angle_config():
    return _new_config(
        name="conflict_dsstimflex_weight_angle",
        param_dict=_dsstimflex_weight_param_dict() | dict(
            theta=_new_param(0.0, 0.0, 1.3),
        ),
        boundary_name="angle",
        boundary=bf.angle,
        drift_name="conflict_dsstimflex_drift",
        drift_fun=df.conflict_dsstimflex_drift,
        weight_name="weight_window",
        weight_fun=df.weight_window,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_weight,
    )


def get_conflict_dsstimflex_leak2_config():
    return _new_config(
        name="conflict_dsstimflex_leak2",
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
            tonset=_new_param(0.0, 0.0, 1.0),
            donset=_new_param(0.0, 0.0, 1.0),
            toffset=_new_param(0.2, 0.0, 1.0),
            doffset=_new_param(0.2, 0.0, 1.0),
            gt=_new_param(0.0, 0.0, 1.0),
            gd=_new_param(0.0, 0.0, 1.0),
        ),
        boundary_name="constant",
        boundary=bf.constant,
        drift_name="conflict_dsstimflex_dual_drift",
        drift_fun=df.conflict_dsstimflex_dual_drift,
        choices=[-1, 1],
        n_particles=1,
        simulator=cssm.ddm_flex_leak2,
    )
