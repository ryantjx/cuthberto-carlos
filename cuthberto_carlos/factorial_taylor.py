"""Minor fix to `cuthbert.gaussian.taylor` to support factorial inference.

This is really a minor shape handling point allowing different `linearization_point`s
in `init_prepare` and `filter_prepare`. This should shortly be fixed in `cuthbert`.
The only change from https://github.com/state-space-models/cuthbert/blob/main/cuthbert/gaussian/taylor/filter.py
is to add the optional `get_init_log_density_filter_prepare` arg, (as well as removing
`associative=True` support for brevity here).
"""

from functools import partial
from cuthbert.gaussian.taylor import non_associative_filter
from cuthbert.gaussian.taylor.types import (
    GetDynamicsLogDensity,
    GetInitLogDensity,
    GetObservationFunc,
)
from cuthbert.inference import Filter


def build_filter(
    get_init_log_density: GetInitLogDensity,
    get_dynamics_log_density: GetDynamicsLogDensity,
    get_observation_func: GetObservationFunc,
    get_init_log_density_filter_prepare: GetInitLogDensity | None = None,
    rtol: float | None = None,
    ignore_nan_dims: bool = False,
) -> Filter:
    """Build linearized Taylor Kalman inference filter.

    Args:
        get_init_log_density: Function to get log density log p(x_0)
            and linearization point.
            Only takes `model_inputs` as input.
        get_dynamics_log_density: Function to get dynamics log density log p(x_t+1 | x_t)
            and linearization points (for the previous and current time points)
            If `associative` is True, the `state` argument should be ignored.
        get_observation_func: Function to get observation function (either conditional
            log density or log potential), linearization point and optional observation
            (not required for log potential functions).
            If `associative` is True, the `state` argument should be ignored.
        get_init_log_density_filter_prepare: Optional GetInitLogDensity to be sent to
            `filter_prepare` for shape inference there. Defaults to `get_init_log_density`
        rtol: The relative tolerance for the singular values of precision matrices
            when passed to `symmetric_inv_sqrt` during linearization.
            Cutoff for small singular values; singular values smaller than
            `rtol * largest_singular_value` are treated as zero.
            The default is determined based on the floating point precision of the dtype.
            See https://docs.jax.dev/en/latest/_autosummary/jax.numpy.linalg.pinv.html.
        ignore_nan_dims: Whether to treat dimensions with NaN on the diagonal of the
            precision matrices (found via linearization) as missing and ignore all rows
            and columns associated with them.

    Returns:
        Linearized Taylor Kalman filter object.
    """
    if get_init_log_density_filter_prepare is None:
        get_init_log_density_filter_prepare = get_init_log_density

    return Filter(
        init_prepare=partial(
            non_associative_filter.init_prepare,
            get_init_log_density=get_init_log_density,
            rtol=rtol,
            ignore_nan_dims=ignore_nan_dims,
        ),  # type: ignore
        # TODO: remove ignore - seems ty is complaining about keyword-only-ness, I don't think pyright does
        filter_prepare=partial(
            non_associative_filter.filter_prepare,
            get_init_log_density=get_init_log_density_filter_prepare,
        ),  # type: ignore -
        # TODO: same as above
        filter_combine=partial(
            non_associative_filter.filter_combine,
            get_dynamics_log_density=get_dynamics_log_density,
            get_observation_func=get_observation_func,
            rtol=rtol,
            ignore_nan_dims=ignore_nan_dims,
        ),
        associative=False,
    )
