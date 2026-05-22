from menipy.common.validation import QAResult, validate
from menipy.models.context import Context
from menipy.models.geometry import Contour


def test_validate_empty_context():
    ctx = Context()
    res = validate(ctx)

    assert isinstance(res, QAResult)
    assert not res.ok
    assert "convergence" in res.checks
    assert "residuals" in res.checks
    assert "contour" in res.checks

    assert not res.checks["convergence"].passed
    assert not res.checks["contour"].passed
    assert res.checks["residuals"].passed  # No info = passed


def test_validate_success():
    ctx = Context()
    ctx.fit = {"solver": {"success": True}, "residuals": {"rmse": 2.5}}

    # 51 points contour
    xy = [[float(i), float(i)] for i in range(51)]
    ctx.contour = Contour(xy=xy)

    res = validate(ctx, thresholds={"rmse": 5.0, "min_contour_points": 50})

    assert res.ok
    assert res.checks["convergence"].passed
    assert res.checks["residuals"].passed
    assert res.checks["contour"].passed


def test_validate_failure():
    ctx = Context()
    ctx.fit = {
        "solver": {"success": True},
        "residuals": {"rmse": 10.0},  # greater than threshold
    }

    # 10 points contour (insufficient)
    xy = [[float(i), float(i)] for i in range(10)]
    ctx.contour = Contour(xy=xy)

    res = validate(ctx, thresholds={"rmse": 5.0, "min_contour_points": 50})

    assert not res.ok
    assert res.checks["convergence"].passed
    assert not res.checks["residuals"].passed
    assert not res.checks["contour"].passed


def test_to_dict():
    ctx = Context()
    res = validate(ctx)
    d = res.to_dict()
    assert "ok" in d
    assert "score" in d
    assert "checks" in d
    assert isinstance(d["checks"], dict)
