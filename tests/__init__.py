"""Test-environment compatibility shim.

The shipped pickle was created with scikit-learn 1.4.2. If the tests run under a different
scikit-learn (e.g. in a sandbox where 1.4.2 can't be installed), newer SimpleImputer code
expects an attribute `_fill_dtype` that old pickles lack. For float data it equals
`_fit_dtype`, so we emulate it here. Production does NOT use this: the app refuses to
start on a mismatched scikit-learn unless ALLOW_SKLEARN_MISMATCH=true. With the pinned
1.4.2 this shim does nothing.
"""
import joblib
import sklearn

_original_load = joblib.load


def _load(*args, **kwargs):
    obj = _original_load(*args, **kwargs)
    if sklearn.__version__.split(".")[:2] != ["1", "4"]:
        for _, step in getattr(obj, "steps", []):
            if type(step).__name__ == "SimpleImputer" and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = step._fit_dtype
    return obj


joblib.load = _load
