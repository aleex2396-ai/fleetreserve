"""The package imports and carries a version."""

import fleetreserve


def test_imports_with_version():
    assert isinstance(fleetreserve.__version__, str)
    assert fleetreserve.__version__
