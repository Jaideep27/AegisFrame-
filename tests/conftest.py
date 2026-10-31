import pytest


@pytest.fixture(autouse=True)
def _no_installed_local_model(monkeypatch, request):
    """Tests must not change behaviour because a model happens to be installed on this machine.
    Tests that exercise the local model point AEGIS_LOCAL_MODEL at it explicitly."""
    if "uses_local_model" not in request.keywords:
        monkeypatch.setenv("AEGIS_LOCAL_MODEL", "__no_local_model_in_tests__")
