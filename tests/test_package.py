from hardsector_tool import __version__


def test_version_is_defined() -> None:
    """Ensure the package exposes a version string."""
    assert isinstance(__version__, str) and __version__
