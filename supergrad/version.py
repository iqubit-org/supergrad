_release_version = '0.2.1'


def _get_version_for_build() -> str:
    """Determine the version at build time."""
    if _release_version is not None:
        return _release_version


def _get_version_string() -> str:
    # The build/source distribution could overwrite _release_version.
    if _release_version is not None:
        return _release_version


__version__ = _get_version_string()
