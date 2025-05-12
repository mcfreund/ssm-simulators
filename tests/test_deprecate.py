import pytest

from ssms.config.generator_config.data_generator_config import (
    DeprecatedDict,
    get_default_generator_config,
)


def test_deprecated_dict():
    """
    Test the DeprecatedDict class to ensure it raises a DeprecationWarning
    when accessed.
    """
    _match = "Accessing this configuration dict is deprecated and will be removed in a future version. Use `get_default_generator_config` instead."
    with pytest.warns(DeprecationWarning, match=_match):
        deprecated_dict = DeprecatedDict(
            get_default_generator_config, "get_default_generator_config"
        )
    assert isinstance(
        deprecated_dict, DeprecatedDict
    ), "The object is not an instance of DeprecatedDict"
