import pytest
from dp_pipeline import resnet20


def test_resnet20_requires_torch():
    with pytest.raises(RuntimeError):
        resnet20()
