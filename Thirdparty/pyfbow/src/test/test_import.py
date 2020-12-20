#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx


def test_import_main():
    import pyDBoW3

    assert('Vocabulary' in pyDBoW3)
    assert('Database' in pyDBoW3)


if __name__ == "__main__":
    import os

    basename = os.path.basename(__file__)
    pytest.main([basename, "-s", "--tb=native"])
