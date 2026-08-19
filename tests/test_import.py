"""Smoke tests for the installed package."""

import importlib


def test_package_imports() -> None:
    package = importlib.import_module("rxn_checker")

    assert callable(package.load_case)
