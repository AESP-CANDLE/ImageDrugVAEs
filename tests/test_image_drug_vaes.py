#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `image_drug_vaes` package."""


import unittest
from click.testing import CliRunner

from image_drug_vaes import image_drug_vaes
from image_drug_vaes import cli


class TestImage_drug_vaes(unittest.TestCase):
    """Tests for `image_drug_vaes` package."""

    def setUp(self):
        """Set up test fixtures, if any."""

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_something(self):
        """Test something."""

    def test_command_line_interface(self):
        """Test the CLI."""
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        assert 'image_drug_vaes.cli.main' in result.output
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output
