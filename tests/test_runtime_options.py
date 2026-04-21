from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

from cyberdrop_dl.config import GlobalSettings
from cyberdrop_dl.config.config_model import ConfigSettings
from cyberdrop_dl.managers.manager import Manager
from cyberdrop_dl.utils import yaml
from cyberdrop_dl.utils.args import ParsedArgs

TEST_DIR = Path("tmp_runtime_options_tests")


def test_runtime_options_deep_scrape_defaults_true() -> None:
    settings = ConfigSettings()

    assert settings.runtime_options.deep_scrape is True


def test_runtime_options_deep_scrape_yaml_round_trip_defaults_true() -> None:
    shutil.rmtree(TEST_DIR, ignore_errors=True)
    TEST_DIR.mkdir()
    config_file = TEST_DIR / "settings.yaml"

    try:
        yaml.save(config_file, ConfigSettings())
        serialized_config = yaml.load(config_file)

        assert serialized_config["runtime_options"]["deep_scrape"] is True
        assert ConfigSettings.model_validate(serialized_config).runtime_options.deep_scrape is True
    finally:
        shutil.rmtree(TEST_DIR, ignore_errors=True)


def test_args_consolidation_respects_explicit_deep_scrape_false() -> None:
    manager = Manager.__new__(Manager)
    manager.parsed_args = ParsedArgs.model_validate({})
    manager.config_manager = SimpleNamespace(
        settings_data=ConfigSettings.model_validate({"runtime_options": {"deep_scrape": False}}),
        global_settings_data=GlobalSettings(),
        deep_scrape=False,
    )

    Manager.args_consolidation(manager)

    assert manager.config_manager.settings_data.runtime_options.deep_scrape is False
    assert manager.config_manager.deep_scrape is False
