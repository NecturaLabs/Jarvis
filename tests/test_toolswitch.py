"""Switching tools on and off, and how much each asks first (`jarvis.toolswitch`)."""

import pytest

from jarvis.app import ALWAYS_OFFERED, spotify_operations
from jarvis.config import BoundariesConfig, Config, ConfigError, OpClassName, SpotifyConfig
from jarvis.policy import OpClass
from jarvis.toolswitch import ToolSetting, tool_settings, with_class


def _config(**classes: OpClassName) -> Config:
    return Config(
        spotify=SpotifyConfig(client_id="id"), boundaries=BoundariesConfig(classes=classes)
    )


def _setting(config: Config, name: str) -> ToolSetting:
    return next(s for s in tool_settings(config, spotify_operations(config)) if s.name == name)


def test_every_tool_is_listed_but_how_jarvis_answers() -> None:
    config = _config()
    names = [s.name for s in tool_settings(config, spotify_operations(config))]

    assert {"web_search", "reddit_search", "spotify_play", "stop_process", "show_image"} <= set(
        names
    )
    assert not set(names) & ALWAYS_OFFERED
    assert names == sorted(names)


def test_a_tool_offers_only_the_settings_it_allows() -> None:
    config = _config()

    assert _setting(config, "stop_process").choices == (OpClass.STRONG, OpClass.DENY)
    assert _setting(config, "web_search").choices == tuple(OpClass)


def test_the_setting_shown_is_the_one_in_the_config() -> None:
    setting = _setting(_config(reddit_search="deny"), "reddit_search")

    assert setting.state is OpClass.DENY
    assert setting.shipped is OpClass.SAFE
    assert setting.label == "off"


def test_setting_a_tool_writes_its_class_and_its_shipped_class_as_no_entry() -> None:
    config = _config(weather="confirm")
    setting = _setting(config, "weather")

    off = with_class(config, setting, OpClass.DENY)
    back = with_class(off, setting, OpClass.SAFE)

    assert off.boundaries.classes == {"weather": "deny"}
    assert back.boundaries.classes == {}


def test_a_setting_a_tool_does_not_allow_is_refused() -> None:
    config = _config()

    with pytest.raises(ConfigError, match="stop_process can only be"):
        with_class(config, _setting(config, "stop_process"), OpClass.SAFE)


def test_the_next_setting_cycles_through_what_the_tool_allows() -> None:
    config = _config()
    seen: list[str] = []
    setting = _setting(config, "web_search")
    for _ in range(4):
        config = with_class(config, setting, setting.following())
        setting = _setting(config, "web_search")
        seen.append(setting.label)

    assert seen == ["asks first", "needs the confirm key", "off", "on"]
    assert _setting(config, "stop_process").following() is OpClass.DENY


def test_cycling_never_takes_a_tool_below_its_shipped_class() -> None:
    config = _config()
    seen: list[str] = []
    for _ in range(3):
        setting = _setting(config, "write_file")
        config = with_class(config, setting, setting.following())
        seen.append(_setting(config, "write_file").label)

    assert seen == ["needs the confirm key", "off", "asks first"]
