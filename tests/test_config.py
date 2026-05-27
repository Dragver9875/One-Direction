from pathlib import Path

from src.config import apply_cli_overrides, deep_get, deep_merge, deep_set, load_project_config, parse_override_value


def test_deep_merge():
    out = deep_merge({"a": {"b": 1}, "c": 2}, {"a": {"d": 3}})
    assert out["a"]["b"] == 1
    assert out["a"]["d"] == 3
    assert out["c"] == 2


def test_deep_get_set():
    cfg = {}
    deep_set(cfg, "a.b.c", 10)
    assert deep_get(cfg, "a.b.c") == 10
    assert deep_get(cfg, "x.y", "fallback") == "fallback"


def test_parse_override_value():
    assert parse_override_value("true") is True
    assert parse_override_value("12") == 12
    assert parse_override_value("3.5") == 3.5
    assert parse_override_value("[1,2,3]") == [1, 2, 3]


def test_apply_cli_overrides():
    cfg = {"a": {"b": 1}}
    out = apply_cli_overrides(cfg, ["a.b=2", "x.y=true"])
    assert out["a"]["b"] == 2
    assert out["x"]["y"] is True
