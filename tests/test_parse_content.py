from pydantic import BaseModel
from llmcast.parser.utils import parse_content


class Result(BaseModel):
    value: int
    name: str


def test_parse_json():
    result = parse_content('{"value": 42, "name": "test"}', Result)
    assert result is not None
    assert result.value == 42
    assert result.name == "test"


def test_parse_json_strips_fences():
    result = parse_content('```json\n{"value": 1, "name": "a"}\n```', Result)
    assert result is not None
    assert result.value == 1


def test_parse_json_strips_plain_fences():
    result = parse_content('```\n{"value": 2, "name": "b"}\n```', Result)
    assert result is not None
    assert result.value == 2


def test_parse_yaml():
    result = parse_content("value: 10\nname: hello", Result, result_format="yaml")
    assert result is not None
    assert result.value == 10
    assert result.name == "hello"


def test_parse_yaml_strips_fences():
    result = parse_content(
        "```yaml\nvalue: 5\nname: y\n```", Result, result_format="yaml"
    )
    assert result is not None
    assert result.value == 5


def test_parse_toml():
    result = parse_content('value = 7\nname = "world"', Result, result_format="toml")
    assert result is not None
    assert result.value == 7
    assert result.name == "world"


def test_invalid_json_returns_none():
    result = parse_content("not valid json", Result)
    assert result is None


def test_invalid_yaml_returns_none():
    result = parse_content("value: [unclosed", Result, result_format="yaml")
    assert result is None


def test_schema_mismatch_returns_none():
    result = parse_content('{"value": "not_an_int", "name": "x"}', Result)
    assert result is None
