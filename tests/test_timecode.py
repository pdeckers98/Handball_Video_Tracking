import pytest

from handball.timecode import format_timecode, parse_timecode


@pytest.mark.parametrize(
    "text, expected",
    [("75.5", 75.5), ("1:15.5", 75.5), ("0:01:15.5", 75.5), ("1:02:03", 3723.0), (12, 12.0)],
)
def test_parse(text, expected):
    assert parse_timecode(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "1::2", "a:b", "1:2:3:4", "-5"])
def test_parse_invalid(text):
    with pytest.raises(ValueError):
        parse_timecode(text)


def test_format():
    assert format_timecode(75.5) == "1:15.50"
    assert format_timecode(3723) == "1:02:03.00"
    assert parse_timecode(format_timecode(498.25)) == pytest.approx(498.25)
