# Unit tests for the API

import os

import pytest

import gcode_thumbnail_tool as gtt

from . import const


def _id_to_attr(prefix: str, id: str) -> str:
    return f"{prefix.upper()}_" + id.upper().replace(".", "_")


def _pytest_param(prefix1: str, prefix2: str, id: str) -> pytest.param:
    return pytest.param(
        getattr(const, _id_to_attr(prefix1, id)),
        getattr(const, _id_to_attr(prefix2, id)),
        id=id,
    )


@pytest.mark.parametrize(
    "input, expected",
    [_pytest_param("SAMPLE", "EXPECTED", id) for id in const.IDS],
)
def test_extract_thumbnails_from_gcode(
    input: str, expected: tuple[str, list[tuple[str, int, int]]]
):
    actual = gtt.extract_thumbnails_from_gcode(input)
    _verify_thumbnails(actual, expected)


@pytest.mark.parametrize(
    "filename, expected", [_pytest_param("FILE", "EXPECTED", id) for id in const.IDS]
)
def test_extract_thumbnails_from_gcode_file(
    filename: str, expected: tuple[str, list[tuple[str, int, int]]]
):
    path = os.path.join(os.path.dirname(__file__), "_files", filename)
    actual = gtt.extract_thumbnails_from_gcode_file(path)
    _verify_thumbnails(actual, expected)


@pytest.mark.parametrize(
    "input, expected",
    [_pytest_param("SAMPLE", "EXPECTED_BYTES", id) for id in const.IDS],
)
def test_extract_thumbnail_bytes_from_gcode(
    input: str, expected: tuple[str, list[tuple[str, int, int]]]
):
    actual = gtt.extract_thumbnail_bytes_from_gcode(input)
    _verify_thumbnail_bytes(actual, expected)


@pytest.mark.parametrize(
    "filename, expected",
    [_pytest_param("FILE", "EXPECTED_BYTES", id) for id in const.IDS],
)
def test_extract_thumbnail_bytes_from_gcode_file(
    filename: str, expected: tuple[str, list[tuple[str, int, int]]]
):
    path = os.path.join(os.path.dirname(__file__), "_files", filename)
    actual = gtt.extract_thumbnail_bytes_from_gcode_file(path)
    _verify_thumbnail_bytes(actual, expected)


def _verify_thumbnails(
    thumbnails: gtt.ExtractedImages,
    expected_thumbnails: tuple[str, list[tuple[str, int, int]]],
):
    extractor, expected_thumbnail_data = expected_thumbnails

    assert thumbnails is not None
    assert thumbnails.extractor == extractor
    assert len(thumbnails.images) == len(expected_thumbnail_data)

    for actual, expected in zip(thumbnails.images, expected_thumbnail_data):
        fmt, width, height = expected
        assert actual.format == fmt
        assert actual.width == width
        assert actual.height == height


def _verify_thumbnail_bytes(
    thumbnails: gtt.ExtractedBytes,
    expected_thumbnails: tuple[str, list[tuple[str, int, int]]],
):
    extractor, expected_thumbnail_data = expected_thumbnails

    assert thumbnails is not None
    assert thumbnails.extractor == extractor
    assert len(thumbnails.images) == len(expected_thumbnail_data)

    for actual, expected in zip(thumbnails.images.keys(), expected_thumbnail_data):
        assert actual == expected
