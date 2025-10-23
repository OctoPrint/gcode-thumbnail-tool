# based on OctoPrint-PrusaSlicerThumbnails plugin by jneilliii,
# released under terms of the AGPLv3

import base64
import dataclasses
import io
import logging
import os
import re
from typing import Optional

from PIL import Image
from PIL.Image import Image as PILImage

FORMAT_LOOKUP = {"JPG": "JPEG", "QOI": "QOI", "PNG": "PNG"}

REGEX_GENERIC = re.compile(
    r"^; thumbnail(?P<suffix>(_(?P<format>JPG|QOI))?) begin \d+[x ]\d+ \d+$(?P<data>.*?)^; thumbnail(?P=suffix) end",
    re.MULTILINE | re.DOTALL,
)
"""
Generic format:

    ; thumbnail begin <width>x<height> <size>
    ; <base64 encoded data>
    ; thumbnail end

    ; thumbnail begin <width> <height> <size>
    ; <base64 encoded data>
    ; thumbnail end

    ; thumbnail_JPG begin <width and height> <size>
    ; <base64 encoded data>
    ; thumbnail_JPG end

    ; thumbnail_QOI begin <width and height> <size>
    ; <base64 encoded data>
    ; thumbnail_QOI end

Spread across multiple lines.
"""

REGEX_SNAPMAKER = re.compile(
    r"^;[Tt]humbnail: data:image/png;base64,(?P<data>.+?)$", re.MULTILINE
)
"""
Luban format:

    ;thumbnail: data:image/png;base64,<base 64 encoded data><eol>

    ;Thumbnail: data:image/png;base64,<base 64 encoded data><eol>

    ;thumbnail:data:image/png;base64,<base 64 encoded data><eol>

Should be single lines only.
"""

REGEX_MKS = re.compile(
    r"(?P<prefix>;simage|;;gimage):(?P<data>.*?)M10086 ;$",
    re.DOTALL | re.MULTILINE,
)
"""
MKS format:

    ;simage:<hex encoded data>M10086;<nl>

TODO: Single lines or multiple lines? Need sample!
"""

REGEX_WEEDO = re.compile(
    r"W221(?:\n|\r\n?)(?P<lines>(W220\s+.*?(?:\n|\r\n?))+)W222",
    re.DOTALL | re.MULTILINE,
)
"""
Weedo format:

    W221<nl>
    W220 <hex encoded data><nl>
    W220 <more hex encoded data><nl>
    [...]
    W222<nl>

Spread across multiple lines
"""

REGEX_QIDI = re.compile(
    r"^M4010\s+X(?P<width>\d+)\s+Y(?P<height>\d+)(?:\n|\r\n?)(?P<lines>(M4010.*?'.*?'(?:\n|\r\n?))+)",
    re.MULTILINE,
)
"""
Qidi format:

    M4010 X<width> Y<height><nl>
    M4010 I<index pixel> T<pixel count> '<hex encoded data>'<nl>
    M4010 I<index pixel> T<pixel count> '<more hex encoded data>'<nl>
    [...]
    <something else>

Single line? TODO: Need sample
"""

REGEX_CREALITY = re.compile(
    r"^; jpg begin.*$(?P<data>.+?)^; jpg end",
    re.MULTILINE | re.VERBOSE,
)
"""
Creality format:

    ; jpg begin ...
    ; <base64 encoded data>
    ; jpg end

Spread across multiple lines
"""

REGEX_EXTRUSION_COMMAND = re.compile(r"^\s*G1.*E\d+")
REGEX_COMMENTED_LINE = re.compile(r"^\s*;\s*", re.MULTILINE)


@dataclasses.dataclass
class ExtractedImages:
    extractor: str
    images: list[PILImage]


@dataclasses.dataclass
class ExtractedBytes:
    extractor: str
    images: dict[str, bytes]


def _potential_lines(path: str) -> str:
    if not os.path.exists(path):
        return ""

    line_no = 0
    content_lines = []
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        line_no = 0
        for line in f:
            line_no += 1

            if REGEX_EXTRUSION_COMMAND.match(line):
                logging.getLogger(__name__).info(
                    f"Detected first extrusion at line {line_no}, that ends the possible thumbnail locations in the file"
                )
                break

            content_lines.append(line)

    return "".join(content_lines).replace("\r\n", "\n").replace(";\n;\n", ";\n\n;\n")


def extract_thumbnails_from_gcode(gcode_path: str) -> Optional[ExtractedImages]:
    logger = logging.getLogger(__name__)

    logger.info(f"Extracting thumbnails from {gcode_path}...")

    potential_lines = _potential_lines(gcode_path)
    logger.debug(
        f"Searching for matches in:\n{_prefix_lines(potential_lines, prefix=' | ')}"
    )

    extractors = [
        ("generic", (REGEX_GENERIC, _extract_generic_base64_thumbnails)),
        ("snapmaker", (REGEX_SNAPMAKER, _extract_generic_base64_thumbnails)),
        ("mks", (REGEX_MKS, _extract_mks_thumbnails)),
        ("weedo", (REGEX_WEEDO, _extract_weedo_thumbnails)),
        ("qidi", (REGEX_QIDI, _extract_qidi_thumbnails)),
        ("flashprint", _extract_flashprint_thumbnails),
        ("creality", (REGEX_CREALITY, _extract_generic_base64_thumbnails)),
    ]

    for name, tooling in extractors:
        if isinstance(tooling, tuple):
            # regex based extractor
            regex, extractor = tooling

            matches = list(regex.finditer(potential_lines))
            if matches:
                logger.debug(f"Detected {name} thumbnails, extracting...")
                return ExtractedImages(extractor=name, images=extractor(matches))

        elif callable(tooling):
            # custom extractor function
            thumbnails = tooling(gcode_path, potential_lines)
            if thumbnails:
                return ExtractedImages(extractor=name, images=thumbnails)

    # none of the regex based extractors matched, could this be flashprint?
    with open(gcode_path, "rb") as f:
        f.seek(58)
        buffer = f.read(14454)
        if buffer[0] == 0x42 and buffer[1] == 0x4D:  # BMP magic numbers
            logger.debug("Detected flashprint thumbnails, extracting...")
            return ExtractedImages(
                extractor="flashprint", images=_extract_flashprint_thumbnails(buffer)
            )

    # if we reach this point, we could not find any thumbnails
    return None


def extract_thumbnail_bytes_from_gcode(
    gcode_path: str, format="PNG"
) -> Optional[ExtractedBytes]:
    result = extract_thumbnails_from_gcode(gcode_path)
    if not result:
        return None

    data = {}
    for image in result.images:
        sizehint = _image_to_sizehint(image)
        if sizehint in data:
            continue
        data[sizehint] = _image_to_bytes(image, format=format)
    return ExtractedBytes(extractor=result.extractor, images=data)


# ~~ extractors


def _extract_generic_base64_thumbnails(matches: list[re.Match]) -> list[PILImage]:
    """
    Extracts thumbnails from base64 encoded data

    Will remove any comment prefixes from lines.

    Expected match groups:
    * ``data``: base 64 encoded data
    * ``format`` (optional): image format, e.g. "JPG", "BMP", "QOI"
    """
    result = []
    for match in matches:
        data = _remove_comment_prefix(match.group("data"))

        formats = None
        if "format" in match.groups():
            format = FORMAT_LOOKUP.get(match.group("format"))
            if format:
                formats = [format]

        image = _image_from_bytes(base64.b64decode(data.encode()), formats=formats)
        result.append(image)
    return result


def _extract_generic_hex_thumbnails(matches: list[re.Match]) -> list[PILImage]:
    """
    Extracts thumbnails from hex encoded data

    Will remove any comment prefixes from lines.

    Expected match groups:
    * ``data``: hex encoded data
    """
    result = []
    for match in matches:
        data = _remove_whitespace(_remove_comment_prefix(match.group("data")))
        image = _image_from_hex(data)
        result.append(image)
    return result


def _extract_mks_thumbnails(matches: list[re.Match]) -> list[PILImage]:
    """Extracts a thumbnail from hex binary data used by MKS printers"""

    OPTIONS = {";;gimage": (200, 200), ";simage": (100, 100)}

    result = []

    extracted = set()
    for match in matches:
        for prefix, dimensions in OPTIONS.items():
            if prefix in extracted:
                continue

            if match.group("prefix") != prefix:
                continue

            encoded_image = bytes(
                bytearray.fromhex(_remove_whitespace(match.group("data")))
            )

            image = Image.frombytes(
                "RGB", dimensions, encoded_image, "raw", "BGR;16", 0, 1
            )
            result.append(image)
            extracted.add(prefix)

            break

    return result


def _extract_weedo_thumbnails(matches: list[re.Match]) -> list[PILImage]:
    result = []

    for match in matches:
        lines = [
            line[len("M220 ") :].strip() for line in match.group("lines").splitlines()
        ]

        hex_data = _remove_whitespace("".join(lines))
        result.append(_image_from_hex(hex_data))

    return result


def _extract_qidi_thumbnails(matches: list[re.Match]) -> list[PILImage]:
    """
    Qidi extractor

    Thumbnails as a sequence of pixels encoded into 2 bytes each, with
    5 bits per channel:

    .. code-block:: none

       | r | r | r | r | r | g | g | g | g | g | f | b | b | b | b | b |
         15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0

    If ``f`` at bit index 5 is set, the pixel is repeated and the count
    follows in a second 2 byte sequence:

    .. code-block:: none

       | 0 | 0 | 1 | 1 | c | c | c | c | c | c | c | c | c | c | c | c |
         15  14  13  12  11  10   9   8   7   6   5   4   3   2   1   0

    The pixel count should match the width x height.

    To convert to 24 bit color, bitshift to the right as needed to fetch
    the 5 bits of the channel and then shift each value by 3 bits to the left
    again.
    """

    logger = logging.getLogger(__name__)

    RUNLENGTH_MASK_PIXEL = 32
    RUNLENGTH_MASK_NUM = 12288

    def pop4b(byts: bytearray):
        v1, v2 = byts.pop(0), byts.pop(0)
        return v1 << 8 | v2

    def val2rgb(val: int) -> tuple[int, int, int]:
        r = ((val >> 11) & 31) << 3
        g = ((val >> 6) & 31) << 3
        b = ((val) & 31) << 3
        return r, g, b

    result = []

    for match in matches:
        width = int(match.group("width"))
        height = int(match.group("height"))

        hex_bytes = bytearray()
        for line in match.group("lines").splitlines():
            data = line.split("'")[1]
            decoded = bytearray.fromhex(data)
            hex_bytes.extend(decoded)

        logger.debug(f"Extracted {len(hex_bytes)} bytes")

        pixel_data = bytearray()
        pixel_count = 0
        while len(hex_bytes):
            val = pop4b(hex_bytes)
            pixel = val2rgb(val)

            if val & RUNLENGTH_MASK_PIXEL == RUNLENGTH_MASK_PIXEL:
                # this is a repeated pixel, fetch the count!
                val = pop4b(hex_bytes)
                assert val & RUNLENGTH_MASK_NUM == RUNLENGTH_MASK_NUM

                for _ in range(val & 4095):
                    pixel_data.extend(pixel)
                    pixel_count += 1

            else:
                # just add the pixel
                pixel_data.extend(pixel)
                pixel_count += 1

        logger.debug(f"Width: {width}, height: {height}, pixels: {pixel_count}")
        assert pixel_count == width * height

        image = Image.frombytes("RGB", (width, height), pixel_data, "raw", "RGB", 0, 1)
        result.append(image)

    return result


def _extract_flashprint_thumbnails(path: str, _lines: str) -> list[PILImage]:
    with open(path, "rb") as f:
        f.seek(58)
        buffer = f.read(14454)

    if not (buffer[0] == 0x42 and buffer[1] == 0x4D):  # BMP magic numbers
        return {}

    def dark_to_transparent(
        pixel: tuple[int, int, int, int], threshold=35
    ) -> tuple[int, int, int, int]:
        r, g, b, _ = pixel
        if all(lambda value: value <= threshold, (r, g, b)):
            return (255, 255, 255, 0)
        return pixel

    image = Image.open(io.BytesIO(buffer)).resize((160, 120))

    rgba = image.convert("RGBA")
    rgba.putdata([dark_to_transparent(pixel) for pixel in rgba.getdata()])

    return [rgba]


# ~~ image helpers


def _image_to_bytes(image: PILImage, format="PNG") -> bytes:
    """Returns PIL ``image`` as bytes in the requested image ``format``"""
    with io.BytesIO() as png_bytes:
        image.save(png_bytes, format=format)
        return png_bytes.getvalue()


def _image_to_sizehint(image: PILImage) -> str:
    w, h = image.size
    return f"{w}x{h}"


def _image_from_hex(encoded: str) -> PILImage:
    encoded_image = bytes(bytearray.fromhex(encoded))
    return _image_from_bytes(encoded_image)


def _image_from_bytes(data: bytes, formats=None) -> list[PILImage]:
    return Image.open(io.BytesIO(data), formats=formats)


# ~~ other helpers


def _prefix_lines(lines: str, prefix="", nl="\n") -> str:
    if not prefix:
        return lines

    return nl.join([f"{prefix}{line}" for line in lines.splitlines()])


def _remove_comment_prefix(data: str) -> str:
    return _remove_line_prefix(data, prefix=r"^; ")


def _remove_line_prefix(data: str, prefix: str) -> str:
    if not data:
        return data

    return re.sub(prefix, "", data, flags=re.MULTILINE)


def _remove_whitespace(data: str) -> str:
    if not data:
        return data

    return re.sub(r"\s+", "", data)


# ~~ Command line interface


def _setup_logging(verbosity: int = 0):
    import logging.config

    VERBOSITY_MAP = ["WARN", "INFO", "DEBUG"]

    if verbosity is None:
        verbosity = 0
    elif verbosity > 2:
        verbosity = 2

    config = {
        "version": 1,
        "formatters": {"simple": {"format": "[%(levelname)s] %(message)s"}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": VERBOSITY_MAP[verbosity], "handlers": ["console"]},
    }
    logging.config.dictConfig(config)


def main():
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(
        prog="gcode-thumbnail-tool",
        description="A small CLI tool to extract thumbnail images from GCODE files",
    )
    parser.add_argument("--verbose", "-v", dest="verbosity", action="count")
    subparsers = parser.add_subparsers(dest="subcommand")

    extract_parser = subparsers.add_parser(
        "extract", help="Extracts thumbnails from the provided file as PNGs"
    )
    extract_parser.add_argument("path", help="Path to the GCODE file")
    extract_parser.add_argument(
        "-o", "--output", dest="output", help="Output path for the extracted thumbnails"
    )

    analyse_parser = subparsers.add_parser(
        "analyse", help="Provides information on the GCODE file's thumbnails"
    )
    analyse_parser.add_argument("path", help="Path to the GCODE file")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    _setup_logging(args.verbosity)

    if args.subcommand == "extract":
        if not os.path.exists(args.path):
            print(f"{args.path} doesn't exist, exiting!", file=sys.stderr)
            sys.exit(1)

        result = extract_thumbnail_bytes_from_gcode(args.path, format="PNG")
        if result:
            output_folder = args.output
            if not output_folder:
                output_folder = os.getcwd()

            print(f"Extracting thumbnails to {output_folder}...")

            gcode_name = os.path.splitext(os.path.basename(args.path))[0]

            for sizehint, image in result.images.items():
                output_name = f"{gcode_name}.thumb-{sizehint}.png"
                output_path = os.path.join(output_folder, output_name)
                with open(output_path, mode="wb") as f:
                    f.write(image)
                print(f"\tExtracted {output_name}")

        else:
            print(f"Didn't find any thumbnails in {args.path}")

    elif args.subcommand == "analyse":
        if not os.path.exists(args.path):
            print(f"{args.path} doesn't exist, exiting!", file=sys.stderr)
            sys.exit(1)

        result = extract_thumbnails_from_gcode(args.path)
        if result:
            print(
                f'Found {len(result.images)} thumbnails in {args.path}, in format "{result.extractor}":'
            )
            for image in result.images:
                print(f"\t{image.format} @ {image.width}x{image.height}")
        else:
            print(f"Didn't find any thumbnails in {args.path}")


if __name__ == "__main__":
    main()
