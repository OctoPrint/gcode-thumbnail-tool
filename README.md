# gcode-thumbnail-tool

A small tool to analyse and extract thumbnails rendered into GCODE files.

## Installation

```
pip install gcode-thumbnail-tool
```

## Usage

### As a Python library

```python
import gcode_thumbnail_tool as gtt

path = "/path/to/a/file.gcode"
thumbnails = gtt.extract_thumbnails_from_gcode_file(path)

if result:
    print(
        f'Found {len(result.images)} thumbnails in {args.path}, in format "{result.extractor}":'
    )
    for image in result.images:
        print(f"\t{image.format} @ {image.width}x{image.height}")
else:
    print(f"Didn't find any thumbnails in {args.path}")
```

See `gcode_thumbnail_tool.py:main` and `tests/test_api.py` for more usage examples.

### As a command line tool

<!--INSERT:help-->

```
usage: gcode-thumbnail-tool [-h] [--verbose] {extract,analyse} ...

A small CLI tool to extract thumbnail images from GCODE files.

positional arguments:
  {extract,analyse}
    extract          extracts thumbnails from the provided file as PNGs
    analyse          provides information on the GCODE file's thumbnails

options:
  -h, --help         show this help message and exit
  --verbose, -v      increase logging verbosity
```

<!--/INSERT:help-->

#### `extract`

<!--INSERT:extract-->

```
usage: gcode-thumbnail-tool extract [-h] [-o OUTPUT] path

positional arguments:
  path                  path to the GCODE file

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        output path for the extracted thumbnails
```

<!--/INSERT:extract-->

#### `analyse`

<!--INSERT:analyse-->

```
usage: gcode-thumbnail-tool analyse [-h] path

positional arguments:
  path        path to the GCODE file

options:
  -h, --help  show this help message and exit
```

<!--/INSERT:analyse-->

## Acknowledgements

`gcode-thumbnail-tool` is based on the [Slicer Thumbnails OctoPrint Plugin](https://github.com/jneilliii/OctoPrint-PrusaSlicerThumbnails) by jneilliii.
Big thanks to him and all the contributors!
