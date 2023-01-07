[![MIT](https://img.shields.io/github/license/finsberg/fenicsx-plotly)](https://github.com/finsberg/fenicsx-plotly/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/fenicsx-plotly.svg)](https://pypi.org/project/fenicsx-plotly/)
[![Test package](https://github.com/finsberg/fenicsx-plotly/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/fenicsx-plotly/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/fenicsx-plotly/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/fenicsx-plotly/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/pkgs/container/fenicsx-plotly)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-plotly-coverage.json)](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/finsberg/a7290de789564f03eb6b1ee122fce423/raw/fenicsx-plotly-coverage.json)

# fenicsx-plotly

`fenicsx-plotly` is package for plotting FEniCSx objects using plotly. It is a successor of [`fenics-plotly`](https://github.com/finsberg/pulse).

* Documentation: https://finsberg.github.io/fenicsx-plotly/
* Source code: https://github.com/finsberg/fenicsx-plotly

## Install

To install `fenicsx-plotly` you need to first [install FEniCSx](https://github.com/FEniCS/dolfinx#installation). Next you can install `fenicsx-plotly` via pip
```
python3 -m pip install fenicsx-plotly
```
We also provide a pre-built docker image with FEniCSx and `fenicsx-plotly` installed. You pull this image using the command
```
docker pull ghcr.io/finsberg/fenicsx-plotly:v0.2.0
```

## Simple Example
```python
import dolfinx
from mpi4py import MPI
from fenicsx_plotly import plot

mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3)

plot(mesh)
```

## Supported objects (and object that will be supported in the future)
- [x] dolfinx.mesh.Mesh
- [x] dolfinx.mesh.MeshTagsMetaClass
- [x] dolfinx.fem.FunctionSpace
- [x] dolfinx.fem.Function
    - [x] Scalar
    - [x] Vector
- [ ] dolfinx.fem.bcs.DirichletBCMetaClass
    - [x] Scalar
    - [ ] Vector


## Usage with JupyterBook
If you want to embed the visualizations generated by `fenicsx-plotly` into a webpage generated by JupyterBook such as [the documentation for `fenicsx-plotly`](https://finsberg.github.io/fenicsx-plotly/) you need to add the following configuration in your `_config.yml`
```yaml
sphinx:
  config:
    html_js_files:
      - https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js
```
See <https://jupyterbook.org/en/stable/interactive/interactive.html#plotly> for more information.

## Contributing
Contributions are welcomed!

See https://finsberg.github.io/fenicsx-plotly/CONTRIBUTING.html for more info.
