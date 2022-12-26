[![MIT](https://img.shields.io/github/license/finsberg/fenicsx-plotly)](https://github.com/finsberg/fenicsx-plotly/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/fenicsx-plotly.svg)](https://pypi.org/project/fenicsx-plotly/)
[![Test package](https://github.com/finsberg/fenicsx-plotly/actions/workflows/test_package_coverage.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/test_package_coverage.yml)
[![Pre-commit](https://github.com/finsberg/fenicsx-plotly/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/pre-commit.yml)
[![Deploy static content to Pages](https://github.com/finsberg/fenicsx-plotly/actions/workflows/build_docs.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/actions/workflows/build_docs.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Create and publish a Docker image](https://github.com/finsberg/fenicsx-plotly/actions/workflows/docker-image.yml/badge.svg)](https://github.com/finsberg/fenicsx-plotly/pkgs/container/fenicsx-plotly)

# fenicsx-plotly

`fenicsx-plotly` is package for plotting FEniCSx objects using plotly. It is a successor of [`fenics-plotly`](https://github.com/finsberg/pulse).

---

## Notice

**This repo is a complete rewrite of `fenics-plotly` to work with FEniCSx. The package is not yet ready for release.**

If you are using FEniCS please check out [`fenics-plotly`](https://github.com/finsberg/fenics-plotly) instead

---

* Documentation: https://finsberg.github.io/fenicsx-plotly/
* Source code: https://github.com/finsberg/fenicsx-plotly

## Install

To install `fenicsx-plotly` you need to first [install FEniCSx](https://github.com/FEniCS/dolfinx#installation). Next you can install `fenicsx-plotly` via pip
```
python3 -m pip install fenicsx-plotly
```
We also provide a pre-built docker image with FEniCSx and `fenicsx-plotly` installed. You pull this image using the command
```
docker pull ghcr.io/finsberg/fenicsx-plotly:v0.1.1
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
- [] dolfinx.mesh.MeshTagsMetaClass
- [x] dolfinx.fem.FunctionSpace
- [] dolfinx.fem.Function
- [] dolfinx.fem.bcs.DirichletBCMetaClass


## Contributing
Contributions are welcomed!

See https://finsberg.github.io/fenicsx-plotly/CONTRIBUTING.html for more info.
