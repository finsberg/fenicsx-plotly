# We choose ubuntu 22.04 as our base docker image
FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

ARG REPO_BRANCH="main"
ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

# We set the working directory to install docker dependencies
WORKDIR /tmp/

# Install the Python-module
RUN git clone --branch ${REPO_BRANCH} --single-branch https://github.com/finsberg/fenicsx-plotly
RUN python3 -m pip install  --no-cache-dir ./fenicsx-plotly[all]


# We remove the contents of the temporary directory to minimize the size of the image
RUN rm -rf /tmp
