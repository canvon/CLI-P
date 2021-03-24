#!/bin/bash

set -e

# Use python virtualenv from setup.sh
[ -z "${VIRTUAL_ENV-}" ] && . env/bin/activate

# Install GUI dependencies via pip
pip3 install -r requirements-gui.txt

echo "All done."
