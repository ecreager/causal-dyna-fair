#!/bin/bash
# Sets up a virtual environment with the right package dependencies
# 
# NOTE: Source from root repo like source ./bin/setup.sh

set -e
mkdir -p ~/venv
python3 -m venv ~/venv/causal-dyna-fair
source ~/venv/causal-dyna-fair/bin/activate
pip install -r requirements.txt
git clone git@github.com:zykls/whynot.git
cd whynot
pip install .
cd ../

# Later on, this virtual env can be entered using
# source ~/venv/causal-dyna-fair/bin/activate