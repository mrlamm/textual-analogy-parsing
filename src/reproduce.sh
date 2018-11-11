#!/bin/bash
# This script reproduces results using the existing trained models.

# First ensure we are in a virtual environment.
if [ ! -e .env ]; then
  echo "Creating virtual environment in .env";
  virtualenv -p python3 .env
  source .env/bin/activate
  pip install -r ./requirements.txt;
fi;


