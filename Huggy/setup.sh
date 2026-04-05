#!/bin/bash
set -e

# Download the Huggy executable
echo "Downloading Huggy executable..."
mkdir -p ./trained-envs-executables/linux
wget -q "https://github.com/huggingface/Huggy/raw/main/Huggy.zip" -O ./trained-envs-executables/linux/Huggy.zip
unzip -o -q ./trained-envs-executables/linux/Huggy.zip -d ./trained-envs-executables/linux/
chmod -R 755 ./trained-envs-executables/linux/Huggy

# Setup virtual environment with Python 3.10.12 using uv
echo "Setting up virtual environment with uv..."
uv venv --python 3.10.12 .venv

# Install dependencies using uv
echo "Installing dependencies..."
uv pip install -r requirements.txt

echo "Setup complete!"
