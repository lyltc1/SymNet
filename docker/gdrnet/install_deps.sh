set -x
sudo apt-get install libjpeg-dev zlib1g-dev -y
sudo apt-get install libopenexr-dev -y
sudo apt-get install openexr -y
sudo apt-get install libglfw3-dev libglfw3 -y
sudo apt-get install libassimp-dev -y

pip install -r /tmp/dockerfile_scripts/gdrnet/requirements.txt

# pip uninstall pillow
# CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

