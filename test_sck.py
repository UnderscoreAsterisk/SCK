import os
from subprocess import call

FOLDER = "WebcamRelease/Panorama/test/image_gray"
for file in os.listdir(FOLDER):
    if file.endswith('.png'):
        print(f"Running from {file}")
        call(['python3', 'sck.py', f"{FOLDER}/{file}"])