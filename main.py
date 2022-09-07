#!/bin/python3
from PIL import Image
import os
import sys





def convert_getsize(to_ext):
    imageToAnalyze = Image.open(source_file)
    print(imageToAnalyze.mode)
    if to_ext == ".jpg":
        imageToAnalyze.convert("RGB").save(destination_temp_file + to_ext)
    else:
        imageToAnalyze.save(destination_temp_file + to_ext)
    file_size = os.path.getsize(destination_temp_file + to_ext)
    print("Image size from " + source_extention.upper() + " to " + to_ext.upper() + ": " + str(file_size) + " bytes")
    os.remove(destination_temp_file + to_ext)

def main():
    source_file = sys.argv[1]

    if not os.path.exists(source_file):
        print("ERROR: This file does not exists: " + source_file)
        sys.exit(1)

    source_extention = os.path.splitext(source_file)[1]
    destination_temp_file = "theater_temp"


    original_file_size = os.path.getsize(source_file)
    print("Original file: " + source_file)
    print("Original size: " + str(original_file_size) + " bytes")
    # Need to find the conversion quality
    convert_getsize(".jpg")
    convert_getsize(".png")
    convert_getsize(".tiff")
    convert_getsize(".gif")
    convert_getsize(".bmp")

if __name__ == '__main__':
    main()
