#!/bin/python3
from PIL import Image
import os
import sys
import io
import getopt
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


# Default settings
quality=80
compress_level=6
optimize=False
lossless=False
output_file_name=""
debug=False
# Starting formats: webp, png, jpg

all_formats={
    "webp":["P","RGB","RGBA"],
    "png":["P","RGB","RGBA"],
    "jpeg":["RGB"]
}

lossless_formats={
    "webp":["P","RGB","RGBA"],
    "png":["P","RGB","RGBA"]
}

lossy_formats={
    "webp":["P","RGB","RGBA"],
    "jpeg":["RGB"]
}

transparents_formats={
    "webp":["P","RGBA"],
    "png":["P","RGBA"],
}

# src: https://stackoverflow.com/questions/56243676/python-human-readable-to-byte-conversion
def byte_to_human(byte):
    if byte == 0:
        raise ValueError("Size is not valid.")
    byte = int(byte)
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    index = int(math.floor(math.log(byte, 1024)))
    power = math.pow(1024, index)
    size = round(byte / power, 2)
    return "{} {}".format(size, size_name[index])

def is_same(img_a_input, img_b_input):
    img_a = np.array(img_a_input)
    img_b = np.array(img_b_input)
    # Calculate the MSE number
    try:
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
    except ValueError as e:
        # Sometimes the convertion rotates the image, usualy images in portrait mode
        img_b = cv2.rotate(img_b, cv2.ROTATE_90_CLOCKWISE)
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)

    # This gives a number between 0 and 195075
    err /= float(img_a.shape[0] * img_b.shape[1])

    # map the result between 1 and 0. same if 1, totaly different if 0
    res = np.interp(err,[0,195075],[1,0])

    # same if 1
    if debug:
        print("Image different:" + str(res))
    return res

def convert_getsize(image_to_analyze, src_img_path, to_ext, to_mode, losl, opt, qua):
    # Convert the image to the format in memory
    new_image_obj = io.BytesIO()
    image_to_analyze.convert(mode=to_mode,palette=0).save(new_image_obj, format=to_ext, quality=qua, optimize=opt, compress_level=0, lossless=losl)

    if debug:
        print("Before",image_to_analyze.mode)
        print("After",Image.open(new_image_obj, formats=[to_ext.upper()]).mode)

    # Calculate difference
    # Convert images to RGB so we can compare P and RGBA images as well pixel by pixel
    diff = is_same(image_to_analyze.convert("RGB"), (Image.open(new_image_obj).convert("RGB")))

    # determin the size of the image file
    file_size = new_image_obj.tell()

    if debug:
        source_extention = os.path.splitext(src_img_path)[1]
        print("Image size from " + source_extention.upper() + " to " + to_ext.upper() + (" with " + to_mode if to_mode!=None else "") + ": " + str(file_size) + " bytes")
        print("-------------------")
    return {
        "ext": to_ext,
        "mode": to_mode,
        "size": file_size,
        "diff": diff
    }
# Face detection
def count_faces(src_img):
    # Loading the cascade
    cascade_path = "./haarcascade.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    # Reading in the image
    gray_image = cv2.imread(src_img, cv2.COLOR_BGR2GRAY)
    # Running the cascase on the image
    faces = cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    if debug:
        print("Found {0} faces!".format(len(faces)))
    return faces

# This does not run
def is_document(input_file_path):
    img = cv2.imread(input_file_path, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    print("sum,mean,max,min")
# blue channel
    blue_matrix = np.matrix(b)
    print("blue:", blue_matrix.sum(), blue_matrix.mean(), blue_matrix.max(), blue_matrix.min())
# green channel
    green_matrix = np.matrix(g)
    print("green:", green_matrix.sum(), green_matrix.mean(), green_matrix.max(), green_matrix.min())
# red channel
    red_matrix = np.matrix(r)
    print("red:", red_matrix.sum(), red_matrix.mean(), red_matrix.max(), red_matrix.min())


    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([img_gray], [0], None, [256], [0, 256])

# gray channel
    gray_matrix = np.matrix(img_gray)
    print("gray:", gray_matrix.sum(), gray_matrix.mean(), gray_matrix.max(), gray_matrix.min())

# source: https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
def has_transparency(img):
    # search metadata
    if img.info.get("transparency", None) is not None:
        if debug:
            print("Found transparency flag in image metadata.")
        return True
    # If image is Palette then search the palette
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                if debug:
                    print("Found transparency color in Palette.")
                return True
    # RGBA mode
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            if debug:
                print("Found transparency by RGBA color mode.")
            return True

    return False

def main():
    global compress_level, quality, lossless, output_file_name, optimize, debug
    # Help message
    if len(sys.argv) < 2:
        print ("Usage: ./main.py [-q <quality_level> -c <compress_level> -l -f <output_file_name>] <image_to_analyze>")
        print ("-q <quality_level> is a number between 0 and 100, 0 is worst quality best compression, 100 is almost lossless, best quality. Default: 80")
        print ("-c <compress_level> is a number between 0 and 9. 0 is lossless, 9 is best compression. Deafult: 6")
        print ("-l if present it will use only lossless file formats or formats which is capable to lossless. Will ignore -q and -c flags. Default: false")
        print ("-o if present it will optimize the compression. Default: false")
        print ("-f <output_file_name> output file name")
        print ("-v if present will show debug informations")
        exit (1)
    # Reading in the flags and arguments
    opts, args = getopt.getopt(sys.argv[1:], 'q:c:f:lov')
    for k, v in opts:
        if k == '-q' and lossless==False:
            if int(v) > 100:
                quality=100
            elif int(v)<0:
                quality=0
            else:
                quality=int(v)
        if k == '-c':
            if int(v) > 9:
                compress_level=9
            elif int(v)<0:
                compress_level=0
            else:
                compress_level=int(v)
        if k == '-l':
            lossless=True
            quality=100
        if k == '-o':
            optimize=True
        if k == '-f':
            output_file_name=v
        if k == '-v':
            debug=True

    input_file_path = args[0]

    # Checkif file exist
    if not os.path.exists(input_file_path):
        print("ERROR: This file does not exists: " + input_file_path)
        sys.exit(1)

    # Reading the image and some data about
    image_to_analyze = Image.open(input_file_path)
    input_file_size = os.path.getsize(input_file_path)

    print("Original file: " + input_file_path)
    print("Original size: " + str(byte_to_human(input_file_size)) + " bytes")
    if debug:
        print("Image dimension: " + str(image_to_analyze.size[0]) + "x" + str(image_to_analyze.size[1]))
        print("Src image mode: " + image_to_analyze.mode)

    all_result=[]
    formats_to_use=all_formats

# TODO: or is_document(input_file_path)

    # Setting what formats to use based on flags, or other information about the image
    # is_document(input_file_path)
    if lossless:
        # If lossless flag is given use lossless formats
        formats_to_use=lossless_formats
    elif len(count_faces(input_file_path)) > 0:
        # Detecting a face prefer lossy formats but only if lossless flag is not given
        formats_to_use=lossy_formats


    if has_transparency(image_to_analyze):
        # Transparency do not affect lossless because trasparent modes are lossless
        formats_to_use=transparents_formats

    # Convert the images to all posibile conbinations of formats and modes
    for ext, modes in formats_to_use.items():
        for mode in modes:
            result=convert_getsize(image_to_analyze, input_file_path, ext, mode, lossless, optimize, quality)
            # Decide if we want to use the result at all in the end or not, diff == 1 meens there is no loss in the conversion
            if lossless and result["diff"] == 1:
                all_result.append(result)
            elif not lossless:
                all_result.append(result)

    # Sort the final list by size
    final = sorted(all_result, key=lambda dic: dic['size'])[0]

    # Check if the output file is larger than the original do nothing.
    if final["size"] > input_file_size:
        print("I couldn't find a better format for this image with these options.")
        sys.exit(0)

    # The final image file name, what if they give an extension with the filename? TODO: only use that extension
    original_path = os.path.splitext(input_file_path)
    if output_file_name == "":
        output_file_name = "new_" + original_path[-2] + "." + final["ext"]
    else:
        output_file_name += "." + final["ext"]

    # Convert and save the new image
    image_to_analyze = image_to_analyze.convert(final["mode"])
    image_to_analyze.save(output_file_name, quality=quality, optimize=optimize, compress_level=0, lossless=lossless)

    # Result msg
    true_size = int(os.path.getsize(output_file_name))

    print("------------------------------------------------------------")
    print("The smallest format to store this picture: " + final["ext"])
    print("New image is {0}% the same.".format(round((final["diff"]*100),2)))
    print("New image size is " + str(byte_to_human(true_size)) + " byte.")
    print("New image size is " + str(round(((true_size/input_file_size)*100),2))+"% of the original size.")
    print("Converted to " + final["ext"].upper() + " with " + str(final["mode"]) + " mode: " + str(byte_to_human(final["size"])))


if __name__ == '__main__':
    main()
