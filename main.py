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
DEFAULT_QUALITY=80
DEFAULT_COMPRESS_LEVEL=6
DEFAULT_OPTIMIZE=False
DEFAULT_LOSSLESS=False
output_file_name=""
DEFAULT_DEBUG=False
# currently supported formats: webp, png, jpg

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

# it is in script mode or module mode
SCRIPT=False

# src: https://stackoverflow.com/questions/56243676/python-human-readable-to-byte-conversion
def byte_to_human(byte):
    """Converts from byte to a human readable size format"""
    if byte == 0:
        raise ValueError("Size is not valid.")
    byte = int(byte)
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    index = int(math.floor(math.log(byte, 1024)))
    power = math.pow(1024, index)
    size = round(byte / power, 2)
    return "{} {}".format(size, size_name[index])

def is_same(img_a_input, img_b_input, debug=DEFAULT_DEBUG):
    """Recives two input image, and returns a 0-1 number how similiar the two picture is. The function is using the MSE comparative method. The return value if 1 it meand the two input picture are identical."""
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

def convert_getsize(image_to_analyze, src_img_path, to_ext, to_mode, lossless=DEFAULT_LOSSLESS, optimize=DEFAULT_OPTIMIZE, quality=DEFAULT_QUALITY,debug=DEFAULT_DEBUG):
    """Converting the Image and collecting info about it. The conversion is in memory. Doeas not write the result file"""
    # Convert the image to the format in memory
    new_image_obj = io.BytesIO()
    image_to_analyze.convert(mode=to_mode,palette=0).save(new_image_obj, format=to_ext, quality=quality, optimize=optimize, compress_level=0, lossless=lossless)

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
def count_faces(src_img,debug=DEFAULT_DEBUG):
    """Runs a face detection on the image and returns the number of faces it found."""
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
    """A try to detect if the image contains a picture of a document. But did not found a working solution. This function is not in use."""
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
def has_transparency(img,debug=DEFAULT_DEBUG):
    """Scan the if there is transparency in the image."""
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
        if k == '-q':
            if int(v) > 100:
                quality=100
            elif int(v)<0:
                quality=0
            else:
                quality=int(v)
        else:
            quality=DEFAULT_QUALITY
        if k == '-c':
            if int(v) > 9:
                compress_level=9
            elif int(v)<0:
                compress_level=0
            else:
                compress_level=int(v)
        else:
            compress_level=DEFAULT_COMPRESS_LEVEL
        if k == '-l':
            lossless=True
            quality=100
        else:
            lossless=DEFAULT_LOSSLESS
        if k == '-o':
            optimize=True
        else:
            lossless=DEFAULT_OPTIMIZE
        if k == '-f':
            output_file_name=v
        if k == '-v':
            debug=True
        else:
            debug=DEFAULT_DEBUG

    input_file_path = args[0]
    result = scan(input_file_path, quality=quality, compress_level=compress_level, lossless=lossless, optimize=optimize, debug=debug)

    print("Original file: " + result["original_filepath"])
    print("Original size: " + str(byte_to_human(result["original_size"])) + " bytes")
    if debug:
        print("Image dimension: " + result["dimensions"])
        print("Src image mode: " + result["original_mode"])
    print("------------------------------------------------------------")
    print("The smallest format to store this picture: " + result["result"]["ext"])
    print("New image is {0}% identical to the original.".format(result["output_difference_precentage"]))
    print("New image size is " + str(byte_to_human(result["output_true_size"])) + " byte.")
    print("New image size is " + str(result["output_smaller_precentage"]) + "% of the original size.")
    print("Converted to " + result["result"]["ext"].upper() + " with " + str(result["result"]["mode"]) + " mode: " + str(byte_to_human(result["result"]["size"])))

def scan(image_to_analyze, quality=80, compress_level=6, lossless=False, optimize=False, debug=False):
    # Checking the inputs
    if lossless==False:
        if quality > 100:
            quality = 100
        elif quality < 0:
            quality = 0
    else:
        quality=100


    if compress_level > 9:
        compress_level=9
    elif compress_level<0:
        compress_level=0

    input_file_path = image_to_analyze

    # Checkif file exist
    if not os.path.exists(input_file_path):
        raise ValueError("ERROR: This file does not exists: " + input_file_path)

    # Reading the image and some data about
    image_to_analyze = Image.open(input_file_path)
    input_file_size = os.path.getsize(input_file_path)



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
            result=convert_getsize(image_to_analyze, input_file_path, ext, mode, quality=quality, compress_level=compress_level, lossless=lossless, optimize=optimize, debug=debug)
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
        return {
            "original_filepath": input_file_path,
            "original_size": input_file_size,
            "dimensions": str(image_to_analyze.size[0]) + "x" + str(image_to_analyze.size[1]),
            "original_mode": image_to_analyze.mode,
            "formats_used": formats_to_use,
            "result": {

            }
        }

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

    return {
        "original_filepath": input_file_path,
        "original_size": input_file_size,
        "dimensions": str(image_to_analyze.size[0]) + "x" + str(image_to_analyze.size[1]),
        "original_mode": image_to_analyze.mode,
        "formats_used": formats_to_use,
        "output_file_name": output_file_name,
        "output_true_size": true_size,
        "output_difference_precentage": round((final["diff"]*100),2),
        "output_smaller_precentage": round(((true_size/input_file_size)*100),2),
        "result": final
    }


if __name__ == '__main__':
    SCRIPT=True
    main()
