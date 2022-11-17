#!/bin/python3
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
import sys
import getopt
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

quality=80
compress_level=6
optimize=False
lossless=False
output_file_name=""
debug=False
# Starting formats: webp, png, jpg

all_formats={
    ".webp":["P","RGB","RGBA"],
    ".png":["P","RGB","RGBA"],
    ".jpg":["RGB"]
}

lossless_formats={
    ".webp":["P","RGB","RGBA"],
    ".png":["P","RGB","RGBA"]
}

lossy_formats={
    ".webp":["P","RGB","RGBA"],
    ".jpg":["RGB"]
}

transparents_formats={
    ".webp":["P","RGBA"],
    ".png":["P","RGBA"],
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

def is_same(source_a_path, source_b_path):
    img_a = cv2.imread(source_a_path)
    img_b = cv2.imread(source_b_path)
    try:
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)
    except ValueError as e:
        img_b = cv2.rotate(img_b, cv2.ROTATE_90_CLOCKWISE)
        err = np.sum((img_a.astype("float") - img_b.astype("float")) ** 2)

    err /= float(img_a.shape[0] * img_b.shape[1])

    # same if 0
    if debug:
        print("Image different:" + str(err))

    # same if 1
    if len(img_a.shape)==2:
        diff=ssim(img_a, img_b)
    else:
        diff=ssim(img_a, img_b, channel_axis=2)

    if debug:
        print("Image different: {0}".format(diff))
    return diff

def convert_getsize(image_to_analyze, src_img_path, to_ext, conv, losl, opt, qua):
    source_name = os.path.splitext(src_img_path)[0]
    source_extention = os.path.splitext(src_img_path)[1]
    if output_file_name=="":
        destination_temp_file = source_name + "_" + source_extention[1:] + "-to-"+ to_ext[1:]+ ("_"+conv if conv!=None else "") +"_temp"+ to_ext
    else:
        destination_temp_file = output_file_name

    # Save the file in the format
    if conv != None:
        image_to_analyze.convert(conv).save(destination_temp_file, quality=qua, optimize=opt, compress_level=0, lossless=losl)
    else:
        image_to_analyze.save(destination_temp_file, quality=qua, optimize=opt, compress_level=0, lossless=losl)

    diff=is_same(src_img_path, destination_temp_file)

    # determin the size of the saved file
    file_size = int(os.path.getsize(destination_temp_file))

    if debug:
        print("Image size from " + source_extention.upper() + " to " + to_ext.upper() + (" with " + conv if conv!=None else "") + ": " + str(file_size) + " bytes")
        print("-------------------")
    return {
        "path": destination_temp_file,
        "ext": to_ext,
        "mode": conv,
        "size": file_size,
        "diff": diff
    }

def count_faces(src_img):
    cascade_path = "./haarcascade.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    gray_image = cv2.imread(src_img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    # if debug:
    print("Found {0} faces!".format(len(faces)))
    return faces

def is_document(source_img):
    img = cv2.imread(source_img, cv2.IMREAD_COLOR)
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
    if img.info.get("transparency", None) is not None:
        if debug:
            print("Found transparency flag in image metadata.")
        return True
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                if debug:
                    print("Found transparency color in Palette.")
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            if debug:
                print("Found transparency by RGBA color mode.")
            return True

    return False

def main():
    global compress_level, quality, lossless, output_file_name, optimize, debug
    if len(sys.argv) < 2:
        print ("Usage: ./main.py [-q <quality_level> -c <compress_level> -l -f <output_file_name>] <image_to_analyze>")
        print ("-q <quality_level> is a number between 0 and 100, 0 is worst quality best compression, 100 is almost lossless, best quality. Default: 80")
        print ("-c <compress_level> is a number between 0 and 9. 0 is lossless, 9 is best compression. Deafult: 6")
        print ("-l if present it will use only lossless file formats or formats which is capable to lossless. Will ignore -q and -c flags. Default: false")
        print ("-o if present it will optimize the compression. Default: false")
        print ("-f <output_file_name> output file name")
        print ("-v if present will show debug informations")
        exit (1)
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

    source_img = args[0]

    if not os.path.exists(source_img):
        print("ERROR: This file does not exists: " + source_img)
        sys.exit(1)


    original_image_size = os.path.getsize(source_img)
    print("Original file: " + source_img)
    print("Original size: " + str(byte_to_human(original_image_size)) + " bytes")

    all_result=[]
    formats_to_use=all_formats

    image_to_analyze = Image.open(source_img)
    if debug:
        print("Src image mode: " + image_to_analyze.mode)

# TODO: or is_document(source_img)
    is_document(source_img)
    if lossless:
        formats_to_use=lossless_formats
    elif len(count_faces(source_img)) > 0:
        formats_to_use=lossy_formats

    if has_transparency(image_to_analyze):
        formats_to_use=transparents_formats

    for ext, modes in formats_to_use.items():
        for mode in modes:
            result=convert_getsize(image_to_analyze, source_img, ext, mode, lossless, optimize, quality)
            if lossless and result["diff"] == 1:
                all_result.append(result)
            elif not lossless:
                all_result.append(result)
            os.remove(result["path"])

    final = sorted(all_result, key=lambda dic: dic['size'])[0]

    if final["size"] > original_image_size:
        print("I couldn't find a better format for this image with these options.")
        sys.exit(0)

    original_path = os.path.splitext(source_img)
    if output_file_name == "":
        output_file_name = "new_" + original_path[-2] + final["ext"]
    else:
        output_file_name += final["ext"]

        convert_getsize(image_to_analyze, source_img, final["ext"], final["mode"], lossless, optimize, quality)

    print("The smallest format to store this picture: " + final["ext"])
    print("New image is {0}% the same.".format(round((final["diff"]*100),2)))
    print("New image size is " + str(round(((final["size"]/original_image_size)*100),2))+"% of the original size.")
    print("Converted to " + final["ext"].upper() + " with " + str(final["mode"]) + " mode: " + str(byte_to_human(final["size"])) + " bytes")


if __name__ == '__main__':
    main()
