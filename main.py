#!/bin/python3
from PIL import Image
import os
import sys

quality=100
compress_level=0

lossless=True

# Starting formats: webp, png, jpg

lossless_formats= (
    ".webp",
    ".png"
)
lossy_formats=(
    ".webp",
    ".jpg",
    ".png"
)

format_modes={
    ".webp":["P","RGB","RGBA"],
    ".jpg":["RGB"],
    ".png":["P","RGB","RGBA"],
}



def convert_getsize(source_img,to_ext,conv=None, losl=False, opt=False, qua=70):
    image_to_analyze = Image.open(source_img)
    source_name = os.path.splitext(source_img)[0]
    source_extention = os.path.splitext(source_img)[1]
    destination_temp_file = source_name + "_" + source_extention[1:] + "-to-"+ to_ext[1:]+ ("_"+conv if conv!=None else "") +"_temp"+ to_ext

    print(image_to_analyze.mode)

    # Save the file in the format
    if conv != None:
        image_to_analyze.convert(conv).save(destination_temp_file, quality=qua, optimize=opt, compress_level=0, lossless=losl)
    else:
        image_to_analyze.save(destination_temp_file, quality=qua, optimize=opt, compress_level=0, lossless=losl)

    # determin the size of the saved file
    file_size = int(os.path.getsize(destination_temp_file))

    print("Image size from " + source_extention.upper() + " to " + to_ext.upper() + (" with " + conv if conv!=None else "") + ": " + str(file_size) + " bytes")
    return {
        "path": destination_temp_file,
        "size": file_size
    }

def main():
    source_img = sys.argv[1]

    if not os.path.exists(source_img):
        print("ERROR: This file does not exists: " + source_img)
        sys.exit(1)


    original_image_size = os.path.getsize(source_img)
    print("Original file: " + source_img)
    print("Original size: " + str(original_image_size) + " bytes")

    all_result=[]

    for format in lossy_formats:
        result=convert_getsize(source_img, format, conv=format_modes[format][0],qua=quality,losl=lossless)
        all_result+=result
        # os.remove(result["path"])

    # sorted_result = sorted(all_result, key=lambda dic: dic['size'])
    #
    # print("The smallest format to store this picture:" + str(sorted_result[0]["path"][-4:]))



if __name__ == '__main__':
    main()
