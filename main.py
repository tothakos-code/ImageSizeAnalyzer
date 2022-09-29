#!/bin/python3
from PIL import Image
import os
import sys
import getopt

quality=80
compress_level=6
optimize=False
lossless=False
output_file_name=""
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



def convert_getsize(source_img,to_ext, conv, losl, opt, qua):
    image_to_analyze = Image.open(source_img)
    source_name = os.path.splitext(source_img)[0]
    source_extention = os.path.splitext(source_img)[1]
    if output_file_name=="":
        destination_temp_file = source_name + "_" + source_extention[1:] + "-to-"+ to_ext[1:]+ ("_"+conv if conv!=None else "") +"_temp"+ to_ext
    else:
        destination_temp_file = output_file_name

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
    global compress_level, quality, lossless, output_file_name, optimize
    if len(sys.argv) < 2:
        print ("Usage: ./main.py [-q <quality_level> -c <compress_level> -l -f <output_file_name>] <image_to_analyze>")
        print ("-q <quality_level> is a number between 0 and 100, 0 is worst quality best compression, 100 is almost lossless, best quality. Default: 80")
        print ("-c <compress_level> is a number between 0 and 9. 0 is lossless, 9 is best compression. Deafult: 6")
        print ("-l if present it will use only lossless file formats or formats which is capable to lossless. Default: false")
        print ("-o if present it will optimize the compression. Default: false")
        print ("-f <output_file_name> output file name")
        exit (1)
    opts, args = getopt.getopt(sys.argv[1:], 'q:c:f:lo')
    for k, v in opts:
        if k == '-q':
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
        if k == '-o':
            optimize=True
        if k == '-f':
            output_file_name=v

    source_img = args[0]

    if not os.path.exists(source_img):
        print("ERROR: This file does not exists: " + source_img)
        sys.exit(1)


    original_image_size = os.path.getsize(source_img)
    print("Original file: " + source_img)
    print("Original size: " + str(original_image_size) + " bytes")

    all_result=[]

    for format in lossy_formats:
        for mode in format_modes[format]:
            result=convert_getsize(source_img, format, mode, lossless, optimize, quality)
            all_result.append(result)
        # os.remove(result["path"])

    sorted_result = sorted(all_result, key=lambda dic: dic['size'])
    #
    print("The smallest format to store this picture: " + str(sorted_result[0]["path"][-4:]))
    print("New image size is " + str(round(((sorted_result[0]["size"]/original_image_size)*100),2))+"% of the original size.")



if __name__ == '__main__':
    main()
