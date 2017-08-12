#!/bin/bash
#
# At least one of the coco validation images has a JPEG suffix
# but is in reality a PNG file, which will cause the vl_imredjpeg
# function to hang.  This script aims to ensure filetype consistency
#
# Copyright (C) 2017 Samuel Albanie 
# Licensed under The MIT License [see LICENSE.md for details]

COCO_VAL_DIR="${HOME}/data/datasets/mscoco/images/val2014"
NUM_IMS=`ls -l $COCO_VAL_DIR | wc | awk '{ print $1 }'`
echo "Checking image types for ${NUM_IMS} images.  This may take a while..."

for f in ${COCO_VAL_DIR}/*.jpg ; do
    TYPE=`identify $f | awk '{ print $2 }'` ; 
    if [ "$TYPE" == "PNG" ] 
    then 
        echo "$f is not a JPEG image, converting ..."
        stem=${f%.jpg} # rename to allow simple conversion
        tmp=${stem}.png
        mv $f $tmp
        convert $tmp $f
    fi
done
