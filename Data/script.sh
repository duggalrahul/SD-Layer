#!/bin/bash

# This script iterates through each pic in each subfolder
# and 
# 1. appends the folder name to pic
# 2. copies it to the folder


for dr in */ ; do
    directory="$dr*"
    if [ $dr != "ZZZ/" ]
    then
        for p in $directory; do	
	    img_name=$(echo $p | rev | cut -d"/" -f1 | rev)
            new_name=$(echo "ZZZ/blast_"${dr::-1}"_"$img_name)
	    cp $p $new_name
        done
    fi
done


