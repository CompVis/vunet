#!/usr/bin/env bash

if [ $# -lt 2 ]
then
    echo "Usage: $0 <dataset> <dest>"
    echo "where <dataset> is one of coco, deepfashion, market"
    echo "and <dest> is the destination folder"
    exit 1
fi

data=$1
dest=$2
mkdir -p "$dest"
src="http://129.206.117.181:8080/${data}.tar.gz"
wget -P "$dest" -c $src

tar --skip-old-files -xzf "${dest}/${data}.tar.gz" -C "${dest}"

srcdir=$(dirname $(realpath $0))
mkdir -p "${srcdir}/data"
ln -s "${dest}/${data}" "${srcdir}/data/"

