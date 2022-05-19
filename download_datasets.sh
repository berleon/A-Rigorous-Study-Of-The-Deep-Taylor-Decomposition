#! /usr/bin/env bash

source ./env.sh

echo "[TODO] download also CLEVR_XAI"
echo "Downloading datasets..."

if [ ! -e "$CLEVR" ]; then
    (cd "$DATA_DIR"
        wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
        unzip CLEVR_v1.0.zip
        rm CLEVR_v1.0.zip
    )
fi

if [ ! -e "$NLTP_DATA" ]; then
    python -m nltk.downloader -d "$NLTK_DATA" all
fi
