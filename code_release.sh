#! /usr/bin/env bash

rm -rf /tmp/dtd_tmlr/*

rsync -av \
    --exclude=".*" \
    --exclude="site" \
    --exclude="htmlcov" \
    --exclude="code_release.sh" \
    --exclude="lrp_relations/notebooks/figures" \
    --exclude="CONTRIBUTING.md" \
    --exclude="docs" \
    --exclude="dist" \
    --exclude="relation_network/accuracy.png" \
    --exclude="syncing.sh" \
    --exclude="*_pycache_*" \
    . /tmp/dtd_tmlr


rsync -av \
    --exclude=".*" \
    --exclude="site" \
    --exclude="htmlcov" \
    --exclude="code_release.sh" \
    --exclude="lrp_relations/notebooks/figures" \
    --exclude="CONTRIBUTING.md" \
    --exclude="docs" \
    --exclude="dist" \
    --exclude="syncing.sh" \
    --exclude="syncing.sh" \
    --exclude="data" \
    --exclude="*_pycache_*" \
    /Users/leonsixt/Documents/projects/phd_flow/phd_flow \
    /tmp/dtd_tmlr/savethat

rsync -av \
    --exclude=".*" \
    --exclude="site" \
    --exclude="htmlcov" \
    --exclude="code_release.sh" \
    --exclude="lrp_relations/notebooks/figures" \
    --exclude="CONTRIBUTING.md" \
    --exclude="docs" \
    --exclude="dist" \
    --exclude="syncing.sh" \
    --exclude="syncing.sh" \
    --exclude="data" \
    --exclude="clevr-xai/images" \
    --exclude="*_pycache_*" \
    /Users/leonsixt/research/papers/lrp_fails_the_sanity_check/clevr-xai \
    /tmp/dtd_tmlr/clevr-xai

cd /tmp/dtd_tmlr
find . -type f
find . -type f | xargs sed -i.bak 's/leonsixt/XXXXXXX/g'
find . -type f | xargs sed -i.bak 's/berleon/XXXXXXX/g'
find . -type f | xargs sed -i.bak 's/github@leon-sixt.de/XXXXXXX/g'
find . -name '*.bak' | xargs rm

rm -rf .git
rm -rf .tox

cd ..
zip -r dtd_tmlr_code.zip dtd_tmlr
