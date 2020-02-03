#! /bin/sh

# Exit on error
set -ev

git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"

# only non-cron job deploys
RUN=1
if [ "$TRAVIS_EVENT_TYPE" = "cron" ]; then
  RUN=0
fi
if [ "$RUN" = "0" ]; then
  echo "Deployment skipped"
  exit 0
fi

# deploy from the build area
cd ${BUILD_PREFIX}/tiledarray

### deploy docs
# see https://gist.github.com/willprice/e07efd73fb7f13f917ea

# build docs
export VERBOSE=1
cmake --build . --target html
if [ ! -f "${BUILD_PREFIX}/mpqc/doc/dox/html/index.html" ]; then
  echo "Target html built successfully but did not produce index.html"
  exit 1
fi

# check out current docs + template
git clone --depth=1 https://github.com/ValeevGroup/tiledarray.git --branch gh-pages --single-branch tiledarray-docs-current
git clone --depth=1 https://github.com/ValeevGroup/tiledarray.git --branch gh-pages-template --single-branch tiledarray-docs-template
mkdir tiledarray-docs
cp -rp tiledarray-docs-current/* tiledarray-docs
rm -rf tiledarray-docs-current
cp -p tiledarray-docs-template/* tiledarray-docs
rm -rf tiledarray-docs-template
cd tiledarray-docs
# copy TA's README.md into index.md
cp ${TRAVIS_BUILD_DIR}/README.md index.md
# update dox
if [ -d dox-master ]; then
  rm -rf dox-master
fi
mv ${BUILD_PREFIX}/tiledarray/doc/dox/html dox-master
# make empty repo to ensure gh-pages contains no history
git init
git add *
git commit -a -q -m "rebuilt TA master docs via Travis build: $TRAVIS_BUILD_NUMBER"
git checkout -b gh-pages
git remote add origin https://${GH_TILEDARRAY_TOKEN}@github.com/ValeevGroup/tiledarray.git > /dev/null 2>&1
git push origin +gh-pages --force
cd ..
rm -rf tiledarray-docs
