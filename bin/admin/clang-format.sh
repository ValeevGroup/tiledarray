#!/bin/sh

# append common locations of clang-format
unameOut="$(uname -s)"
case "${unameOut}" in
    Darwin*)    export PATH=$PATH:/opt/homebrew/bin;;
esac

path_to_clang_format=`which clang-format`
if [[ "X$path_to_clang_format" = "X" ]]; then
  echo "clang-format not found, PATH=$PATH"
  exit 1
fi

clang-format -i $*
