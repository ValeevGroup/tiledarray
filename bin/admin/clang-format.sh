#!/bin/bash

# these are the versions of clang-format that are supported required
# should be ordered from oldest to newest to make sure the newest is picked
supported_clang_format_versions="16 17"
preferred_clang_format_version=""  # prefer most recent supported clang-format version
for v in $supported_clang_format_versions; do
  preferred_clang_format_version=$v
done

# append common locations of clang-format to PATH
unameOut="$(uname -s)"
case "${unameOut}" in
    Darwin*)
      extra_path=""
      # this prefers more recent versions
      for v in $supported_clang_format_versions; do
        extra_path=/opt/homebrew/opt/llvm@$v/bin:/opt/homebrew/opt/clang-format@$v/bin:$extra_path
      done
      # prepend paths
      export PATH=$extra_path:$PATH:/opt/homebrew/bin
    ;;
esac

path_to_clang_format=`which clang-format`
have_supported_clang_format_version=0
if [[ "X$path_to_clang_format" != "X" ]]; then

  # check clang-format version
  clang_format_version=`clang-format --version | sed 's/.* version //' | awk -F'[.]' '{print $1}'`

  #echo "supported_clang_format_versions=\"$supported_clang_format_versions\" clang_format_version=$clang_format_version"

  # if found clang-format, but wrong version, check if docker is available
  for v in $supported_clang_format_versions; do
    if [[ $clang_format_version -eq $v ]]; then
      have_supported_clang_format_version=1
      break
    fi
  done
fi

if [[ $have_supported_clang_format_version -eq 0 ]]; then
  echo "WARNING: found clang-format with unsupported version $clang_format_version (supported versions: $supported_clang_format_versions)"

  # look for docker
  path_to_docker=`which docker`
  if [[ "X$path_to_docker" = "X" ]]; then
    echo "ERROR: docker is not found either, PATH=$PATH, install one of supported clang-format versions (any of these: $supported_clang_format_versions) or install docker"
    exit 1
  fi

  # if docker up?
  docker info >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "ERROR: docker is found but not running, start it"
    exit 1
  fi

  # use docker to run clang-format
  mount_path=$(readlink -f "$HOME")

  # convert file names in the arguments to relative paths
  args=""
  for i in "$@"; do
    # skip options
    if [[ "$i" == -* ]]; then
      args="$args $i"
      continue
    fi
    abs_file_path=$(readlink -f "$i")
    if [[ "X$abs_file_path" = "X" ]]; then
      echo "ERROR: given file $i is not found"
      exit 1
    fi

    dir=$(dirname $abs_file_path)
    file_path_relative_to_project_root=$(basename $abs_file_path)
    while [[ "$dir" != "$mount_path" && "$dir" != "/" ]]; do
      file_path_relative_to_project_root="$(basename $dir)/$file_path_relative_to_project_root"
      dir=$(dirname $dir)
      #echo "dir=$dir file_path_relative_to_project_root=$file_path_relative_to_project_root"
    done
    if [[ "$dir" == "/" ]]; then
      echo "ERROR: given file $i (absolute path $abs_file_path) is not under \$HOME=$mount_path, cannot use docker-based clang-format in this case"
      exit 1
    fi
    args="$args /hostHOME/$file_path_relative_to_project_root"
  done
  docker run --platform linux/x86_64 -v $mount_path:/hostHOME xianpengshen/clang-tools:$preferred_clang_format_version clang-format $args
else
  #echo "found $path_to_clang_format with required version $clang_format_version"
  clang-format $*
fi
