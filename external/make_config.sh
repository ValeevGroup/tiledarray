#!/bin/sh

# Generate configure script for autotools project

aclocal -I ./config
autoconf
autoheader
automake --add-missing
