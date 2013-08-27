#!/bin/sh

# Generate configure script for autotools project

aclocal -I ./
autoconf
autoheader
automake --add-missing
