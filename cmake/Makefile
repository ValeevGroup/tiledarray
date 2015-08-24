#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  Makefile
#  Jul 19, 2013
#

CMAKE := cmake-3.3.1
URL := http://www.cmake.org/files/v3.3/$(CMAKE).tar.gz
PREFIX := $(PWD)

all: cmake

wget := $(shell wget 2>/dev/null)
curl := $(shell curl 2>/dev/null)

ifneq  ($(wget), )
download := wget $(URL) -O $(CMAKE).tar.gz
else
download := curl $(URL) -o $(CMAKE).tar.gz
endif


$(CMAKE).tar.gz:
	$(download)

.PHONY: cmake
cmake: $(CMAKE).tar.gz
	tar -xzf $(CMAKE).tar.gz
	cd ./$(CMAKE)
	cd $(CMAKE) && ./configure --prefix=$(PREFIX) # --system-curl
	cd $(CMAKE) && make install


