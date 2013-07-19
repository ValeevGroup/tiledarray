CMAKE := cmake-2.8.11.1
URL := http://www.cmake.org/files/v2.8/$(CMAKE).tar.gz
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


