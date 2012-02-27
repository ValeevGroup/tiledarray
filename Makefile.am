SUBDIRS = src tests examples
ACLOCAL_AMFLAGS = -I config

ccd ccsd dgemm mpi_latency mpi_bandwidth:
	$(MAKE) -C ./examples $@

ta-test:
	$(MAKE) -C ./tests

doc:
	$(MAKE) -C ./doc