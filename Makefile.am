SUBDIRS = src tests examples
ACLOCAL_AMFLAGS = -I config

ccd ccsd dgemm mm_sim mpi_latency mpi_bandwidth pmap:
	$(MAKE) -C ./examples $@

ta_test:
	$(MAKE) -C ./tests $@

doc:
	$(MAKE) -C ./doc