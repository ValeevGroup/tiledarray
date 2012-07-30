# examples

SUBDIRS = cc ta_dgemm mpi_tests pmap_test

ccd ccsd:
	$(MAKE) -C ./cc $@

dgemm:
	$(MAKE) -C ./ta_dgemm $@

mpi_latency mpi_bandwidth:
	$(MAKE) -C ./mpi_tests $@

pmap:
	$(MAKE) -C ./pmap_test $@