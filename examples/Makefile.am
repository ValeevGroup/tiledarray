# examples

SUBDIRS = cc ta_dgemm mpi_tests

ccd ccsd:
	$(MAKE) -C ./cc $@

dgemm:
	$(MAKE) -C ./ta_dgemm $@

mpi_latency mpi_bandwidth:
	$(MAKE) -C ./mpi_tests $@
	
mm_sim:
	$(MAKE) -C ./mmsim $@