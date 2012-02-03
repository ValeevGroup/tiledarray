SUBDIRS = src Tests examples
ACLOCAL_AMFLAGS = -I config

ccd ccsd dgemm:
	$(MAKE) -C examples $@

ta-test:
	$(MAKE) -C Tests

doc:
	$(MAKE) -C doc