ORANGEPLOT_BUILD_DIR=build
ifndef OLD
  OLD=..
endif

OS = $(shell uname)

all:
	mkdir -p $(ORANGEPLOT_BUILD_DIR)
	cd $(ORANGEPLOT_BUILD_DIR); cmake -DCMAKE_BUILD_TYPE=Debug -DORANGE_LIB_DIR=$(abspath $(OLD)) $(EXTRA_ORANGEQT_CMAKE_ARGS) ..
	if ! $(MAKE) $@ -C $(ORANGEPLOT_BUILD_DIR); then exit 1; fi;
ifeq ($(OS), Darwin)
	install_name_tool -id $(DESTDIR)/orangeqt.so $(OLD)/orangeqt.so
endif
	

cleantemp:
	rm -rf $(ORANGEPLOT_BUILD_DIR)

clean: cleantemp
	rm -f $(OLD)/orangeqt.so
