ORANGEPLOT_BUILD_DIR=build
ifndef OLD
  OLD=..
endif


all:
	mkdir -p $(ORANGEPLOT_BUILD_DIR)
	cd $(ORANGEPLOT_BUILD_DIR); cmake -DCMAKE_BUILD_TYPE=Release -DORANGE_LIB_DIR=$(abspath $(OLD)) ..
	if ! $(MAKE) $@ -C $(ORANGEPLOT_BUILD_DIR); then exit 1; fi;

cleantemp:
	rm -rf $(ORANGEPLOT_BUILD_DIR)

clean: cleantemp
	rm -f $(OLD)/orangeqt.so