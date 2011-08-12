ORANGEPLOT_BUILD_DIR=build

all:
	mkdir -p $(ORANGEPLOT_BUILD_DIR)
	cd $(ORANGEPLOT_BUILD_DIR); cmake -DCMAKE_BUILD_TYPE=Release ..
	if ! $(MAKE) $@ -C $(ORANGEPLOT_BUILD_DIR); then exit 1; fi;
	cp $(ORANGEPLOT_BUILD_DIR)/orangeplot.so $(OLD)

cleantemp:
	rm -rf $(ORANGEPLOT_BUILD_DIR)

clean: cleantemp
	rm -f $(OLD)/orangeplot.so