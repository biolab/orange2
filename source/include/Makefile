all:	objdir $(OLD)/liborange_include.a

MODULENAME=INCLUDE
include ../makefile.defs
include makefile.deps

$(OLD)/liborange_include.a:	$(INCLUDE_OBJECTS)
	ar -r $(OLD)/liborange_include.a $(INCLUDE_OBJECTS)

clean:	cleantemp
	rm -f $(OLD)/liborange_include.a