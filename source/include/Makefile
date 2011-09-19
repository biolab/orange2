all:	objdir $(OLD)/liborange_include.a

MODULENAME=INCLUDE
include ../makefile.defs
-include makefile.deps

# Redifine the build rule to use STATICCOMPILEOPTIONS
obj/%.o : %.cpp
	$(COMPILER) $(STATICCOMPILEOPTIONS) -c $< -o $@
	
$(OLD)/liborange_include.a:	$(INCLUDE_OBJECTS)
	ar -r $(OLD)/liborange_include.a $(INCLUDE_OBJECTS)
# On Mac this makes invalid universal archive (archive of universal object files),
# so we fix it (make an universal file containing multiple archives) - we could also
# use other tools than ar
ifeq ($(OS), Darwin)
	ranlib $(OLD)/liborange_include.a
endif

clean:	cleantemp
	rm -f $(OLD)/liborange_include.a
