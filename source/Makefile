PYTHONINCLUDE = $(shell python -c "import sys; print sys.prefix+'/include/python'+sys.version[:3]")
PYTHONLIB = $(shell python -c "import sys; print sys.prefix+'/lib/python'+sys.version[:3]")

COMPILEOPTIONS=-fPIC -w -DLINUX -O3
LINKOPTIONS=-shared -lstdc++
export CPATH=orange:orange/ppp:orange/px:include:corn:statc:$(PYTHONINCLUDE)

vpath %.cpp orange:corn:statc:include
vpath %.hpp orange:corn:statc:include
vpath %.h orange:include:$(PYTHONINCLUDE)

obj/%.o :
	g++ $(COMPILEOPTIONS) -c $< -o $@


all	: dirs ../orange.so ../corn.so ../statc.so
            
include makefile.deps

dirs:
	-@mkdir -p obj > /dev/null

../orange.so: orange/ppp/stamp orange/px/stamp $(ORANGE_OBJECTS)
	gcc $(ORANGE_OBJECTS) $(LINKOPTIONS) -o ../orange.so

../corn.so:	$(CORN_OBJECTS)
	gcc $(CORN_OBJECTS) $(LINKOPTIONS) -o ../corn.so

../statc.so:	$(STATC_OBJECTS)
	gcc $(STATC_OBJECTS) $(LINKOPTIONS) -o ../statc.so

install:	all
	@echo Put \"export PYTHONPATH=$(subst /source,,$(shell pwd))\"
	@echo in your profile or create a file
	@echo $(PYTHONLIB)/site-packages/orange.pth
	@echo containg \"$(subst /source,,$(shell pwd))\"

clean:
	rm -f obj/* ../*.so orange/ppp/* orange/px/*
