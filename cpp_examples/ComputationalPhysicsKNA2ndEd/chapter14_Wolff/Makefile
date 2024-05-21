# ########################   Makefile.1   ############################
# $@ target, $< first dependency, $^ all dependencies, 
# $* "stem" of target filename e.g. %.f: %.F will five foo for foo.f

OBJS   = main.o init.o wolff.o met.o measure.o end.o options.o mixmax.o
CXXFLAGS = -O2 -std=c++11

is: $(OBJS)
	$(CXX)  $(CXXFLAGS) $^   -o $@

mixmax.o: 
	$(CXX)  $(CXXFLAGS) -c -o $@ MIXMAX/mixmax.cpp
$(OBJS): include.h

clean:
	/bin/rm -f *.o is core*
