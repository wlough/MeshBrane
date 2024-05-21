# $@ target, $< first dependency, $^ all dependencies, 
# $* "stem" of target filename e.g. %.f: %.F will five foo for foo.f
SHELL    = /bin/tcsh -f
CC       = g++
CXX      = g++
CXXFLAGS = -O2 -std=c++11

all:  autoc jack boot boot_bin

autoc:    autoc.o
jack:     jack.o
boot:     boot.o     mixmax.o
boot_bin: boot_bin.o mixmax.o
mixmax.o:
	$(CXX) -c $(CXXFLAGS) MIXMAX/mixmax.cpp -o $@


clean:
	-/bin/rm *.o
