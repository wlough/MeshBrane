// Compile:
// g++ urandom.cpp -o u ; ./u
#include <iostream>
#include <fstream>
#include <cstdlib>
using namespace std;

int seedby_urandom();
int seedby_pid    ();
int main(){
  cout << seedby_urandom() << endl;
  cout << seedby_pid    () << endl;
}

// Use /dev/urandom for more randomness.
#include <unistd.h>
#include <fcntl.h>
int seedby_urandom(){
  int ur,fd;
  fd = open("/dev/urandom", O_RDONLY);
  read (fd,&ur,sizeof(int));
  close(fd);
  return (ur>0)?ur:-ur;
}
// Seed using the time of running and the
// process id (pid) in the system
#include <sys/types.h>
#include <sys/stat.h>
int seedby_pid(){
  int pid,sd;
  sd   = (int) time( (time_t) 0);
  pid  = (int) getpid(); //Process ID number
  sd   = sd ^ (pid + (pid << 15));
  return (sd>0)?sd:-sd;
}
//  ---------------------------------------------------------------------
//  Copyright by Konstantinos N. Anagnostopoulos (2004-2014)
//  Physics Dept., National Technical University,
//  konstant@mail.ntua.gr, www.physics.ntua.gr/~konstant
//  
//  This program is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, version 3 of the License.
//  
//  This program is distributed in the hope that it will be useful, but
//  WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//  General Public License for more details.
//  
//  You should have received a copy of the GNU General Public Liense along
//  with this program.  If not, see <http://www.gnu.org/licenses/>.
//  -----------------------------------------------------------------------

