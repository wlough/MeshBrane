//====================================================================
// Simple variation of the rw.f90 program. In order to use it:
// 1. compile:
//> g++ -O2 -Wno-unused-result -std=c++11 rw1.cpp MIXMAX/mixmax.cpp -o rw1
// 2. run and save standard output in a file
//> ./rw1 <Nwalk> <Nstep>   >&  rwalk.dat
// where Nwalk the number of walks and Nstep the length N of the walk.
// 3. Output: 
// The first 15 configurations are explicitly printed in order to 
// visualize them. The format is:
//w <Nwalk>  <x> <y>
// so that the following command puts each configuration in a file 
// rwalk.<Nwalk>:
//> grep ^w rwalk.dat | awk '{print $3,$4 > "rwalk." $2}'
// The other data printed is the final position x,y, the distance 
// squared R^2 and the number of loops n in the format:
//R <Nwalk> <R^2> <x> <y> <n>
// and the average <R^2> can be computed for example by the command:
//> grep ^R rwalk.dat | awk '{print $3}' | ./average
//====================================================================
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>
using namespace std;

#include "MIXMAX/mixmax.hpp"

int seedby_urandom();
int main(int argc,char**argv){
  int Nwalk = 1000;
  int Nstep = 100000;
  int     n = 0;
  long   x ,y ;
  double xr,yr;
  //---------------------------------------
  //Initialize:
  mixmax_engine mxmx(0,0,0,1);
  uniform_real_distribution<double> drandom;
  mxmx.seed(seedby_urandom());
  cout.precision(17);
  if(argc < 3){
    cerr << "Usage: " << argv[0] << "<Nwalk> <Nstep>\n";
    exit(1);
  }
  Nwalk = atoi(argv[1]);
  Nstep = atoi(argv[2]);
  //---------------------------------------
  //Generate random walks:
  for(int iwalk=1;iwalk<=Nwalk;iwalk++){
    x=0;y=0;
    for(int istep=1;istep<=Nstep;istep++){
      int ir = int(drandom(mxmx)*4);
      switch(ir){
      case 0:
        x ++;
        break;
      case 1:
        x --;
        break;
      case 2:
        y ++;
        break;
      case 3:
        y --;
        break;
      }//switch(ir)
      if(iwalk <= 15) cout <<"w "<<iwalk<<" "<<x<<" "<<y<<endl;
      if(x==0 && y==0) n++;
    }//for(istep=1;istep<=Nstep;istep++)
    xr=double(x);yr=double(y);
    cout << "R "<<iwalk<<" "<<x*x+y*y<<" "<<x<<" "<<y<<" "<<n<<endl;
  }//for(iwalk=1;iwalk<=Nwalk;iwalk++)
}//main()
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

