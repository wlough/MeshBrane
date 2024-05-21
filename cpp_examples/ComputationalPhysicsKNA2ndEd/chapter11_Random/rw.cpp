#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <thread>
using namespace std;

#include "MIXMAX/mixmax.hpp"

int seedby_urandom();
int main(){
  const int Nwalk = 1000;
  const int Nstep = 100000;
  double x,y;
  //---------------------------------------
  mixmax_engine mxmx(0,0,0,1);
  uniform_real_distribution<double> drandom;
  mxmx.seed(seedby_urandom());
  ofstream dataR("dataR");
  ofstream data;
  dataR.precision(17);data.precision(17);
  //---------------------------------------
  //Generate random walks:
  for(int iwalk=1;iwalk<=Nwalk;iwalk++){
    x=0.0;y=0.0;
    data.open("data");
    for(int istep=1;istep<=Nstep;istep++){
      int ir = int(drandom(mxmx)*4);
      switch(ir){
      case 0:
        x += 1.0;
        break;
      case 1:
        x -= 1.0;
        break;
      case 2:
        y += 1.0;
        break;
      case 3:
        y -= 1.0;
        break;
      }//switch(ir)
      data << x << " " << y << endl;
    }//for(istep=1;istep<=Nstep;istep++)
    data.close();
    dataR << x*x+y*y << endl;
    //wait for 2 seconds until next walk:
    this_thread::sleep_for(chrono::seconds(2));
  }//for(iwalk=1;iwalk<=Nwalk;iwalk++)
  dataR.close();
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

