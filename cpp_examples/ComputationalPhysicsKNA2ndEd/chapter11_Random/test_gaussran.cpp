// Compile with:
// g++ -std=c++11 test_gaussran.cpp MIXMAX/mixmax.cpp -o mxmx
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
using namespace std;

#include "MIXMAX/mixmax.hpp"

int seedby_urandom();
int main(int argc,char **argv){
  //-----------------------------------------------------------
  //Number of random numbers that will be generated:
  int    Nrand=2000000; 
  //Mean value and standard deviation of gaussian distribution:
  const double mean=0.0,sigma=1.0; 
  //-----------------------------------------------------------
  //The object mxmx is a MIXMAX random number engine:
  mixmax_engine mxmx(0,0,0,1);
  mxmx.seed(seedby_urandom()); //seed mixmax using /dev/urandom
  //The object gaussran is a gaussian distribution:
  normal_distribution<double> gaussran(mean,sigma);
  //If provided at the command line, read Nrand:
  if(argc>1) Nrand = atoi(argv[1]);
  //-----------------------------------------------------------
  //Calculate random numbers and make a histogram:
  const double xm=6.0;//histogram in [-xm,xm]
  const int    nh=200;//number of histogram bins
  int        hist[nh];
  double         x,dx;
  int              ih;
  const double PI     = 2.0*atan2(1.0,0.0);
  const double sigma2 = 2.0*sigma*sigma;
  dx = 2.0*xm/nh;
  //Calculate histogram: hist[ih] counts # occurences
  for(ih=0;ih<nh;ih++) hist[ih]=0;
  for(int i=0;i<Nrand;i++){
    x = gaussran(mxmx);
    //skip if out of range:
    if(x < -xm || x > xm) continue;
    ih=int((x+xm)/dx);
    if(ih<0||ih>=nh){cerr<<"ih out of range\n";exit(1);}
    hist[ih]++;
  }
  //print results: The normalized histogram and compare to the
  //               gaussian distribution.
  cout << "#-----------------------------------------------------------------\n";
  cout << "# Histogram with: xm,nh,dx= " << xm   << " " << nh    << " " << dx    << endl;
  cout << "#         mean,sigma,Nrand= " << mean << " " << sigma << " " << Nrand << endl;
  cout << "# x                 h(x)    h(x)/(dx*Nrand)          f(x)\n";
  cout << "#-----------------------------------------------------------------\n";
  cout.precision(17);
  for(ih=0;ih<nh;ih++){
    x = ih*dx - xm + 0.5*dx;
    cout << x                                     << " "
         <<        hist[ih]                       << " "
         << double(hist[ih])/dx/Nrand             << " "
         << 1.0/sqrt(PI*sigma2)*
                  exp(-(x-mean)*(x-mean)/sigma2)  << '\n';
  }
  
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

