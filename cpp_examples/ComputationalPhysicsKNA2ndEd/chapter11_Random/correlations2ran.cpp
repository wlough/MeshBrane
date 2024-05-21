//==========================================================
//Program that produces N random points (i,j) with
//0<= i,j < 10000. Simple qualitative test of  serial 
//correlations of random number generators on the plane.
//
//compile:
//g++ correlations2ran.f90 naiveran.f90 drandom.f90 
//==========================================================
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
using namespace std;

double naiveran(),drandom();

int main(int argc,char **argv){
  const int L = 10000;
  int i,N;
  N = 1000;
  //read the number of points from the command line:
  if(argc > 1) N = atoi(argv[1]);
  for(i=1;i<=N;i++){
    cout << int(L*naiveran()) << " "
         << int(L*naiveran()) << '\n';
  //cout << int(L*drandom ()) << " "
  //     << int(L*drandom ()) << '\n';
  }
}//main()
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
