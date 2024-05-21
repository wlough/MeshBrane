//============== init.cpp    ==================
// file init.cpp
// init(start): start = 0: cold start
//              start = 1: hot  start
//=============================================
#include "include.h"
//Global variables:
int    s[N];
int    seed;
double beta;
double prob[5];
mixmax_engine mxmx(0,0,0,1);
uniform_real_distribution<double> drandom;

void init(int start){
  int i;
  
  mxmx.seed(seed);
  //Initialize probabilities:
  for(i=2;i<5;i+=2) prob[i] = exp(-2.0*beta*i);
  //Initial configuration:
  switch(start){
  case 0://cold start
    for(i=0;i<N;i++) s[i]=1;
    break;
  case 1://hot start
    for(i=0;i<N;i++){
      if(drandom(mxmx) < 0.5)
	s[i] =  1;
      else
	s[i] = -1;
    }
    break;
  default:
    cout << "start= " << start
         << " not valid. Exiting....\n";
    exit(1);
    break;
  }
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
