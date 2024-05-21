//============== measure.cpp ==================
#include "include.h"

void measure(){
  cout << E() << " " << M() << '\n';
}

int E(){
  int e,snn,i,nn;
  e=0;
  for(i=0;i<N;i++){
    //Sum of neighboring spins:
    //only forward nn necessary in the sum
    if((nn=i+XNN)>= N) nn -= N; snn  = s[nn];
    if((nn=i+YNN)>= N) nn -= N; snn += s[nn];
    e += snn*s[i];
  }
  return -e;
}

int M(){
  int i,m;
  m=0;
  for(i=0;i<N;i++) m+=s[i];
  return m;
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
