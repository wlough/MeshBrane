//========================================================
//The acceleration functions f3,f4(t,x1,x2,v1,v2) provided 
//by the user
//========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
extern double k1,k2;
//--------------------------------------------------------
//Motion in central force potential F ~ 1/r^k, k=k2-1:
//ax= k1*x1/r^k2 ay= k1*x2/r^k2
double
f3(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r;
  r=sqrt(x1*x1+x2*x2);
  if(r>0.0)
    return k1*x1*pow(r,-k2); // dx3/dt=dv1/dt=a1 
  else
    return 0.0;
}
//--------------------------------------------------------
double
f4(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r;
  r=sqrt(x1*x1+x2*x2);
  if(r>0.0)
    return k1*x2*pow(r,-k2); // dx4/dt=dv2/dt=a2 
  else
    return 0.0;
}
//--------------------------------------------------------
double
energy
  (const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r;
  r=sqrt(x1*x1+x2*x2);
  if( r > 0.0)
    return 0.5*(v1*v1+v2*v2) +  k1*pow(r,2.0-k2)/(k2-2.0);
  else
    return 0.0;
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
