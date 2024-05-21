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
//Motion Yukawa potential
//f(r) = k1 e^(-r/k2) (1+r/k2)
//ax= f(r)*x1/r ay= f(r)*x2/r
double
f3(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r2,r,fr;
  r2=x1*x1+x2*x2;
  r =sqrt(r2);
  if(r2 > 0.0)
    fr = k1*exp(-r/k2)/r2*(1.0+r/k2); 
  else
    fr = 0.0;

  if(r > 0.0)
    return fr*x1/r;
  else
    return 0.0;
}
//--------------------------------------------------------
double
f4(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r2,r,fr;
  r2=x1*x1+x2*x2;
  r =sqrt(r2);
  if(r2 > 0.0)
    fr = k1*exp(-r/k2)/r2*(1.0+r/k2); 
  else
    fr = 0.0;

  if(r > 0.0)
    return fr*x2/r;
  else
    return 0.0;
}
//--------------------------------------------------------
double
energy
  (const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r,Vr;
  r=sqrt(x1*x1+x2*x2);
  if( r > 0.0)
    Vr = k1*exp(-r/k2)/r;
  else
    Vr = 0.0;
  return 0.5*(v1*v1+v2*v2) + Vr;
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
