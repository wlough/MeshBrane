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
//Motion in hydrogen atom + positron:
//f(r) = 1/r^2-r/k1^3
//ax= f(r)*x1/r ay= f(r)*x2/r
double
f3(const double& t , const double& x1, const double& x2,
   const double& v1, const double& v2){
  double r2,r,fr;
  r2=x1*x1+x2*x2;
  r =sqrt(r2);
  if(r <= k1 && r2 > 0.0)
    fr = 1.0/r2-r/(k1*k1*k1);
  else
    fr = 0.0;

  if(fr > 0.0 && r > 0.0)
    return fr*x1/r; // dx3/dt=dv1/dt=a1 
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
  if(r <= k1 && r2 > 0.0)
    fr = 1.0/r2-r/(k1*k1*k1);
  else
    fr = 0.0;

  if(fr > 0.0 && r > 0.0)
    return fr*x2/r; // dx4/dt=dv2/dt=a2 
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
  if(r <= k1 && r > 0.0)
    Vr = 1/r + 0.5*r*r/(k1*k1*k1) - 1.5 / k1;
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
