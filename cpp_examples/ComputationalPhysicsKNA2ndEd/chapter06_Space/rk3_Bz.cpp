#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
#include "rk3.h"
using namespace std;
void f(double& t,double* Y, double* YP){
  double x1,x2,x3,v1,v2,v3;
  double x3p;
  x1 = Y[0]; v1 = Y[3];
  x2 = Y[1]; v2 = Y[4];
  x3 = Y[2]; v3 = Y[5];
  //Velocities:   dx_i/dt = v_i
  YP[0] = v1;
  YP[1] = v2;
  YP[2] = v3;
  //If x3>0 we add a linear term to B:
  x3p   = 0.0;
  if(x3 > 0.0) x3p = x3;
  //Acceleration: dv_i/dt = a_i
  YP[3] =  (k1+k2*x3p) * v2;
  YP[4] = -(k1+k2*x3p) * v1;
  YP[5] =  0.0;
}
//---------------------------------
double energy(const double& t, double* Y){
  double e;
  double x1,x2,x3,v1,v2,v3;
  x1 = Y[0]; v1 = Y[3];
  x2 = Y[1]; v2 = Y[4];
  x3 = Y[2]; v3 = Y[5];
  //Kinetic   Energy:
  e  = 0.5*(v1*v1+v2*v2+v3*v3);
  
  return e;
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
