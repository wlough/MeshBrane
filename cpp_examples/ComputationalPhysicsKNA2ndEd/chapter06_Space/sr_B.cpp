//========================================================
//  Particle in constant Magnetic and electric field
//  q B/m = k1 z   q E/m = k2 x + k3 y + k4 z
//========================================================
#include "sr.h"
void f(double& t,double* Y, double* YP){
  double x1,x2,x3,v1,v2,v3,p1,p2,p3;
  x1 = Y[0]; p1 = Y[3];
  x2 = Y[1]; p2 = Y[4];
  x3 = Y[2]; p3 = Y[5];
  velocity(p1,p2,p3,v1,v2,v3);
  //now we can use all x1,x2,x3,p1,p2,p3,v1,v2,v3
  YP[0] = v1;
  YP[1] = v2;
  YP[2] = v3;
  //Acceleration:
  YP[3] = k2 + k1 * v2;
  YP[4] = k3 - k1 * v1;
  YP[5] = k4;
}
//---------------------------------
double energy(const double& t, double* Y){
  double e;
  double x1,x2,x3,v1,v2,v3,p1,p2,p3,psq;
  x1 = Y[0]; p1 = Y[3];
  x2 = Y[1]; p2 = Y[4];
  x3 = Y[2]; p3 = Y[5];
  psq= p1*p1+p2*p2+p3*p3;
  //Kinetic   Energy:
  e  = sqrt(1.0+psq)-1.0;
  //Potential Energy/m_0
  e += - k2*x1 - k3*x2 - k4*x3;
  
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
