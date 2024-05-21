//========================================================
//  Particle in Magnetic dipole field:
//  q B_1/m = k1 (3 x1 x3)/r^5 
//  q B_2/m = k1 (3 x2 x3)/r^5 
//  q B_3/m = k1[(3 x3 x3)/r^5-1/r^3]
//========================================================
#include "sr.h"
void f(double& t,double* Y, double* YP){
  double x1,x2,x3,v1,v2,v3,p1,p2,p3;
  double B1,B2,B3;
  double r,r5,r3;
  x1 = Y[0]; p1 = Y[3];
  x2 = Y[1]; p2 = Y[4];
  x3 = Y[2]; p3 = Y[5];
  velocity(p1,p2,p3,v1,v2,v3);
  //now we can use all x1,x2,x3,p1,p2,p3,v1,v2,v3
  YP[0]   = v1;
  YP[1]   = v2;
  YP[2]   = v3;
  //Acceleration:
  r       = sqrt(x1*x1+x2*x2+x3*x3);
  r3      = r*r*r;
  r5      = r*r*r3;
  if( r > 0.0){
    B1    = k1*( 3.0*x1*x3)/r5;
    B2    = k1*( 3.0*x2*x3)/r5;
    B3    = k1*((3.0*x3*x3)/r5-1/r3);
    YP[3] = v2*B3-v3*B2;
    YP[4] = v3*B1-v1*B3;
    YP[5] = v1*B2-v2*B1;
  }else{
    YP[3] = 0.0;
    YP[4] = 0.0;
    YP[5] = 0.0;
  }
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
