#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
extern double k1,k2;
//--------------------------------------------------------
//Sets number of equations:
void finit(int& NEQ){
  NEQ = 4;
}
//===============================
//Motion on the plane
//a1 = -k2 vx    a2 = -k2 vy - k1
//===============================
void f(const double& t, double* X,double* dXdt){
  double x1,x2;
  double v1,v2;
  //----------------
  x1 = X[0];
  x2 = X[1];
  v1 = X[2];
  v2 = X[3];
  //----------------
  dXdt[0] =     v1;
  dXdt[1] =     v2;
  dXdt[2] = -k2*v1;    // a1=dv1/dt
  dXdt[3] = -k2*v2-k1; // a2=dv2/dt
}
//===============================
double energy(const double& t, double* X){
  double x1,x2;
  double v1,v2;
  //----------------
  x1 = X[0];
  x2 = X[1];
  v1 = X[2];
  v2 = X[3];
  //----------------
  return 0.5*(v1*v1+v2*v2) + k1*x2;
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


