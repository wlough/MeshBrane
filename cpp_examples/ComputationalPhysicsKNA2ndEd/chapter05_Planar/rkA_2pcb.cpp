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
  NEQ = 8;
}
//===============================
//Two particles of the same 
//mass on the plane interacting 
//via Coulombic force
//===============================
void f(const double& t, double* X,double* dXdt){
  double x11,x12,x21,x22;
  double v11,v12,v21,v22;
  double r,r3;
  //----------------
  x11 = X[0];x21 = X[4];
  x12 = X[1];x22 = X[5];
  v11 = X[2];v21 = X[6];
  v12 = X[3];v22 = X[7];
  //----------------
  r   = sqrt((x11-x21)*(x11-x21)+(x12-x22)*(x12-x22));
  r3  = 1.0/(r*r*r);
  //----------------
  dXdt[0] = v11;
  dXdt[1] = v12;
  dXdt[2] = k1*(x11-x21)*r3; // a11=dv11/dt
  dXdt[3] = k1*(x12-x22)*r3; // a12=dv12/dt
  dXdt[4] = v21;
  dXdt[5] = v22;
  dXdt[6] = -dXdt[2];        // a21=dv21/dt
  dXdt[7] = -dXdt[3];        // a22=dv22/dt
}
//===============================
double energy(const double& t, double* X){
  double x11,x12,x21,x22;
  double v11,v12,v21,v22;
  double r,r3;
  double e;
  //----------------
  x11 = X[0];x21 = X[4];
  x12 = X[1];x22 = X[5];
  v11 = X[2];v21 = X[6];
  v12 = X[3];v22 = X[7];
  //----------------
  e   = 0.5*(v11*v11+v12*v12+v21*v21+v22*v22);
  r   = sqrt((x11-x21)*(x11-x21)+(x12-x22)*(x12-x22));
  e  += k1/r;
  return e;
}
//( echo -0.5 0 ; echo 40000 0 10 -1 1 0.3 0 1 -1 -0.3 0 ) | ./a.out
//( echo  0.5 0 ; echo 40000 0 10 -1 1 0.3 0 1 -1 -0.3 0 ) | ./a.out
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


