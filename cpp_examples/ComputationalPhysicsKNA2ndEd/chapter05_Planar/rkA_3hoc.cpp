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
  NEQ = 12;
}
//===============================
//Three particles on the plane 
//connected with springs of same
//k and of equal mass.
//k1: is k/m
//F = -k(1-l/r).\vec r
//===============================
void f(const double& t, double* X,double* dXdt){
  double x11,x12,x21,x22,x31,x32;
  double v11,v12,v21,v22,v31,v32;
  double r12,r13,r23;
  const double l = 1.0;
  //----------------
  x11 = X[0];x21 = X[4];x31 = X[8];
  x12 = X[1];x22 = X[5];x32 = X[9];
  v11 = X[2];v21 = X[6];v31 = X[10];
  v12 = X[3];v22 = X[7];v32 = X[11];
  //----------------
  r12 = pow((x11-x21)*(x11-x21)+(x12-x22)*(x12-x22),-0.5);
  r13 = pow((x11-x31)*(x11-x31)+(x12-x32)*(x12-x32),-0.5);
  r23 = pow((x21-x31)*(x21-x31)+(x22-x32)*(x22-x32),-0.5);
  r12 = -(1.0-l*r12);
  r13 = -(1.0-l*r13);
  r23 = -(1.0-l*r23);
  //----------------
  dXdt[0]  = v11;
  dXdt[1]  = v12;
  dXdt[2]  = k1*(x11-x21)*r12+k1*(x11-x31)*r13; // a11=dv11/dt
  dXdt[3]  = k1*(x12-x22)*r12+k1*(x12-x32)*r13; // a12=dv12/dt
  //----------------
  dXdt[4]  = v21;
  dXdt[5]  = v22;
  dXdt[6]  = k1*(x21-x11)*r12+k1*(x21-x31)*r23; // a21=dv21/dt
  dXdt[7]  = k1*(x22-x12)*r12+k1*(x22-x32)*r23; // a22=dv22/dt
  //----------------
  dXdt[8]  = v31;
  dXdt[9]  = v32;
  dXdt[10] = k1*(x31-x11)*r13+k1*(x31-x21)*r23; // a31=dv31/dt
  dXdt[11] = k1*(x32-x12)*r13+k1*(x32-x22)*r23; // a32=dv32/dt
}
//===============================
double energy(const double& t, double* X){
  double x11,x12,x21,x22,x31,x32;
  double v11,v12,v21,v22,v31,v32;
  double r12,r13,r23;
  double e;
  const double l = 1.0;
  //----------------
  x11 = X[0];x21 = X[4];x31 = X[8];
  x12 = X[1];x22 = X[5];x32 = X[9];
  v11 = X[2];v21 = X[6];v31 = X[10];
  v12 = X[3];v22 = X[7];v32 = X[11];
  //----------------
  r12 = pow((x11-x21)*(x11-x21)+(x12-x22)*(x12-x22),0.5);
  r13 = pow((x11-x31)*(x11-x31)+(x12-x32)*(x12-x32),0.5);
  r23 = pow((x21-x31)*(x21-x31)+(x22-x32)*(x22-x32),0.5);
  //----------------
  e   = 0.5*(v11*v11+v12*v12+v21*v21+v22*v22+v31*v31+v32*v32);
  e  += 0.5*k1*((r12-l)*(r12-l)+(r13-l)*(r13-l)+(r23-l)*(r23-l));
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


