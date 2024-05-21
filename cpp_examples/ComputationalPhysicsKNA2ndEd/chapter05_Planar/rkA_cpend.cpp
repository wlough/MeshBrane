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
//Two equal mass pendulums coupled
//with a spring
//===============================
void f(const double& t, double* X,double* dXdt){
  double th1,th2,om1,om2;
  double cth1,sth1,cth2,sth2;
  double r,Dl;
  //----------------
  th1  = X[0]    ; th2  = X[2];
  om1  = X[1]    ; om2  = X[3];
  //----------------
  cth1 = cos(th1); sth1 = sin(th1);
  cth2 = cos(th2); sth2 = sin(th2);
  //----------------
  r    = sqrt((1+sth2-sth1)*(1+sth2-sth1)+(cth1-cth2)*(cth1-cth2));
  Dl   = r - 1.0;
  //----------------
  dXdt[0] =  om1;
  dXdt[1] = -k1*sth1+k2*Dl/r*( (1.0+sth2-sth1)*cth1+(cth1-cth2)*sth1 );
  dXdt[2] =  om2;
  dXdt[3] = -k1*sth2-k2*Dl/r*( (1.0+sth2-sth1)*cth2+(cth1-cth2)*sth2 );
}
//===============================
double energy(const double& t, double* X){
  double th1,th2,om1,om2;
  double cth1,sth1,cth2,sth2;
  double r,Dl;
  double e;
  //----------------
  th1  = X[0]    ; th2  = X[2];
  om1  = X[1]    ; om2  = X[3];
  //----------------
  cth1 = cos(th1); sth1 = sin(th1);
  cth2 = cos(th2); sth2 = sin(th2);
  //----------------
  r    =  sqrt((1.0+sth2-sth1)*(1.0+sth2-sth1)+(cth1-cth2)*(cth1-cth2));
  Dl   =  r - 1.0;
  //----------------
  e    =  0.5*(om1*om1+om2*om2);
  e   +=  -k1*(cth1+cth2) + 0.5*k2*Dl*Dl;
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


