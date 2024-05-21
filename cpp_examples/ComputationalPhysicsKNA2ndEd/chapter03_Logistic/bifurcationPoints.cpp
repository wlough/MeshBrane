//===========================================================
//        bifurcationPoints.cpp
// Calculate bifurcation points of the discrete logistic map
// at period k by solving the condition 
// g1(x,r) = x - F(k,x,r)   = 0
// g2(x,r) = dF(k,x,r)/dx+1 = 0
// determining when the Floquet multiplier bacomes 1
// F(k,x,r) iterates F(x,r) = r*x*(x-1) k times
// The equations are solved by using a Newton-Raphson method
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
double F      (const int& k,const double& x,const double& r);
double dFdx   (const int& k,const double& x,const double& r);
double dFdr   (const int& k,const double& x,const double& r);
double d2Fdx2 (const int& k,const double& x,const double& r);
double d2Fdrdx(const int& k,const double& x,const double& r);
void solve2x2 (double A[2][2],double b[2],double dx[2]);

int main(){
  const double tol  = 1.0e-10;
  double r0,x0;
  double A[2][2],B[2],dX[2];
  double error;
  int k,iter;
  string buf;
  
  // ------ Input:
  cout << "# Enter k,r0,x0:\n";
  cin  >> k >> r0 >> x0;         getline(cin,buf);
  cout << "# Period k= "             << k << endl;
  cout << "# r0= " << r0 << " x0= " << x0 << endl;
  // ------ Initialize
  error = 1.0; //initial large value of error>tol
  iter  =   0;
  cout.precision(17);
  while(error > tol){
    // ---- Calculate jacobian matrix
    A[0][0] = 1.0  -dFdx(k,x0,r0);
    A[0][1] = -dFdr     (k,x0,r0);
    A[1][0] = d2Fdx2    (k,x0,r0);
    A[1][1] = d2Fdrdx   (k,x0,r0);
    B[0]    = -x0 +    F(k,x0,r0);
    B[1]    = -dFdx     (k,x0,r0)-1.0;
    // ---- Solve a 2x2 linear system:
    solve2x2(A,B,dX);
    x0     = x0 + dX[0];
    r0     = r0 + dX[1];
    error  = 0.5*sqrt(dX[0]*dX[0]+dX[1]*dX[1]);
    iter++;
    cout  <<iter  
          << " x0=  " << x0
          << " r0=  " << r0
          << " err= " << error << '\n';
  }//while(error > tol)
}//main()
//===========================================================
//Function F(k,x,r) and its derivatives
double F      (const int& k,const double& x,const double& r){
  double x0;
  int     i;
  x0  = x;
  for(i=1;i<=k;i++) x0 = r*x0*(1.0-x0);
  return x0;
}
// ----------------------------------
double dFdx   (const int& k,const double& x,const double& r){
  double eps;
  eps     = 1.0e-6*x;
  return (F(k,x+eps,r)-F(k,x-eps,r))/(2.0*eps);
}
// ----------------------------------
double dFdr   (const int& k,const double& x,const double& r){
  double eps;
  eps     = 1.0e-6*r;
  return (F(k,x,r+eps)-F(k,x,r-eps))/(2.0*eps);
}
// ----------------------------------
double d2Fdx2 (const int& k,const double& x,const double& r){
  double eps;
  eps     = 1.0e-6*x;
  return (F(k,x+eps,r)-2.0*F(k,x,r)+F(k,x-eps,r))/(eps*eps);
}
// ----------------------------------
double d2Fdrdx(const int& k,const double& x,const double& r){
  double epsx,epsr;
  epsx     = 1.0e-6*x;
  epsr     = 1.0e-6*r;
  return ( F(k,x+epsx,r+epsr)-F(k,x+epsx,r-epsr) 
          -F(k,x-epsx,r+epsr)+F(k,x-epsx,r-epsr))     
          /(4.0*epsx*epsr);
}
//===========================================================
void solve2x2(double A[2][2],double b[2],double dx[2]){
  double num0,num1,det;

  num0   = A[1][1] * b[0]    - A[0][1] * b[1];
  num1   = A[0][0] * b[1]    - A[1][0] * b[0];
  det    = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  if(det == 0.0){cerr << "solve2x2: det=0\n";exit(1);}
  dx[0]  = num0/det;
  dx[1]  = num1/det;

}//solve2x2()
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

 
