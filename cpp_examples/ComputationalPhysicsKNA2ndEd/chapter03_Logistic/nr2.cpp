//===========================================================
//Newton Raphson of two functions of two variables
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

void solve2x2(double A[2][2],double b[2],double dx[2]);

int main(){
  const double eps  = 1.0e-6;
  const int    NMAX = 1000;
  double A[2][2],b[2],dx[2];
  double x, y, err;
  int    i;
  string buf;
  // ----- Input:
  cout << "# Enter x0,y0:\n";
  cin  >> x >> y;    getline(cin,buf);
  err = 1.0;
  cout << "iter       x           y              error    \n";
  cout << "-----------------------------------------------\n";
  cout << 0 << " " << x << " " << y << " " <<    err  << '\n';

  cout.precision(17);
  for(i=1;i<=NMAX;i++){
    b[0] =  -(2.0*x*x-3.0*x*y + y - 2.0); // -g1(x,y)
    b[1] =  -(3.0*x  +    x*y + y - 1.0); // -g2(x,y)
    // dg1/dx                    dg1/dy
    A[0][0] = 4.0*x-3.0*y; A[0][1] = 1.0-3.0*x;
    // dg2/dx                    dg2/dy
    A[1][0] = 3.0  +    y; A[1][1] = 1.0+    x;
    solve2x2(A,b,dx);
    x += dx[0];
    y += dx[1];
    err = 0.5*sqrt(dx[0]*dx[0]+dx[1]*dx[1]);
    cout << i << " " << x << " " << y << " " << err << endl;
    if(err < eps) break;
  }
}//main()
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

 
