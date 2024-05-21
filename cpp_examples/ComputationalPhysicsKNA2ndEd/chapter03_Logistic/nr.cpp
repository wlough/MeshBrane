//===========================================================
//Newton Raphson of function of one variable
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
  const double rho  = 15.0;
  const double eps  = 1.0e-6;
  const int    NMAX = 1000;
  double x0, x1, err, g, gp;
  int    i;
  string buf;
  // ----- Input:
  cout << "# Enter x0:\n";
  cin  >> x0;    getline(cin,buf);
  err = 1.0;
  cout << "iter           x                      error    \n";
  cout << "-----------------------------------------------\n";
  cout << 0   << " "   << x0   << " "   <<       err  << '\n';

  cout.precision(17);
  for(i=1;i<=NMAX;i++){
    //value of function g(x):
    g   = x0*tan(x0)-sqrt(rho*rho-x0*x0);
    //value of the derivative g'(x):
    gp  = x0/sqrt(rho*rho-x0*x0)+x0/(cos(x0)*cos(x0))+tan(x0);
    x1  = x0 - g/gp;
    err = abs(x1-x0);
    cout  << i << " " << x1 << " " << err << '\n';
    if(err < eps) break;
    x0  = x1;
  }
}//main()
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

 
