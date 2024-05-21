//===========================================================
//Discrete Logistic Map:
//Liapunov exponent from sum_i ln|f'(x_i)|
//Calculation for r in [rmin,rmax] with RSTEPS steps
// RSTEPS: values or r studied: r=rmin+(rmax-rmin)/RSTEPS
// NTRANS: number of discarted iterations in order to discart
//         transient behaviour
// NSTEPS: number of terms in the sum
// xstart: value of initial x0 for every r
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
  const double rmin   = 2.5;
  const double rmax   = 4.0;
  const double xstart = 0.2;
  const int    RSTEPS = 1000;
  const int    NSTEPS = 60000;
  const int    NTRANS = 2000;
  int    i,ir;
  double r,x0,x1,sum,dr;
  string buf;
  
  // ----- Initialize:
  ofstream myfile("lia.dat");
  myfile.precision(17);
  // ----- Calculate:
  dr      = (rmax-rmin)/(RSTEPS-1);
  for(ir  = 0; ir < RSTEPS;ir++){
    r     = rmin+ir*dr;
    x0    = xstart;
    for(i = 1; i <= NTRANS; i++){
      x1  = r * x0  * (1.0-x0 );
      x0  = x1;
    }
    sum   = log(abs(r*(1.0-2.0*x0)));
    //Calculate:
    for(i = 2; i <= NSTEPS; i++){
      x1  = r * x0  * (1.0-x0 );
      sum+= log(abs(r*(1.0-2.0*x1)));
      x0  = x1;
    }
    myfile << r << " " << sum/NSTEPS << '\n';
  }//for(ir=0;ir<RSTEPS;ir++)
  myfile.close();
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

 
