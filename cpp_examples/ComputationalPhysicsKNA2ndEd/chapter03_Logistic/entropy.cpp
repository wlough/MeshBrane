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
  const int    NHIST  = 10000;
  const int    NTRANS = 2000;
  const int    NSTEPS = 5000000;
  const double xmin=0.0,xmax=1.0;
  int    i,ir,isum,n;
  double r,x0,x1,sum,dr,dx;
  double p[NHIST],S;
  string buf;
  
  // ----- Initialize:
  ofstream myfile("entropy.dat");
  myfile.precision(17);
  // ----- Calculate:
  for(i=0;i<NHIST;i++) p[i] = 0.0;
  dr = (rmax-rmin)/(RSTEPS-1);
  dx = (xmax-xmin)/(NHIST -1);
  for(ir=0;ir<RSTEPS;ir++){
    r = rmin+ir*dr;
    x0= xstart;
    for(i=1;i<=NTRANS;i++){
      x1  = r * x0  * (1.0-x0 );
      x0  = x1;
    }
    //make histogram:
    n=int(x0/dx); p[n]+=1.0;
    for(i=2;i<=NSTEPS;i++){
      x1    = r * x0  * (1.0-x0 );
      n     = int(x1/dx);
      p[n] += 1.0;
      x0    = x1;
    }
    //p[k] is now histogram of x-values.
    //Normalize so that sum_k p[k]*dx=1
    //to get probability distribution:
    for(i=0;i < NHIST;i++) p[i] /= (NSTEPS*dx);
    //sum all non zero terms: p[n]*log(p[n])*dx
    S = 0.0;
    for(i=0;i < NHIST;i++)
      if(p[i] > 0.0)
        S -= p[i]*log(p[i])*dx;
    myfile << r << " " << S << '\n';
  }//for(ir=0;ir<RSTEPS;ir++)
  myfile.close();

  myfile.open("entropy_hist.dat");
  myfile.precision(17);
  for(n=0;n<NHIST;n++){
    x0 = xmin + n*dx + 0.5*dx;
    myfile << r << " " << x0 << " " << p[n] << '\n';
  }
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

 
