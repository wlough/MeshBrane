//===========================================================
// Bifurcation Diagram of the Logistic Map
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
  const double NTRANS = 500;   //Number of discarted steps
  const double NSTEPS = 100;   //Number of recorded  steps
  const double RSTEPS = 2000;  //Number of values of r
  int i;
  double r,dr,x0,x1;
  
  // ------ Initialize:
  dr     = (rmax-rmin)/RSTEPS; //Increment in r
  ofstream myfile("bif.dat");
  myfile.precision(17);
  // ------ Calculate:
  r      = rmin;
  while( r <= rmax){
    x0   = 0.5;
    // ---- Transient steps: skip
    for(i=1;i<=NTRANS;i++){
      x1 = r * x0 * (1.0-x0);
      x0 = x1;
    }
    for(i=1;i<=NSTEPS;i++){
      x1 = r * x0 * (1.0-x0);
      myfile << r << " " << x1 << '\n';
      x0 = x1;
    }
    r += dr;
  }//while( r <= rmax)
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

 
