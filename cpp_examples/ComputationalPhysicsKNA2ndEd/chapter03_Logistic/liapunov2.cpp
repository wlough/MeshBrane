//===========================================================
//Discrete Logistic Map:
//Liapunov exponent from sum_i ln|f'(x_i)|
// NTRANS: number of discarted iterations in order to discart
//         transient behaviour
// NSTEPS: number of terms in the sum
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
  int    NTRANS,NSTEPS,i;
  double r,x0,x1,sum;
  string buf;
  // ----- Input:
  cout  << "# Enter NTRANS,NSTEPS, r, x0:\n";
  cin   >> NTRANS >> NSTEPS >> r   >>   x0; getline(cin,buf);
  cout  << "# NTRANS = " << NTRANS << endl;
  cout  << "# NSTEPS = " << NSTEPS << endl;
  cout  << "# r      = " << r      << endl;
  cout  << "# x0     = " << x0     << endl;

  for(i=1;i<=NTRANS;i++){
    x1   = r * x0  * (1.0 - x0);
    x0   = x1;
  }
  sum    = log(abs(r*(1.0 - 2.0*x0)));
  // ----- Initialize:
  ofstream myfile("lia.dat");
  myfile.precision(17);
  // ----- Calculate:
  myfile   << 1   << " " << x0  << " " << sum   << "\n";
  for(i=2;i<=NSTEPS;i++){
    x1   = r * x0  * (1.0-x0 );
    sum += log(abs(r*(1.0-2.0*x1)));
    myfile << i   << " " << x1  << " " << sum/i << "\n";
    x0   = x1;
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

 
