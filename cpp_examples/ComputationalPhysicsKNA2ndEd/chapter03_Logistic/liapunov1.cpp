//===========================================================
//Discrete Logistic Map:
//Two trajectories with close initial conditions.
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
  int    NSTEPS,i;
  double r,x0,x1,x0t,x1t,epsilon;
  string buf;
  // ----- Input:
  cout << "# Enter NSTEPS, r, x0, epsilon:\n";
  cin  >> NSTEPS    >> r >> x0 >> epsilon;getline(cin,buf);
  cout << "# NSTEPS  = " << NSTEPS  << endl;
  cout << "# r       = " << r       << endl;
  cout << "# x0      = " << x0      << endl;
  cout << "# epsilon = " << epsilon << endl; 

  x0t = x0+epsilon;
  // ----- Initialize:
  ofstream myfile("lia.dat");
  myfile.precision(17);
  // ----- Calculate:
  myfile   << 1   << " "   << x0  << " "  << x0t << " "
           << abs(x0t-x0)/epsilon << "\n";
  for(i=2;i<=NSTEPS;i++){
    x1  = r * x0  * (1.0-x0 );
    x1t = r * x0t * (1.0-x0t);
    myfile << i   << " "   << x1  << " "  << x1t << " "
           << abs(x1t-x1)/epsilon << "\n";
    x0  = x1; x0t  = x1t;
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

 
