#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
  int    NSTEPS,i;
  double r,x0,x1;
  string buf;
  // ----- Input:
  cout << "# Enter NSTEPS, r, x0:\n";
  cin  >> NSTEPS   >> r >> x0;    getline(cin,buf);
  cout << "# NSTEPS = " << NSTEPS << endl;
  cout << "# r      = " << r      << endl;
  cout << "# x0     = " << x0     << endl;
  // ----- Initialize:
  ofstream myfile("log.dat");
  myfile.precision(17);
  // ----- Calculate:
  myfile << 0 << x0;
  for(i=1;i<=NSTEPS;i++){
    x1 = r * x0 * (1.0-x0);
    myfile << i << " " << x1 << "\n";
    x0 = x1;
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

 
