// Compile with:
// g++ -std=c++11 test_mixmax.cpp MIXMAX/mixmax.cpp -o mxmx
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
using namespace std;

#include "MIXMAX/mixmax.hpp"

int main(){

  //The object mxmx is a MIXMAX random number engine:
  mixmax_engine mxmx(0,0,0,1);
  //The object drandom is a uniform distribution:
  uniform_real_distribution<double> drandom;
  //--------------------------------------------------
  //Random numbers after seeding with a chosen seed:
  mxmx.seed(1234);
  cout   << "mixmax: ";
  for(int i=1;i<=5;i++) cout << drandom(mxmx) << " ";
  cout   << endl;
  //--------------------------------------------------
  //Saving the state to a file "seeds":
  ofstream oseeds("seeds");
  oseeds << mxmx << endl;oseeds.close();
  cout   << "more  : ";
  for(int i=1;i<=5;i++) cout << drandom(mxmx) << " ";
  cout   << endl;
  //--------------------------------------------------
  //Reading an old state from the file seeds:
  ifstream iseeds("seeds");
  iseeds >> mxmx; iseeds.close();
  cout   << "same  : ";
  for(int i=1;i<=5;i++) cout << drandom(mxmx) << " ";
  cout   << endl;
  
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

