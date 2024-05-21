// Compile:
// g++ -std=c++11 test_ranlux.cpp -o r
#include <random>
#include <iostream>
#include <fstream>
using namespace std;

int main(){
  //The object rlx is a Ranlux random number engine:
  ranlux48 rlx;
  //drandom is a distribution that produces uniformly
  //distributed numbers in [0,1)
  uniform_real_distribution<double> drandom;
  //--------------------------------------------------
  //Random numbers starting from the default state:
  cout  << "ranlux: ";
  for(int i=1;i<=5;i++) cout << drandom(rlx) << " ";
  cout  << endl;
  //--------------------------------------------------
  //Seeding by a seed:
  rlx.seed(377493872);
  cout  << "seed  : ";
  for(int i=1;i<=5;i++) cout << drandom(rlx) << " ";
  cout  << endl;
  //--------------------------------------------------
  //Saving the state to a file "seeds":
  ofstream oseed("seeds");
  oseed << rlx << endl;
  cout  << "more  : ";
  for(int i=1;i<=5;i++) cout << drandom(rlx) << " ";
  cout  << endl;
  //--------------------------------------------------
  //Reading an old state from the file seeds:
  ifstream iseed("seeds");
  iseed >> rlx;
  cout  << "same  : ";
  for(int i=1;i<=5;i++) cout << drandom(rlx) << " ";
  cout  << endl;
}//main
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

