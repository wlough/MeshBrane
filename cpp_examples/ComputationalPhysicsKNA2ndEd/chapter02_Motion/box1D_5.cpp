//============================================================
//File box1D_5.cpp
//Motion of a free particle in a box  0<x<L
//Use constant velocity equation: x = x0 + v0*(t-t0)
//Reverse velocity and redefine x0,t0 on boundaries
//------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

int main(){
//------------------------------------------------------------
//Declaration of variables
  float L,x0,v0,t0,tf,dt,t,ti,x,v;
  int   i;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter L:\n";
  cin  >> L;           getline(cin,buf);
  cout << "# L = " << L  << endl;
  cout << "# Enter x0,v0:\n";
  cin  >> x0 >> v0;    getline(cin,buf);
  cout << "# x0= " << x0 << " v0= "  << v0 << endl;
  cout << "# Enter t0,tf,dt:\n";
  cin  >> t0 >> tf >> dt;  getline(cin,buf);
  cout << "# t0= " << t0 << " tf= " << tf
       << " dt= "  << dt << endl;
  if( L <= 0.0f){cerr << "L <=0\n"; exit(1);}
  if( x0<  0.0f){cerr << "x0<=0\n"; exit(1);}
  if( x0>  L   ){cerr << "x0> L\n"; exit(1);}
  if( v0== 0.0f){cerr << "v0 =0\n"; exit(1);}
//------------------------------------------------------------
//Initialize
  t = t0;
  ti= t0;
  i = 0;
  ofstream myfile("box1D_5.dat");
  myfile.precision(9); // float precision (and a bit more...)
//------------------------------------------------------------
//Compute:
  while(t <= tf){
    x = x0 + v0*(t-t0);
    myfile << setw(17) << t  << " "  
	   << setw(17) << x  << " "  
	   << setw(17) << v0 << '\n';
    if(x < 0.0f || x > L) {
      x0=  x;
      t0=  t;
      v0= -v0;
    }
    i  +=  1;
    t   =  ti + i*dt;
  }
  myfile.close();
} //main()
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
