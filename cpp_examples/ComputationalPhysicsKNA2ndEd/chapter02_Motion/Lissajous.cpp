//============================================================
//File Lissajous.cpp
//Lissajous curves (special case) x(t)= cos(o1 t), y(t)= sin(o2 t)
//------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932

int main(){
//------------------------------------------------------------
//Declaration of variables
  double x0,y0,R,x,y,vx,vy,t,t0,tf,dt;
  double o1,o2,T1,T2;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter omega1 and omega2:\n";
  cin  >> o1      >> o2;getline(cin,buf);
  cout << "# Enter tf,dt:\n";
  cin  >> tf      >> dt;getline(cin,buf);
  cout <<"# o1= " << o1  << " o2= " << o2 << endl;
  cout <<"# t0= " << 0.0 << " tf= " << tf
       << " dt= " << dt  <<  endl;
//------------------------------------------------------------
//Initialize
  if(o1 <=0.0){cerr <<"Illegal value of o1\n";exit(1);}
  if(o2 <=0.0){cerr <<"Illegal value of o2\n";exit(1);}
  T1     = 2.0*PI/o1;
  T2     = 2.0*PI/o2;
  cout   << "# T1= "  << T1 << " T2= " << T2 << endl;
  ofstream myfile("Lissajous.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  t = t0;
  while( t <= tf ){
    x  =  cos(o1*t);
    y  =  sin(o2*t);
    vx = -o1*sin(o1*t);
    vy =  o2*cos(o2*t);
    myfile <<  t  << " "
	   <<  x  << " " <<  y << " "
	   << vx  << " " << vy << endl;
    t  =  t + dt;
  }   
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

 
