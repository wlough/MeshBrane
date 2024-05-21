//============================================================
//File ChargeInB.cpp
//A charged particle of mass m and charge q enters a magnetic
//field B in +z direction. It enters with velocity 
//v0x=0,v0y=v0 cos(theta),v0z=v0 sin(theta), 0<=theta<pi/2
//at the position x0=-v0y/omega, omega=q B/m
//
//Enter v0 and theta and see trajectory from 
//t0=0 to tf at step dt
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
  double x,y,z,vx,vy,vz,t,tf,dt;
  double x0,y0,z0,v0x,v0y,v0z,v0;
  double theta,omega;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter omega:\n";
  cin  >> omega;           getline(cin,buf);
  cout << "# Enter v0, theta (degrees):\n";
  cin  >> v0 >> theta;     getline(cin,buf);
  cout << "# Enter tf,dt:\n";
  cin  >> tf >> dt;              getline(cin,buf);
  cout << "# omega= " << omega
       << " T= "      << 2.0*PI/omega << endl;
  cout << "# v0=    " << v0
       << " theta=  " << theta
       << "o(degrees)"<< endl;
  cout <<"# t0= "     << 0.0 << " tf= "    << tf
       << " dt= "     << dt  << endl;
//------------------------------------------------------------
//Initialize
  if(theta<0.0 || theta>=90.0) exit(1);
  theta = (PI/180.0)*theta; //convert to radians
  v0y   = v0*cos(theta);
  v0z   = v0*sin(theta);
  cout << "# v0x= " << 0.0
       << "  v0y= " << v0y
       << "  v0z= " << v0z  << endl;
  x0    = - v0y/omega;
  cout << "# x0= " << x0
       << "  y0= " << y0
       << "  z0= " << z0    << endl;
  cout << "# xy plane: Circle with center (0,0) and R= "
       << abs(x0)           << endl;
  cout << "# step of helix: s=v0z*T= "
       <<  v0z*2.0*PI/omega << endl;
  ofstream myfile("ChargeInB.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  t    =  0.0;
  vz   = v0z;
  while( t <= tf ){
    x  =  x0*cos(omega*t);
    y  = -x0*sin(omega*t);
    z  =  v0z*t;
    vx =  v0y*sin(omega*t);
    vy =  v0y*cos(omega*t);
    myfile <<  t    << " "
	   <<  x    << " " <<  y << " " <<  z << " "
	   << vx    << " " << vy << " " << vz << " "
	   << endl;
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

 
