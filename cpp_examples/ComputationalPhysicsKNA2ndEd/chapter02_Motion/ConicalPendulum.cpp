//============================================================
//File ConicalPendulum.cpp
//Set pendulum angular velocity omega and display motion in 3D
//------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932
#define g  9.81

int main(){
//------------------------------------------------------------
//Declaration of variables
  double l,r,x,y,z,vx,vy,vz,t,tf,dt;
  double theta,cos_theta,sin_theta,omega;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter l,omega:\n";
  cin  >> l  >> omega;           getline(cin,buf);
  cout << "# Enter tf,dt:\n";
  cin  >> tf >> dt;              getline(cin,buf);
  cout << "# l= "  << l   << " omega= " << omega << endl;
  cout << "# T=  " << 2.0*PI/omega
       << " omega_min= "  << sqrt(g/l)  << endl;
  cout <<"# t0= "  << 0.0 << " tf= "    << tf
       << " dt= "  << dt  << endl;
//------------------------------------------------------------
//Initialize
  cos_theta = g/(omega*omega*l);
  if( cos_theta >= 1.0){
    cerr << "cos(theta)>= 1\n";
    exit(1);
  }
  sin_theta = sqrt(1.0-cos_theta*cos_theta);
  z = -g/(omega*omega); //they remain constant throught;
  vz= 0.0;              //the  motion
  r =  g/(omega*omega)*sin_theta/cos_theta;
  ofstream myfile("ConicalPendulum.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  t    =  0.0;
  while( t <= tf ){
    x  =  r*cos(omega*t);
    y  =  r*sin(omega*t);
    vx = -r*sin(omega*t)*omega;
    vy =  r*cos(omega*t)*omega;
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

 
