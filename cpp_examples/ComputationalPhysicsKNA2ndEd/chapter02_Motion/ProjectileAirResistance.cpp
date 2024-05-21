//============================================================
//File ProjectileAirResistance.cpp
//Shooting a progectile near the earth surface. No air resistance
//Starts at (0,0), set k, (v0,theta).
//------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932
#define  g 9.81

int main(){
//------------------------------------------------------------
//Declaration of variables
  double x0,y0,R,x,y,vx,vy,t,tf,dt,k;
  double theta,v0x,v0y,v0;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter k,v0,theta (in degrees):\n";
  cin  >> k >> v0 >> theta; getline(cin,buf);
  cout << "# Enter tf,dt:\n";
  cin  >> tf >> dt;         getline(cin,buf);
  cout <<"# k = "    << k     << endl;
  cout <<"# v0= "    << v0
       << " theta=  "<< theta << "o (degrees)" << endl;
  cout <<"# t0= "    << 0.0   << " tf= "       << tf
       << " dt= "    << dt    << endl;
//------------------------------------------------------------
//Initialize
  if(v0   <= 0.0)
    {cerr <<"Illegal value of v0   <= 0\n";exit(1);}
  if(k    <= 0.0)
    {cerr <<"Illegal value of k    <= 0\n";exit(1);}
  if(theta<= 0.0)
    {cerr <<"Illegal value of theta<= 0\n";exit(1);}
  if(theta>=90.0)
    {cerr <<"Illegal value of theta>=90\n";exit(1);}
  theta    = (PI/180.0)*theta; //convert to radians
  v0x      = v0*cos(theta);
  v0y      = v0*sin(theta);
  cout    << "# v0x= "  << v0x
	  << "  v0y= "  << v0y  << endl;
  ofstream myfile("ProjectileAirResistance.dat");
  myfile.precision(17);// Set precision for numeric output to myfile to 17 digits
//------------------------------------------------------------
//Compute:
  t = 0.0;
  while( t <= tf ){
    x  =  (v0x/k)*(1.0-exp(-k*t));
    y  =  (1.0/k)*(v0y+(g/k))*(1.0-exp(-k*t))-(g/k)*t;
    vx =  v0x*exp(-k*t);
    vy =  (v0y+(g/k))*exp(-k*t)-(g/k);
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

 
