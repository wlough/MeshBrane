//============================================================
//File SimplePendulum.cpp
//Set pendulum original position at theta0 with no initial speed
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
  double l,x,y,vx,vy,t,t0,tf,dt;
  double theta,theta0,dtheta_dt,omega;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter l:\n";
  cin  >> l;           getline(cin,buf);
  cout << "# Enter theta0:\n";
  cin  >> theta0;   getline(cin,buf);
  cout << "# Enter t0,tf,dt:\n";
  cin  >> t0 >> tf >> dt;  getline(cin,buf);
  cout <<"# l= "   << l  << " theta0= " << theta0 << endl;
  cout <<"# t0= "  << t0 << " tf= " << tf
       << " dt= "  << dt << endl;
//------------------------------------------------------------
//Initialize
  omega = sqrt(g/l);
  cout << "# omega= " << omega
       << " T= "      << 2.0*PI/omega << endl;
  ofstream myfile("SimplePendulum.dat");
  myfile.precision(17);// Set precision for numeric output to myfile to 17 digits
//------------------------------------------------------------
//Compute:
  t    =  t0;
  while( t <= tf ){
    theta     =        theta0*cos(omega*(t-t0));
    dtheta_dt = -omega*theta0*sin(omega*(t-t0));
    x  =  l*sin(theta);
    y  = -l*cos(theta);
    vx =  l*dtheta_dt*cos(theta);
    vy =  l*dtheta_dt*sin(theta);
    myfile <<  t    << " "
	   <<  x    << " " <<  y << " "
	   << vx    << " " << vy << " "
	   << theta << dtheta_dt
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

 
