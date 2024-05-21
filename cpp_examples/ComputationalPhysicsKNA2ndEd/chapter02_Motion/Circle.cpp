//============================================================
//File Circle.cpp
//Constant angular velocity circular motion
//Set (x0,y0) center of circle, its radius R and omega.
//At t=t0, the particle is at theta=0
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
  double theta,omega;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter omega:\n";
  cin  >> omega;           getline(cin,buf);
  cout << "# Enter center of circle (x0,y0) and radius R:\n";
  cin  >> x0 >> y0 >> R;   getline(cin,buf);
  cout << "# Enter t0,tf,dt:\n";
  cin  >> t0 >> tf >> dt;  getline(cin,buf);
  cout <<"# omega= " << omega << endl;
  cout <<"# x0= "    << x0    << " y0= " << y0
       << " R=  "    << R     << endl;
  cout <<"# t0= "    << t0    << " tf= " << tf
       << " dt= "    << dt    << endl;
//------------------------------------------------------------
//Initialize
  if(R    <=0.0){cerr <<"Illegal value of R    \n";exit(1);}
  if(omega<=0.0){cerr <<"Illegal value of omega\n";exit(1);}
  cout    << "# T= "  << 2.0*PI/omega              << endl;
  ofstream myfile("Circle.dat");
  myfile.precision(17);// Set precision for numeric output to myfile to 17 digits
//------------------------------------------------------------
//Compute:
  t = t0;
  while( t <= tf ){
    theta = omega * (t-t0);
    x  =  x0+R*cos(theta);
    y  =  y0+R*sin(theta);
    vx =  -omega*R*sin(theta);
    vy =   omega*R*cos(theta);
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

 
