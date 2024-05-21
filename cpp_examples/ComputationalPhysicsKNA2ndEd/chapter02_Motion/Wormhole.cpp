//============================================================
//File Wormhole.cpp
//------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

#define PI 3.1415926535897932

void  crossC1(      double&  x,       double&  y,
                    double& vx,       double& vy,
                    const double& dt, const double&  R, const double& d);
void  crossC2(      double&  x,       double&  y,
                    double& vx,       double& vy,
                    const double& dt, const double&  R, const double& d);

int main(){
//------------------------------------------------------------
//Declaration of variables
  double Lx,Ly,L,R,d;
  double x0,y0,v0,theta;
  double t0,tf,dt;
  double t,x,y,vx,vy;
  double xc1,yc1,xc2,yc2,r1,r2;
  int    i;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter L,d,R:\n";
  cin  >> L  >> d  >> R;               getline(cin,buf);
  cout << "# Enter (x0,y0), v0, theta(degrees):\n";
  cin  >> x0 >> y0 >> v0 >> theta;     getline(cin,buf);
  cout << "# Enter tf,dt:\n";
  cin  >> tf >> dt;  getline(cin,buf);
  cout << "# L=  " << L  << " d=     " << d
       << "  R=  " << R  << endl;
  cout << "# x0= " << x0 << " y0=    " << y0    << endl;
  cout << "# v0= " << v0 << " theta= "
       << theta    << " degrees"                << endl;
  cout << "# tf= " << tf << " dt=    " << dt    << endl;
  if(L <=  d+2.0*R){cerr <<"L <= d+2*R \n";exit(1);}
  if(d <=    2.0*R){cerr <<"d <=   2*R \n";exit(1);}
  if(v0<=    0.0  ){cerr <<"v0<=     0 \n";exit(1);}
//------------------------------------------------------------
//Initialize
  theta = (PI/180.0)*theta;
  i     =  0;
  t     =  0.0;
  x     =  x0           ; y     =  y0;
  vx    =  v0*cos(theta); vy    =  v0*sin(theta);
  cout << "# x0= " << x0 << "  y0= " << y0
       << " v0x= " << vx << " v0y= " << vy << endl;
//Wormhole's centers:
  xc1   =  0.5*d; yc1   =  0.0;
  xc2   = -0.5*d; yc2   =  0.0;
//Box limits coordinates:
  Lx    =  0.5*L; Ly    =  0.5*L;
//Test if already inside cut region:
  r1    = sqrt((x-xc1)*(x-xc1)+(y-yc1)*(y-yc1));
  r2    = sqrt((x-xc2)*(x-xc2)+(y-yc2)*(y-yc2));
  if(r1<=        R){cerr <<"r1 <=    R \n";exit(1);}
  if(r1<=        R){cerr <<"r2 <=    R \n";exit(1);}
//Test if outside box limits:
  if(abs(x) >=  Lx){cerr <<"|x|>=   Lx \n";exit(1);}
  if(abs(y) >=  Ly){cerr <<"|y|>=   Ly \n";exit(1);}
  ofstream myfile("Wormhole.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  while( t <  tf ){
    myfile <<  t  << " "
	   <<  x  << " " <<  y << " "
	   << vx  << " " << vy << endl;
    i++;
    t  =  i*dt;
    x += vx*dt; y += vy*dt;
// Toroidal boundary conditions:
    if( x >  Lx) x  = x - L;
    if( x < -Lx) x  = x + L;
    if( y >  Ly) y  = y - L;
    if( y < -Ly) y  = y + L;
    r1    = sqrt((x-xc1)*(x-xc1)+(y-yc1)*(y-yc1));
    r2    = sqrt((x-xc2)*(x-xc2)+(y-yc2)*(y-yc2));
// Notice: we pass r1 as radius of circle, not R
    if     (r1 < R)
      crossC1(x,y,vx,vy,dt,r1,d);
    else if(r2 < R)
      crossC2(x,y,vx,vy,dt,r2,d);
// small chance here that still in C1 or C2, but OK since
// another dt-advance given at the beginning of for-loop
  }// while( t <= tf )  
}  // main ()
//------------------------------------------------------------
void  crossC1(      double&  x,       double&  y,
                    double& vx,       double& vy,
              const double& dt, const double&  R,
              const double& d){

  double vr,v0,theta,xc,yc;
  cout << "# Inside C1: (x,y,vx,vy,R)= "
       << x  << " " << y  << " "
       << vx << " " << vy << " " <<R << endl;
  xc    =  0.5*d;             //center of C1
  yc    =  0.0;
  theta =  atan2(y-yc,x-xc);
  x     = -xc - R*cos(theta); //new x-value, y invariant
//Velocity transformation:
  vr    =  vx*cos(theta)+vy*sin(theta);
  v0    = -vx*sin(theta)+vy*cos(theta);
  vx    =  vr*cos(theta)+v0*sin(theta);
  vy    = -vr*sin(theta)+v0*cos(theta);
//advance x,y, hopefully outside C2: 
  x     = x + vx*dt;
  y     = y + vy*dt;
  cout << "# Exit   C2: (x,y,vx,vy  )= "
       << x  << " " << y  << " "
       << vx << " " << vy << endl;
}//void  crossC1( )
//------------------------------------------------------------
void  crossC2(      double&  x,       double&  y,
                    double& vx,       double& vy,
              const double& dt, const double&  R,
              const double& d){

  double vr,v0,theta,xc,yc;
  cout << "# Inside C2: (x,y,vx,vy,R)= "
       << x  << " " << y  << " "
       << vx << " " << vy << " " <<R << endl;
  xc    = -0.5*d;             //center of C2
  yc    =  0.0;
  theta =  PI-atan2(y-yc,x-xc);
  x     = -xc + R*cos(theta); //new x-value, y invariant
//Velocity transformation:
  vr    = -vx*cos(theta)+vy*sin(theta);
  v0    =  vx*sin(theta)+vy*cos(theta);
  vx    = -vr*cos(theta)-v0*sin(theta);
  vy    = -vr*sin(theta)+v0*cos(theta);
//advance x,y, hopefully outside C1: 
  x     = x + vx*dt;
  y     = y + vy*dt;
  cout << "# Exit   C1: (x,y,vx,vy  )= "
       << x  << " " << y  << " "
       << vx << " " << vy << endl;
}//void  crossC2( )
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

 
