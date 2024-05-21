//============================================================
//File box2D_1.cpp
//Motion of a free particle in a box  0<x<Lx 0<y<Ly
//Use integration with time step dt: x = x + vx*dt y=y+vy*dt
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
  double Lx,Ly,x0,y0,v0x,v0y,t0,tf,dt,t,x,y,vx,vy;
  int    i,nx,ny;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter Lx,Ly:\n";
  cin  >> Lx >> Ly;                 getline(cin,buf);
  cout << "# Lx = "<< Lx  << " Ly= "  << Ly  << endl;
  cout << "# Enter x0,y0,v0x,v0y:\n";
  cin  >> x0 >> y0 >> v0x >> v0y;   getline(cin,buf);
  cout << "# x0= " << x0  << " y0= "  << y0
       << " v0x= " << v0x << " v0y= " << v0y << endl;
  cout << "# Enter t0,tf,dt:\n";
  cin  >> t0 >> tf >> dt;           getline(cin,buf);
  cout << "# t0= " << t0  << " tf= "  << tf
       << " dt= "  << dt  << endl;
  if(Lx<= 0.0){cerr  << "Lx<=0 \n"; exit(1);}
  if(Ly<= 0.0){cerr  << "Ly<=0 \n"; exit(1);}
  if(x0<  0.0){cerr  << "x0<=0 \n"; exit(1);}
  if(x0>  Lx ){cerr  << "x0> Lx\n"; exit(1);}
  if(y0<  0.0){cerr  << "x0<=0 \n"; exit(1);}
  if(y0>  Ly ){cerr  << "y0> Ly\n"; exit(1);}
  if(v0x*v0x+v0y*v0y == 0.0 ){cerr << "v0 =0\n"; exit(1);}
//------------------------------------------------------------
//Initialize
  i  =  0 ;
  nx =  0 ;  ny = 0  ;
  t  = t0 ;
  x  = x0 ;  y  = y0 ;
  vx = v0x;  vy = v0y;
  ofstream myfile("box2D_1.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  while(t  <= tf){
    myfile << setw(28) << t  << " "  
	   << setw(28) << x  << " "  
	   << setw(28) << y  << " "  
	   << setw(28) << vx << " " 
	   << setw(28) << vy << '\n';
    i ++;
    t  = t0 + i*dt;
    x += vx*dt;
    y += vy*dt;
    if(x < 0.0 || x > Lx){
      vx = -vx;
      nx++;
    }
    if(y < 0.0 || y > Ly){
      vy = -vy;
      ny++;
    }
  }
  myfile.close();
  cout << "# Number of collisions:\n";
  cout << "# nx= " << nx << " ny= " << ny << endl;
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
