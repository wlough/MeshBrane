//============================================================
//File Cylinder3D.cpp
//Motion of a free particle in a cylinder with axis the z-axis,
//radius R and 0<z<L
//Use integration with time step dt: x = x + vx*dt 
//                                   y = y + vy*dt 
//                                   z = z + vz*dt
//Use function reflectVonCircle for colisions at r=R
//------------------------------------------------------------
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;

void reflectVonCircle(double& vx,double& vy,
		      double& x ,double& y ,
		      const      double& xc,
		      const      double& yc,
		      const      double& R );

int main(){
//------------------------------------------------------------
//Declaration of variables
  double x0,y0,z0,v0x,v0y,v0z,t0,tf,dt,t,x,y,z,vx,vy,vz;
  double L,R,R2,vxy,rxy,r2xy,xc,yc;
  int    i,nr,nz;
  string buf;
//------------------------------------------------------------
//Ask user for input:
  cout << "# Enter R,L:\n";
  cin  >> R  >> L;                      getline(cin,buf);
  cout << "# R= " << R  << " L= "  << L  << endl;
  cout << "# Enter x0,y0,z0,v0x,v0y,v0z:\n";
  cin  >> x0>>y0>>z0>>v0x>>v0y>>v0z;    getline(cin,buf);
  rxy  = sqrt(x0*x0+y0*y0);
  cout << "# x0 = " << x0
       << "  y0 = " << y0
       << "  z0 = " << z0
       << "  rxy= " << rxy << endl;
  cout << "# v0x= " << v0x
       << "  v0y= " << v0y
       << "  v0z= " << v0z << endl;
  cout << "# Enter t0,tf,dt:\n";
  cin  >> t0 >> tf  >> dt;              getline(cin,buf);
  cout << "# t0= "  << t0  << " tf= " << tf
       << "  dt= "  << dt  << endl;
  if(R   <= 0.0){cerr << "R<=0   \n"; exit(1);}
  if(L   <= 0.0){cerr << "L<=0   \n"; exit(1);}
  if(z0  <  0.0){cerr << "z0<0   \n"; exit(1);}
  if(z0  >    L){cerr << "z0>L   \n"; exit(1);}
  if(rxy >    R){cerr << "rxy>R  \n"; exit(1);}
  if(v0x*v0x+v0y*v0y+v0z*v0z == 0.0)
                {cerr << "v0=0   \n"; exit(1);}
//------------------------------------------------------------
//Initialize
  i  =  0 ;
  nr =  0 ;  nz = 0  ;
  t  = t0 ;
  x  = x0 ;  y  = y0 ;  z  = z0 ;
  vx = v0x;  vy = v0y;  vz = v0z;
  R2 = R*R;
  xc = 0.0; //center of circle which is the projection 
  yc = 0.0; //of the cylinder on the xy plane
  ofstream myfile("Cylinder3D.dat");
  myfile.precision(17);
//------------------------------------------------------------
//Compute:
  while(t <= tf){
    myfile << setw(28) << t  << " "  
	   << setw(28) << x  << " "  
	   << setw(28) << y  << " "  
	   << setw(28) << z  << " "  
	   << setw(28) << vx << " "  
	   << setw(28) << vy << " "
	   << setw(28) << vz << '\n';
    i ++;
    t  = t0 + i*dt;
    x += vx*dt;
    y += vy*dt;
    z += vz*dt;
    if( z <= 0.0 || z > L){vz = -vz; nz++;}
    r2xy = x*x+y*y;
    if( r2xy > R2){
      reflectVonCircle(vx,vy,x,y,xc,yc,R);
      nr++;
    }
  }
  myfile.close();
  cout << "# Number of collisions:\n";
  cout << "# nr= " << nr << " nz= " << nz << endl;
} //main()
//------------------------------------------------------------
//============================================================
//------------------------------------------------------------
void reflectVonCircle(double& vx,double& vy,
		      double& x ,double& y ,
		      const      double& xc,
		      const      double& yc,
		      const      double& R ){
  double theta,cth,sth,vr,vth;
  
  theta = atan2(y-yc,x-xc);
  cth   = cos(theta);
  sth   = sin(theta);
 
  vr    =  vx*cth + vy *sth;
  vth   = -vx*sth + vy *cth;
 
  vx    = -vr*cth - vth*sth; //reflect vr -> -vr
  vy    = -vr*sth + vth*cth;
 
  x     =  xc     + R*cth;   //put x,y on the circle
  y     =  yc     + R*sth;
} //reflectVonCircle()
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
