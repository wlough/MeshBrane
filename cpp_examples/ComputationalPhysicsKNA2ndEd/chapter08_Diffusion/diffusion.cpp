//=======================================================
// 1-dimensional Diffusion Equation with simple
// Dirichlet boundary conditions u(0,t)=u(1,t)=0
// 0<= x <= 1 and 0<= t <= tf
// 
// We set initial condition u(x,t=0) that satisfies 
// the given boundary conditions. 
// Nx is the number of points in spatial lattice:
// x = 0 + i*dx, i=0,...,Nx-1 and dx = (1-0)/(Nx-1)
// Nt is the number of points in temporal lattice:
// t = 0 + j*dt, j=0,...,Nt-1 and dt = (tf-0)/(Nt-1)
//
// u(x,0) = sin(pi*x) tested against analytical solution
// u(x,t) = sin(pi*x)*exp(-pi*pi*t)
//
//=======================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
int main(){
  const int    P  = 100000;
  const double PI = 2.0*atan2(1.0,0.0);
  double u[P], d2udx2[P];
  double t,x,dx,dt,tf,courant;
  int    Nx,Nt,i,j;
  string buf;
  //Input:
  cout  << "# Enter: Nx, Nt, tf: (P= " << P
        << "  Nx must be < P)"         << endl;
  cin   >> Nx >> Nt >> tf;    getline(cin,buf);
  if(Nx >= P){cerr  << "Nx >= P\n";    exit(1);}
  if(Nx <= 3){cerr  << "Nx <= 3\n";    exit(1);}
  if(Nt <= 2){cerr  << "Nx <= 2\n";    exit(1);}
  //Initialize:
  dx     = 1.0/(Nx -1);
  dt     = tf /(Nt -1);
  courant= dt /(dx*dx);
  cout  << "# 1d Diffusion Equation: 0<=x<=1, 0<=t<=tf\n";
  cout  << "# dx= " << dx << " dx= " << dt
        << "  tf= " << tf                  << endl;
  cout  << "# Nx= " << Nx << " Nt= " << Nt << endl;
  cout  << "# Courant Number= " << courant << endl;
  if(courant > 0.5) cout  << "# WARNING: courant > 0.5\n";
  ofstream  myfile("d.dat");
  myfile.precision(17);
  //------------------------------------------------------
  // Initial condition at t=0
  // u(x,0) = sin( pi x)
  for(i=0;i<Nx;i++){
    x      = i*dx;
    u[i]   = sin(PI*x);
  }
  u[0   ]  = 0.0;
  u[Nx-1]  = 0.0;
  for(i=0;i<Nx;i++){
    x      = i*dx;
    myfile << 0.0 << " " << x << " " << u[i] << '\n';
  }
  myfile   << " \n";  
  //------------------------------------------------------
  // Calculate time evolution:
  for(j=1;j<Nt;j++){
    t = j*dt;
    // Second derivative:
    for(i=1;i<Nx-1;i++)
      d2udx2[i] = courant*(u[i+1]-2.0*u[i]+u[i-1]);
    // Update:
    for(i=1;i<Nx-1;i++)
      u[i]     += d2udx2[i];
    for(i=0;i<Nx;i++){
      x         = i*dx;
      myfile << t << " " << x << " " << u[i] << '\n';
    }
    myfile   << " \n";
  }//for(j=1;j<Nt;j++)
  myfile.close();
}//main()
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
