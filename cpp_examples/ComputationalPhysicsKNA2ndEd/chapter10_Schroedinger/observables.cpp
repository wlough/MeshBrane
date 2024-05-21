//===========================================================
//
// File observables.cpp
// Compile: g++ observables.cpp -o o
// Usage:   ./o <psi.dat>
//
// Read in a file with a wavefunction in the format of psi.dat:
// # E= <energy> ....
// x1  psi(x1)
// x2  psi(x2)
// ............
//
// Outputs expectation values:
// normalization Energy <x> <p> <x^2> <p^2> Dx Dp DxDp
// where Dx = sqrt(<x^2>-<x>^2) Dp = sqrt(<p^2>-<p>^2)
//  DxDp = Dx * Dp
//
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
//--------------------------------------------------------
double integrate(double* psi ,
           const double& dx  ,const int   & Nx      );
//--------------------------------------------------------

int main(int argc, char **argv){
  const int P = 50000;
  int    Nx,i;
  double xstep[P],psi[P],obs[P];
  double xav, pav, x2av, p2av, Dx, Dp, DxDp,energy,h,norm;
  string buf;
  char   *psifile;

  if(argc != 2){
    cerr << "Usage: " << argv[0] << "  <filename>\n";
    exit(1);
  }
  psifile = argv[1];
  ifstream ifile(psifile);
  if(! ifile){
    cerr << "Error reading from file " << psifile << endl;
    exit(1);
  }
  cout << "# reading wavefunction from file: " << psifile << endl;
  ifile >> buf >> buf >> energy; getline(ifile,buf);
  //-------------------------------------------------------
  //Input data: psi[x]
  Nx = 0;
  while(ifile >> xstep[Nx] >> psi[Nx]){
    Nx++;
    if(Nx == P){cerr << "Too many points\n";exit(1);}
  }
  if(Nx % 2 == 0) Nx--;
  h = (xstep[Nx-1]-xstep[0])/(Nx-1);
  //-------------------------------------------------------
  //Calculate:
  //---------- norm:
  for(i=0;i<Nx;i++) obs[i] = psi[i]*psi[i];
  norm=  integrate(obs,h,Nx);
  //---------- <x> :
  for(i=0;i<Nx;i++) obs[i] = psi[i]*psi[i]*xstep[i];
  xav =  integrate(obs,h,Nx)/norm;
  //---------- <p>/i: use quadratic polynomial at boundaries
  //                  (see Derivatives.nb)
  //obs[0] = psi[0]*(psi[1]-psi[0])/h //Naive derivative
  obs  [0] = psi[0]*(-3.0*psi[0]+4.0*psi[1]-psi[2])/(2.0*h);
  for(i=1;i<Nx-1;i++)
    obs[i] = psi[i]*(psi[i+1]-psi[i-1])/(2.0*h);
  //obs[Nx-1]=psi[Nx-1]*(psi[Nx-1]-psi[Nx-2])/h; //naive
  obs  [Nx-1]=psi[Nx-1]
    *(psi[Nx-3]-4.0*psi[Nx-2]+4.0*psi[Nx-1])/(2.0*h);
  pav = -integrate(obs,h,Nx)/norm;
  //---------- <x^2>:
  for(i=0;i<Nx;i++) obs[i] = psi[i]*psi[i]*xstep[i]*xstep[i];
  x2av=  integrate(obs,h,Nx)/norm;
  //---------- <p^2>:
  //obs[0] = psi[0]*(psi[2]-2.0*psi[1]+psi[0])/(h*h); //naive
  obs[0] = psi[0] * //better: O(h^2) (See Derivatives.nb)
    (2.0*psi[0]-5.0*psi[1]+4.0*psi[2]-psi[3])/(h*h);
  for(i=1;i<Nx-1;i++)
    obs[i] = psi[i]*(psi[i+1]-2.0*psi[i]+psi[i-1])/(h*h);
  //obs[Nx-1] = psi[Nx-1] * //naive
      //(psi[Nx-1]-2.0*psi[Nx-2]+psi[Nx-3])/(h*h);
  obs[Nx-1] = psi[Nx-1] * //better: O(h^2) (See Derivatives.nb)
    (2.0*psi[Nx-1]-5.0*psi[Nx-2]+4.0*psi[Nx-3]-psi[Nx-4])/(h*h);
  p2av= -integrate(obs,h,Nx)/norm;
  //---------- Dx
  Dx = sqrt(x2av - xav*xav);
  //---------- Dp
  Dp = sqrt(p2av - pav*pav);
  //---------- Dx.Dp
  DxDp = Dx*Dp;
  //Print results:
  cout.precision(17);
  cout << "# norm E <x> <p>/i <x^2> <p^2> Dx Dp DxDp\n";
  cout << norm   << " "
       << energy << " "
       << xav    << " "
       << pav    << " "
       << x2av   << " "
       << p2av   << " "
       << Dx     << " "
       << Dp     << " "
       << DxDp   << endl;
    
}//main()
//========================================================
//Simpson's rule to integrate psi(x).
//For n intervals of width dx (n even)
//Simpson's rule is:
//int(f(x)dx) = 
//(dx/3)*(f(x_0)+4 f(x_1)+2 f(x_2)+...+4 f(x_{n-1})+f(x_n))
//
//Input:   Discrete values of function psi[Nx], Nx is odd
//         Integration step dx
//Returns: Integral(psi(x) dx)
//========================================================
double integrate(double* psi,
           const double& dx ,const int& Nx){
  double Integral;
  int    i;
  //zeroth order point:
  i         = 0;
  Integral  = psi[i];
  //odd  order points:
  for(i=1;i<=Nx-2;i+=2) Integral += 4.0*psi[i];
  //even order points:
  for(i=2;i<=Nx-3;i+=2) Integral += 2.0*psi[i];
  //last point:
  i         = Nx-1;
  Integral += psi[i];
  //measure normalization:
  Integral *=dx/3.0;

  return Integral;
}//integrate()
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
