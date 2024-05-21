//===========================================================
//file: wellInfSq.cpp
//
//Functions used in RKSTEP routine. Here:
//f1 = psip(x) = psi(x)'
//f2 = psip(x)'= psi(x)''
//
//All one has to set is V, the potential     
//===========================================================
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <cmath>
using namespace std;
extern double energy;
//-------- trivial function: derivative of psi      
double
f1(const double& x, const double& psi,const double& psip){
  return psip;
}
//===========================================================
//-------- the second derivative of wavefunction:
//psip(x)' = psi(x)'' = -(E-V) psi(x)    
double
f2(const double& x, const double& psi,const double& psip){
  double V;
  //------- potential, set here:
  V  = 0.0;
  //------- Schroedinger eq: RHS
  return (V-energy)*psi;
}
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
