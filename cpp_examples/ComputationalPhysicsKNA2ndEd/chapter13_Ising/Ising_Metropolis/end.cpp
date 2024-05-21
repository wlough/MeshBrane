//============== end.cpp      ==================
#include "include.h"
#include <cstdio>
void endsim(){
  rename("conf","conf.old");
  ofstream conf("conf");
  conf << "# Configuration of 2d Ising model on square lattice. Parameters (N=Lx*Ly) and s[N]\n";
  conf << "Lx= "<<L<<" Ly= "<< L << " beta= " << beta
       << " seed= " << int(100000*drandom(mxmx)) << "\n"; //only for compatibility with old "conf" format
  for(int i=0;i<N;i++) conf << s[i] << '\n';
  conf << mxmx << endl;
  conf.close();
}//endsim()
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
