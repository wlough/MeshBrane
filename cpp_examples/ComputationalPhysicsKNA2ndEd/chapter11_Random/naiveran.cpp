//=============================================
//File: naiveran.cpp
//Program to demonstrate the usage of a modulo
//generator with a bad choice of constants 
//resulting in strong pair correlations between
//generated numbers
//=============================================
static  int    ibm     = 13337;
double naiveran(){
  const int    mult    = 1277;
  const int    modulo  = 131072 ; // equal to 2^17
  const double rmodulo = modulo;

  ibm    *= mult;
  ibm     = ibm % modulo;
  return ibm/rmodulo;
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
