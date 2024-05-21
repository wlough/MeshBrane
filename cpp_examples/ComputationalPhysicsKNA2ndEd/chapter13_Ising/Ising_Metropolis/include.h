//============== include.h   ==================
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
using namespace std;

#include "MIXMAX/mixmax.hpp"

extern int    L;
extern int    N;
extern int    XNN;
extern int    YNN;
extern int    *s;
extern int    seed,nsweep,start;
extern double beta;
extern double prob[5];
extern double acceptance;
extern mixmax_engine mxmx;
extern uniform_real_distribution<double> drandom;

int E(),M();
void init(int,char** ), met(), measure();
void simmessage(ostream&), locerr(string);
void usage(char **);
void get_the_options(int,char**),endsim();
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
