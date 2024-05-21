//============== init.cpp    ==================
// file init.cpp
// init(start): start = 0: cold start
//              start = 1: hot  start
//              start =-1: old  configuration
//=============================================
#include "include.h"
//Global variables:
int    L;
int    N;
int    XNN;
int    YNN;
int    *s;
int    seed;
int    nsweep;
int    start;
double beta;
double prob[5];
double acceptance = 0.0;
mixmax_engine mxmx(0,0,0,1);
uniform_real_distribution<double> drandom;

void init(int argc, char **argv){
  int    i;
  int    OL=-1;
  double obeta=-1.0;
  string buf;
  //define parameters from options:
  L=-1;beta=-1.;nsweep=-1;start=-1;seed=-1;
  get_the_options(argc,argv);
  if( start == 0 || start == 1){
    if(L     <  0 )locerr("L has not been set."   );
    if(seed  <  0 )locerr("seed has not been set.");
    if(beta  <  0 )locerr("beta has not been set.");
    //derived parameters
    N=L*L; XNN=1; YNN = L;
    //allocate memory space for spins:
    s = new int[N];
    if (!s) locerr("allocation failure for s[N]");
  }//if( start == 0 || start == 1)
  if(start  < 0 )locerr("start  has not been set.");
  if(nsweep < 0 )locerr("nsweep has not been set.");
  //Initialize probabilities:
  for(i=2;i<5;i+=2) prob[i] = exp(-2.0*beta*i);
  //--------------------------------------------
  //Initial configuration: cold(0),hot(1),old(2)
  //--------------------------------------------
  switch(start){
  //--------------------------------------------
  case 0://cold start
    simmessage(cout);
    mxmx.seed (seed);
    for(i=0;i<N;i++) s[i]=1;
    break;
  //--------------------------------------------
  case 1://hot start
    simmessage(cout);
    mxmx.seed (seed);
    for(i=0;i<N;i++){
      if(drandom(mxmx) < 0.5)
        s[i] =  1;
      else
        s[i] = -1;
    }
    break;
  //--------------------------------------------
  case 2:{//old configuration
    ifstream conf("conf");
    if(!conf.is_open()) locerr("Configuration file conf not found");
    getline(conf,buf);//discard comment line
    conf >> buf >> OL >> buf >> OL >> buf >> obeta;getline(conf,buf);
    if( L <  0  ) L=OL;
    if( L != OL ) locerr("Given L different from the one read from conf");
    N=L*L; XNN=1; YNN = L;
    if(beta < 0.) beta = obeta;
    //allocate memory space for spins:
    s = new int[N];
    if (!s) locerr("allocation failure in s[N]");
    for(i=0;i<N;i++)
      if((!(conf >> s[i])) ||((s[i] != -1) && (s[i] != 1)))
        locerr("conf ended before reading s[N]");
    if(seed >=0 ) mxmx.seed (seed);
    if(seed < 0 )
      if(!(conf >> mxmx))
        locerr("conf ended before reading mixmax state");
    conf.close();
    simmessage(cout);
    break;
  }// use {...} because newly declared variable
   // ifstream conf must remain in scope
   // otherwise it will be visible in default:, but
   // without the initialization!
  //--------------------------------------------
  default:
    cout << "start= "           << start
         << " not valid. Exiting....\n";
    exit(1);
    break;
  }//switch(start)
  //--------------------------------------------
}//init()
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
