//======================================================
//file: boot_bin.cpp
//
//Compile with:
//g++ -std=c++11 boot_bin.cpp MIXMAX/mixmax.cpp -o boot_bin
//======================================================
//bootstrap function: you can use this module in
//any of your programs.
//------------------------------------------------------
//                Simple binning
//                --------------
//Bins the data in BINS bins. Each bin has binw=ndat/BINS elements.
//. From each bin we collect SAMPLES "measurements":
//   + We pick randomly binw data points from each bin
//   + O,O^2,chi are computed from this data.
//   + O,chi are averaged over the number of SAMPLES, give Ob,chib, the
//     value of the observable in the bin
//. Ob and chib are averaged over the number of bins (=avO,avchi) and
//  we compute the errors as if they are independent measurements (=erO,erchi)
//
//Gives correct results for small number of bins for chi but as the
//number of bins increases, the results become erroneous.
//------------------------------------------------------
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <random>
#include <unistd.h>
#include <libgen.h>
using namespace std;
int maxdat,SAMPLES,BINS;
string prog;
void bootstrap(int,int&,int&,
               double *,double &,
               double &,double &,double &);
void   get_the_options(int,char**);
void   usage (char**);
void   locerr(string);

int    main(int argc,char** argv){
  int    ndat;
  double O,dO,chi,dchi;
  double *x;
  prog.assign((char *)basename(argv[0]));//name of program
  maxdat=2000000;SAMPLES=1000;BINS=20;
  get_the_options(argc,argv);
  x = new double[maxdat];
  ndat=0;
  while((ndat<=maxdat) && (cin >> x[ndat++]));
  ndat--;
  if(    ndat==maxdat)
    cerr << prog
         << ": Warning: read ndat= "    << ndat
         << " and reached the limit: "  << maxdat  << endl;
  bootstrap(ndat,SAMPLES,BINS,x,O,dO,chi,dchi);
  cout << "# NDAT = "                   << ndat
       << " data. SAMPLES = "           << SAMPLES << endl;
  cout << "# BINS = "                   << BINS    << endl;
  cout << "# <o>, chi= (<o^2>-<o>^2)\n";
  cout << "# <o> +/- err                             chi +/- err\n";
  cout.precision(17);
  cout << O <<" "<< dO <<" "<< chi <<" "<< dchi    << endl;
}//main()
//======================================================
#include "MIXMAX/mixmax.hpp"
int seedby_urandom();
void bootstrap(int    ndat,int   & samples,int & bins,
               double* x  ,double& avO    ,
               double& erO,double& avchi,double& erchi){
  int    i,j,k,m,b,binw;
  double *O,*O2,*chi,*Ob,*chib,avOall;
  mixmax_engine mxmx(0,0,0,1);
  uniform_real_distribution<double> drandom;
  Ob   = new double[bins   ]();
  chib = new double[bins   ]();
  O    = new double[samples]();//() initializes to 0
  O2   = new double[samples]();
  chi  = new double[samples]();
  mxmx.seed(seedby_urandom());
  //Compute the real avO:
  avOall  = 0.0;
  for(i=0;i<ndat;i++) avOall += x[i];
  avOall /= ndat;
  //We separate data in bins bins. We cutoff data that is not multiple of bins
  ndat = (ndat/bins)*bins;
  //Number of data in each bin
  binw = ndat/bins;
  for(i=0;i<bins;i++){
    for(j=0;j<samples;j++){
      O[j] = 0.0;O2[j] = 0.0;
      for(m=0;m<binw;m++){
        k=int(binw*drandom(mxmx))+i*binw;
        O [j] += x[k];
        O2[j] += x[k]*x[k];
      }
      O   [j] /= binw;
      O2  [j] /= binw;
      chi [j]  = O2 [j] - O[j]*O[j];
      Ob  [i] += O  [j];
      chib[i] += chi[j];
    }//for(j=0;j<samples;j++)
    Ob    [i] /= samples;
    chib  [i] /= samples;
  }
  //Compute averages:
  avO=0.0;avchi=0.0;
  for(i=0;i<bins;i++){
    avO += Ob[i]; avchi += chib[i];
  }
  avO /= bins;avchi /= bins;
  //Compute errors:
  erO = 0.0; erchi = 0.0;
  for(i=0;i<bins;i++){
    erO   += (Ob  [i]-avO  )*(Ob  [i]-avO  );
    erchi += (chib[i]-avchi)*(chib[i]-avchi);
  }
  erO /= bins*(bins-1);erchi /= bins*(bins-1);
  erO  = sqrt(erO)    ;erchi  = sqrt(erchi);
  //Compute the real avO:
  avO = avOall;
  
  delete [] Ob;
  delete [] chib;
  delete [] O;
  delete [] O2;
  delete [] chi;
}//bootstrap()
//=============================================
// Use /dev/urandom for more randomness.
#include <fcntl.h>
int seedby_urandom(){
  int ur,fd,ir;
  fd = open("/dev/urandom", O_RDONLY);
  ir = read (fd,&ur,sizeof(int));
  close(fd);
  return (ur>0)?ur:-ur;
}
//======================================================
void locerr(string errmes){
  cerr << prog <<": "<< errmes <<"Exiting....."<< endl;
  exit(1);
}
//======================================================
#define OPTARGS  "?hs:d:b:"
void get_the_options(int argc,char **argv){

  int c,errflg = 0;
  while (!errflg &&
         (c = getopt(argc, argv, OPTARGS)) != -1){
    switch(c){
    case 's':
      SAMPLES = atoi(optarg);
      break;
    case 'b':
      BINS    = atoi(optarg);
      break;
    case 'd':
      maxdat  = atoi(optarg);
      break;
    case 'h':
      errflg++;/*call usage*/
      break;
    default:
      errflg++;
    }/*switch*/
    if(errflg) usage(argv);
  }/*while...*/
}//get_the_options()
//======================================================
void usage(char **argv){
  cerr << "\
Usage: " << prog << "  [options] <file>                  \n\
       -s  : No. samples                                 \n\
       -b  : No. of bins. Samples have size ndata/BINS.  \n\
       -d  : Give the maximum number of data points read.\n\
Computes <o>, chi= (<o^2>-<o>^2)                         \n\
Data is in one column from stdin." << endl;
  exit(1);
}/*usage()*/
//======================================================
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

