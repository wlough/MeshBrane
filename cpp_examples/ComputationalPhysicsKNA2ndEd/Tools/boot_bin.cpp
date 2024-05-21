//======================================================
//file: boot_bin.cpp
//
//Compile with:
//g++ -std=c++11 boot_bin.cpp MIXMAX/mixmax.cpp -o boot_bin
//======================================================
//bootstrap function: you can use this module in
//any of your programs.
//------------------------------------------------------
//                Jackknife binning
//                -----------------
//. Separates the data in BINS jackknife bins
//. For each bin collects SAMPLES bootstrap measurements:
//     + Picks ndat-binw random values from each bin
//     + The average O,O^2, and computed chi is the value of one sample
//. The samples of each bin are averaged to give the value of each jackknife bin
//. Bins are averaged and jackknife errors calculated
//Gives correct results. For 100000 data for |M|, error is
//independent of number of bins for BINS=5, ..., 1000
//(Ising 2d data, L=12, beta=0.36, tau_autoc ~ 8)
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
  int    i,j,k,m,n,b,binw,ndat_in_bin;
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
  //Number of data in each bin
  binw         = ndat/bins;
  ndat_in_bin  = ndat - binw;
  for(i=0;i<bins;i++){
    //For each jackknife bin we collect samples samples
    for(j=0;j<samples;j++){
      O[j] = 0.0;O2[j] = 0.0;m=0;
      //Each samples consist of ndat_in_bin "measurements"
      while(m<ndat_in_bin){
        k=int(ndat*drandom(mxmx));
        if((k/binw) != i){//if k not in the i_th bin, consider the data: 
          O [j] += x[k];  //(lazy, improve with computing without wasting)
          O2[j] += x[k]*x[k];
          m++;
        }
      }//while(m<ndat_in_bin)
      //O,O2,chi are the result of each sample
      O   [j] /= ndat_in_bin;
      O2  [j] /= ndat_in_bin;
      chi [j]  = O2 [j] - O[j]*O[j];
      //Ob,chib are the jackknife measurements:
      Ob  [i] += O  [j];
      chib[i] += chi[j];
    }//for(j=0;j<samples;j++)
    //Ob,chib are the jackknife measurements:
    Ob    [i] /= samples;
    chib  [i] /= samples;
  }
  //Compute jackknife averages:
  avO=0.0;avchi=0.0;
  for(i=0;i<bins;i++){
    avO += Ob[i]; avchi += chib[i];
  }
  avO /= bins;avchi /= bins;
  //Compute jackknife errors:
  erO = 0.0; erchi = 0.0;
  for(i=0;i<bins;i++){
    erO   += (Ob  [i]-avO  )*(Ob  [i]-avO  );
    erchi += (chib[i]-avchi)*(chib[i]-avchi);
  }
  //erO /= bins*(bins-1);erchi /= bins*(bins-1);
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
Data is in one column from stdin.                        \n\
---------------------------------\n\
Jackknife binning: Separates the data in BINS bins of binw=ndat/BINS.\n\
BINS Jackknife bins with ndat-binw data are constructed.  For each bin\n\
collects SAMPLES bootstrap measurements. Each measurement consists of\n\
ndat-binw random values from each jackknife bin.In the end, the\n\
SAMPLES measurements are averaged giving the result for each bin.\n\
Then this is jackknife-averaged and jackknife errors are computed.\
" << endl;
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

