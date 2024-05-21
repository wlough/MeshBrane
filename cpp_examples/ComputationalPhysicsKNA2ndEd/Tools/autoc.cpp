//======================================================
//file: autoc.cpp
//
//Compile with:
//g++ -O2 autoc.cpp -o autoc
//======================================================
//Autocorrelation function: you can use this function in
//any of your programs.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <unistd.h>
#include <libgen.h>
using namespace std;
string prog;
int    NMAX,tmax;
double rho (double *,int ,int );
void   get_the_options(int,char**);
void   usage (char**);
void   locerr(string);
int    main(int argc,char** argv){
  double *r,*tau,*x;
  double norm;
  int    i,ndat,t,tcut;
  prog.assign((char *)basename(argv[0]));//name of program
  //------------------------------------------------------
  //Default values for max number of data and max time for
  //rho and tau:
  NMAX=2000001;tmax=1000;//NMAX=2e6 requires ~ 2e6*8=16MB
  get_the_options(argc,argv);
  x = new double[NMAX+1];
  ndat=0;
  while((ndat<=NMAX) && (cin >> x[ndat++]));
  ndat--;
  if(ndat == NMAX)
    cerr << prog
         << ": Warning: read ndat= "   << ndat
         << " and reached the limit: " << NMAX << endl;
  //We decrease tmax if it is comparable or larger of ndat
  if(tmax > (ndat/10)) tmax = ndat/10;
  //r[t] stores the values of the autocorrelation
  //function rho(t)
  r = new double[tmax];
  for(t=0;t<tmax;t++) r[t]  = rho(x,ndat,t);      //rho(t)
  norm=1.0/r[0];for(t=0;t<tmax;t++)r[t] *= norm;  //normalize r[0]=1
  //tau[t] stores integrated autocorrelation times with tcut=t
  tau = new double[tmax];
  for(tcut=0;tcut<tmax;tcut++){
    tau[tcut] = 0.0;
    for(t=0;t<=tcut;t++) tau[tcut] += r[t];//sum of r[t]
  }
  //Output:
  cout << "# ==========================================================" << endl;
  cout << "#      Autoc function rho and integrated autoc time tau     " << endl;
  cout << "# ndat= "<< ndat << "  tmax= " << tmax                        << endl;
  cout << "# t         rho(t)              tau(tcut=t)                 " << endl;
  cout << "# ==========================================================" << endl;
  for(t=0;t<tmax;t++)
    cout << t << " " << r[t] << " " << tau[t] << '\n';
}//main()
//======================================================
double rho(double *x,int ndat,int t){
  int n,t0;
  double xav0=0.0,xavt=0.0,r=0.0;
  n=ndat-t;
  if(n<1)locerr("rho: n<1");
  //Calculate the two averages: xav0= <x>_0 and xavt= <x>_t
  for(t0=0;t0<n;t0++){
    xav0 += x[t0];
    xavt += x[t0+t];
  }
  xav0/=n;xavt/=n; //normalize the averages to number of data
  //Calculate the t-correlations:
  for(t0=0;t0<n;t0++)
    r += (x[t0]-xav0)*(x[t0+t]-xavt);
  r/=n;            //normalize the averages to number of data
  return r;
}
//======================================================
void locerr(string errmes){
  cerr << prog <<": "<< errmes <<"Exiting....."<< endl;
  exit(1);
}
//======================================================
#define OPTARGS  "?h1234567890.t:n:"
void get_the_options(int argc,char **argv){

  int c,errflg = 0;
  while (!errflg &&
         (c = getopt(argc, argv, OPTARGS)) != -1){
    switch(c){
    case 'n':
      NMAX = atoi(optarg);
      break;
    case 't':
      tmax = atoi(optarg);
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
Usage: " << prog << " [-t <maxtime>] [-n <ndata>]     \n\
       Reads data from stdin (one column) and computes\n\
       autocorrelation function and integrated        \n\
       autocorrelation time." << endl;
  exit(1);
}/*usage()*/
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
