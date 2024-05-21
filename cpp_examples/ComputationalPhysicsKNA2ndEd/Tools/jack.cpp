//======================================================
//file: jack.cpp
//
//Compile with:
//g++ -O2 jack.cpp -o jack
//======================================================
//jackknife function: you can use this function in
//any of your programs.
//------------------------------------------------------
//Note: If ndat % jack //= 0, then the algorithm works fine.
//      Each bin has ndat-binw data, bins never cross
//      remainder of data (i >= (jack*binw))
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#include <libgen.h>
using namespace std;
int maxdat,JACK;
string prog;
void jackknife(int,int&,double *,double &,
               double &,double &,double &);
void   get_the_options(int,char**);
void   usage (char**);
void   locerr(string);

int    main(int argc,char** argv){
  int    ndat;
  double O,dO,chi,dchi;
  double *x;
  prog.assign((char *)basename(argv[0]));//name of program
  maxdat=2000000;JACK=10;
  get_the_options(argc,argv);
  x = new double[maxdat];
  ndat=0;
  while((ndat<=maxdat) && (cin >> x[ndat++]));
  ndat--;
  if(    ndat==maxdat)
    cerr << prog
         << ": Warning: read ndat= "    << ndat
         << " and reached the limit: "  << maxdat  << endl;
  jackknife(ndat,JACK,x,O,dO,chi,dchi);
  cout << "# NDAT = "<<ndat<<" data. JACK = "<<JACK<< endl;
  cout << "# <o>, chi= (<o^2>-<o>^2)\n";
  cout << "# <o> +/- err                             chi +/- err\n";
  cout.precision(17);
  cout << O <<" "<< dO <<" "<< chi <<" "<< dchi    << endl;
}//main()
//======================================================
void jackknife(int ndat,int& jack,double* x,double& avO,
               double& erO,double& avchi,double& erchi){
  int    i,j,binw,bin;
  double *O,*chi;
  O    = new double[jack]();//() initializes to 0
  chi  = new double[jack]();
  binw = ndat/jack;
  if(binw<1)locerr("jackknife: binw < 1");
  //----------------------------------------
  //average value:
  for(  i=0;i<ndat;i++)
    for(j=0;j<jack;j++){
      if( (i/binw) != j ){ //then we add this data point to the bin
        O [j] += x[i];
      }
    }
  for(j=0;j<jack;j++) O  [j] /= (ndat-binw); //average in each bin
  //----------------------------------------
  //susceptibility:
  for(  i=0;i<ndat;i++)
    for(j=0;j<jack;j++){
      if( (i/binw) != j ){ //then we add this data point to the bin
	chi[j] += (x[i]-O[j])*(x[i]-O[j]);
      }
    }
  for(j=0;j<jack;j++) chi[j] /= (ndat-binw); //average in each bin
  //----------------------------------------
  //Compute averages:
  avO = 0.0; avchi = 0.0;
  for(j=0;j<jack;j++){
    avO += O[j]; avchi += chi[j];
  }
  avO /= jack; avchi /= jack;
  //----------------------------------------
  //Compute errors:
  erO = 0.0; erchi = 0.0;
  for(j=0;j<jack;j++){
    erO   += (O  [j]-avO  )*(O  [j]-avO  );
    erchi += (chi[j]-avchi)*(chi[j]-avchi);
  }
  erO = sqrt(erO); erchi = sqrt(erchi);
  delete [] O;
  delete [] chi;
  //Note: If ndat % jack != 0, then the algorithm works fine.
  //      Each bin has ndat-binw data, bins never cross
  //      remainder of data (i >= (jack*binw))
}//jackknife()
//======================================================
void locerr(string errmes){
  cerr << prog <<": "<< errmes <<"Exiting....."<< endl;
  exit(1);
}
//======================================================
#define OPTARGS  "?hj:d:"
void get_the_options(int argc,char **argv){

  int c,errflg = 0;
  while (!errflg &&
         (c = getopt(argc, argv, OPTARGS)) != -1){
    switch(c){
    case 'j':
      JACK   = atoi(optarg);
      break;
    case 'd':
      maxdat = atoi(optarg);
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
Usage: " << prog << "  [options]                       \n\
       -j : No. jack groups                            \n\
       -d : Give the maximum number of data points read\n\
Computes <o>, chi= (<o^2>-<o>^2)                       \n\
Data is in one column from stdin.\n" << endl;
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

