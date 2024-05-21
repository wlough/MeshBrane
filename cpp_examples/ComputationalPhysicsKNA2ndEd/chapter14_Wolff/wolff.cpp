//padd = 1 - exp(-2*beta)
#include "include.h"
#include <new>  // bad_alloc
void wolff(){

  int  cseed,nstack,sold,snew,scluster,nn;
  int *stack;
  int  ncluster;

  //ask for the stack memory
  try{stack = new int[N];}
  catch(bad_alloc& stack_failed){
    string err(stack_failed.what());
    err = "allocation failure for stack in wolff(). " + err;
    locerr(err);
  }

  //choose the seed  spin for the cluster,
  //put it on the stack and flip it
  cseed    =  N*drandom(mxmx);
  stack[0] =    cseed;
  nstack   =  1;        //the stack has 1 member, the seed
  sold     =  s[cseed];
  snew     = -s[cseed]; //the new spin value of the  cluster
  s[cseed] =  snew;     //we flip all new members of cluster
  ncluster =  1;        //size of cluster=1
  //start loop on spins on the stack:
  while(nstack>0){
    //Pull a site off the stack:
    scluster  =  stack[--nstack];//value of --nstack is **after** decrement
    //check its four neighbours:
    if((nn    =  scluster+XNN)>=N) nn -= N;
    if(s[nn]  == sold)
      if(drandom(mxmx)<padd){
        stack[nstack++] = nn;//value of nstack++ is **before** increment
        s[nn] = snew; //flip the spin of cluster
        ncluster++;
      }
    if((nn    =  scluster-XNN)<0) nn += N;
    if(s[nn]  == sold)
      if(drandom(mxmx)<padd){
        stack[nstack++] = nn;//value of nstack++ is **before** increment
        s[nn] = snew; //flip the spin of cluster
        ncluster++;
      }
    if((nn    =  scluster+YNN)>=N) nn -= N;
    if(s[nn]  == sold)
      if(drandom(mxmx)<padd){
        stack[nstack++] = nn;//value of nstack++ is **before** increment
        s[nn] = snew; //flip the spin of cluster
        ncluster++;
      }
    if((nn    =  scluster-YNN)<0) nn += N;
    if(s[nn]  == sold)
      if(drandom(mxmx)<padd){
        stack[nstack++] = nn;//value of nstack++ is **before** increment
        s[nn] = snew; //flip the spin of cluster
        ncluster++;
      }
  }/*while(nstack>0)*/
  cout << "#clu " << ncluster << '\n';
  delete [] stack;
}/*wolff()*/
// Code from the book by:
// M.E.J. Newman and G.T. Barkema, Monte Carlo  Methods in Statistical Physics, Clarendon Press, Oxford (2002).

