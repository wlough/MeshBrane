#!/bin/bash

L=10
x0=0.0
v0=1.0
t0=0.0
tf=95
ns=(1 2 4 5) #versions of code
#dts=(0.1 0.05 0.025 0.0125 0.01 0.005 0.0025 0.00125) 
dts=(0.050000 0.025000 0.012500 0.006250 0.003125 0.001563 0.000781 0.000391 0.000195 0.000098 0.000049 0.000024 0.000012)
prog=$(basename $0)                    # name of this file
progn=$(basename $prog .${prog##*.})   # name of this file without the .sh extension
log=$progn.log                         # log  file
dat=$progn.dat                         # data file

date                                                        >>  $log
echo "#===================================================" >>  $log
for n in ${n[@]};do
 echo "# Compiling box1D_${n}.cpp                         " >>  $log
 g++ -O3  box1D_${n}.cpp -o box$n                          &>>  $log
done

# Run for default values:
for dt in ${dts[@]};do
 echo "dt = $dt"
 results=($tf $dt)         # store each lines of results here
 echo "# Running dt = $dts                               " >>  $log
 for n in ${ns[@]};do
  ./box$n <<EOF >> log
$L
$x0 $v0
$t0 $tf $dt
EOF
  results=($results `tail -n 1 box1D_${n}.dat|awk '{print $1,$2}'`)
  \mv -f box1D_${n}.dat box1D_${n}_dt${dt}.dat
 done
 echo $results >> $dat

done
#  ---------------------------------------------------------------------
#  Copyright by Konstantinos N. Anagnostopoulos (2004-2014)
#  Physics Dept., National Technical University,
#  konstant@mail.ntua.gr, www.physics.ntua.gr/~konstant
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, version 3 of the License.
#  
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#  
#  You should have received a copy of the GNU General Public Liense along
#  with this program.  If not, see <http://www.gnu.org/licenses/>.
#  -----------------------------------------------------------------------

