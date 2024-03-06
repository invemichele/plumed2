#! /usr/bin/env python

# gets a RhoSquare.data file and returns a Structure_factor.data with the value of S(k) averaged on each k-shell
# to be explicit: 
#   S(\vec{k})=1/N<\rho_{\vec{k}}\rho_{-\vec{k}}>  while  S(k)=\sum_{|\vec{k}|=k}S(\vec{k})/m_k
# where m_k is the multiplicity of the k-shell (notice that S(\vec{k})=S(-\vec{k}) for us, thus we can use m_k=fwv_sizes[k])
# being \rho_{\vec{k}}= \psi_{\vec{k}} / \rho_0
#
# Thus RhoSquare average * rho_0 / N

import numpy as np
import sys
import os.path 
import argparse
import re #regular expressions
# --> uses the script: bck.meup.sh <--

#parser
parser = argparse.ArgumentParser(description='diffraction pattern as a function of 2theta is calculated only if a lambda is given (-l)')
parser.add_argument('-a',dest='atom_type',type=str,  default="none",required=False,help='atom type, for scattering factor. set to "none" to avoid')
parser.add_argument('-l',dest='Lambda',   type=float,default=-1,    required=False,help='lambda of the incident beam. by default is off')
parser.add_argument('-t',dest='transient',type=int,  default=0,     required=False,help='rows to be skipped')
parser.add_argument('-n',dest='n_rows',   type=int,  default=-1,    required=False,help='how many rows to be considered. default is to use all')
parser.add_argument('-f',dest='in_filename', default='RhoSquare.data',       required=False,help='file containing instantaneous S_k values')
parser.add_argument('-o',dest='out_filename',default='Structure_factor.data',required=False,help='output file')
args = parser.parse_args()

#toggle theta rappresentation
theta_on=False
if args.Lambda!=-1:
  theta_on=True
  from scattering_parameters import scattering_parameters

#load data
data = np.loadtxt(args.in_filename,skiprows=args.transient)

#get k values from first line of plumed output
with open(args.in_filename, 'r') as f:
  head_line = f.readline()
k_values=np.array([float(s) for s in re.findall(r'Sk-([0-9]+.[0-9]+)',head_line)])
if len(k_values)==0:
  sys.stderr.write(' --WARNING: no Sk colvar found, trying with generic CV name\n')
  k_values=np.array([float(s) for s in re.findall(r'.-([0-9\.]+)',head_line)]) #more general
k_num=len(k_values)
if k_num!=len(data[0,:])-1:
  sys.exit(' --ERROR: unexpected format in input file')
#get n_rows
n_rows=len(data[:,0])
if args.n_rows!=-1:
  n_rows=args.n_rows

#calculate Sk
Sk=np.zeros(k_num)
Sk_std=np.zeros(k_num)
for k in range(k_num):
  Sk[k]=np.mean(data[:n_rows,k+1])
  Sk_std[k]=np.std(data[:n_rows,k+1])

#calculate 2theta representation
if theta_on:
  theta2_values=2*np.arcsin(args.Lambda/(4*np.pi)*k_values) #in radiandts
  S_theta2=Sk*(1+np.cos(theta2_values)**2)/np.cos(theta2_values/2)/np.sin(theta2_values/2)**2 #Lorentz polarization factor
  theta2_values*=180/np.pi #in degrees
#add scattering factor
  if (args.atom_type!="none"):
    sp=scattering_parameters(args.atom_type)
    appo=(k_values/(4*np.pi))**2
    S_theta2*=(sp[0]*np.exp(-sp[1]*appo)+sp[2]*np.exp(-sp[3]*appo)+sp[4]*np.exp(-sp[5]*appo)+sp[6]*np.exp(-sp[7]*appo)+sp[8])**2

#backup outputfile if already exists
if os.path.isfile(args.out_filename):
#  os.system("bck.meup.sh -i "+args.out_filename)
  os.system("./bck.meup.sh -i "+args.out_filename)

#save in new file
head='k  Structure_factor   sdt_dev'
comment='  #transient=%d, n_rows=%d'%(args.transient,n_rows)
form='%10f %10f %10f'
outdata=np.c_[k_values,Sk,Sk_std]
if theta_on:
  head+='    2theta    S(2theta)' 
  comment+=', atoms="%s", lambda=%g'%(args.atom_type,args.Lambda)
  form+=' %10f %10f'
  outdata=np.c_[outdata,theta2_values,S_theta2]
np.savetxt(args.out_filename,outdata,header=head+comment,fmt=form)
