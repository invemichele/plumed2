/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2014 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "DensityFourierComp.h"
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"

#include <cmath>
#include <algorithm> //std::stable_sort
#include <sstream>   //std::ostringstream

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR DENSITY_FOURIER_COMP
/*
This CV gives the fourier components of the number density rescaled rho=rho/sqrt(N)
where rho is a sum of delta functions or gaussians

\par Examples

label1: DENSITY_FOURIER_COMP N2_MAX=3
label2: DENSITY_FOURIER_COMP N2_MAX=3 ATOMS=1-100 SIGMA=2.5 SIMPLE_NAME PARALLEL NO_VIRIAL

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(DensityFourierComp,"DENSITY_FOURIER_COMP")

void DensityFourierComp::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);
  keys.add("compulsory","N2_MAX","the maximum square module of the indexing integer vector of the Fourier wave vectors (k) considered");
  keys.add("atoms","ATOMS","calculate fourier components using only these atoms. Default is to use all atoms");
  keys.add("optional","SIGMA","the reference density is a sum of gaussians with this sigma. Default is zero, i.e. delta functions");
  keys.add("optional","BOX_EDGE","the edge L of the cubic box to be considered");

  keys.addFlag("SIMPLE_NAME",false,"uses a simpler name for the components, without special characters");
  keys.addFlag("NO_VIRIAL",false,"skip the virial calculations, useful to speedup when a correct pressure is not needed (e.g. in NVT simulations)");
  keys.addFlag("OLD",false,"perform the calculation with the old algorithm. Calculates all sin and cos instead of creating a table");
  keys.addFlag("PARALLEL",false,"perform the calculation in parallel - CAUTION: might give worse performance");

  keys.addOutputComponent("ReRho","default","the real part of the fuorier component of the order parameter");
  keys.addOutputComponent("ImRho","default","the imaginary part of the fuorier component of the order parameter");
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components
}

DensityFourierComp::DensityFourierComp(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao)
{
  first_run_=true;
//parse needed stuff
  int n2_max=0;
  parse("N2_MAX",n2_max);
  plumed_massert(n2_max>0,"N2_MAX should be an integer greater than zero");
  double sigma=0;
  parse("SIGMA",sigma);
  sigma2_=sigma*sigma;
  box_edge_=-1;
  parse("BOX_EDGE",box_edge_);
  if (box_edge_!=-1) //setting BOX_EDGE=-1 is equivalent to not set it
    plumed_massert(box_edge_>0,"BOX_EDGE must be greater than zero");
  bool simple_name=false;
  parseFlag("SIMPLE_NAME", simple_name);
  no_virial_=false;
  parseFlag("NO_VIRIAL",no_virial_);
  parallel_=false;
  parseFlag("PARALLEL",parallel_);
  old_=false;
  parseFlag("OLD",old_);
  plumed_massert((!parallel_ || old_),"sorry, but a parallel version of the new algorithm is not yet implemented"); //TODO

//generate all the fourier wave vectors
  generate_int_fwv(n2_max);
  Re_rho_.resize(fwv_.size());
  Im_rho_.resize(fwv_.size());

//add components
  valueReRho.resize(fwv_.size());
  valueImRho.resize(fwv_.size());
  std::ostringstream oss;
  for (unsigned int k=0; k<fwv_.size(); k++)
  {
    oss.str("");
    if (simple_name)
      oss<<"ReRho-"<<k;
    else
      oss<<"ReRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]~"<<2*k+1;
    addComponentWithDerivatives(oss.str());
    componentIsNotPeriodic(oss.str());
    valueReRho[k]=getPntrToComponent(oss.str());

    oss.str("");
    if (simple_name)
      oss<<"ImRho-"<<k;
    else
      oss<<"ImRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]~"<<2*k+2;
    addComponentWithDerivatives(oss.str());
    componentIsNotPeriodic(oss.str());
    valueImRho[k]=getPntrToComponent(oss.str());
  }

//finish the parsing, getting the atoms
  vector<AtomNumber> atoms;
  parseAtomList("ATOMS",atoms);
  NumAtom_=atoms.size();
  if (NumAtom_==0) //default is to use all the atoms
  {
    NumAtom_=plumed.getAtoms().getNatoms();
    atoms.resize(NumAtom_);
    for(unsigned int i=0; i<NumAtom_; i++)
      atoms[i].setIndex(i);
  }
  requestAtoms(atoms);//this must stay after the addComponentWithDerivatives otherwise segmentation violation
//parsing finished
  checkRead();

//print some info in the log
  log.printf("  over a total of N_tot=%d, considering a number of atoms N=%d\n",plumed.getAtoms().getNatoms(),NumAtom_);
  log.printf("  calculate the Landau integrals using the following wave vectors:\n");
  log.printf("    k=(2pi/L)*n  -->  n=(n1,n2,n3) such that |n|^2<=%d\n",n2_max);
  log.printf("    a total number of %d indipendent CVs will be used\n",2*fwv_.size());
  if (sigma2_==0)
    log.printf("  the reference density is given by a sum of delta functions (sigma=0)\n");
  else
    log.printf("  the reference density is given by a sum of gaussians with sigma=%g\n",sigma);
  if(no_virial_)
    log.printf("  -- NO_VIRIAL: the virial contribution is not calculated\n");
  if(parallel_)
    log.printf("  -- PARALLEL: running with loop parallelization\n");
  if(old_)
    log.printf("  -- OLD: running with the old implementation\n");
}

// calculator
void DensityFourierComp::calculate()
{
  if (first_run_) //this exists only because the function getBox() does not work before here
  {
    log.printf("\nFirst run:\n");
    first_run_=false;
    //assuming an orthorombic cubic box. up to now there is no need of a broader compatibility
    const bool is_cubic=(abs(getBox()[0][0]-getBox()[1][1])<1e-3 && abs(getBox()[1][1]-getBox()[2][2])<1e-3);
    plumed_massert(is_cubic,"ERROR: the simulation box must be orthorombic and cubic");

    if (box_edge_==-1)
      box_edge_=pow(getBox().determinant(),1./3.);
    log.printf("  The simulation box (which should be cubic) has an edge L=%f\n",box_edge_);
    if (old_)
    {
      const double k_const=2*PLMD::pi/box_edge_;
      for (unsigned int k=0; k<fwv_.size(); k++)
        fwv_[k]*=k_const;
    }
    else
      k_const_=2*PLMD::pi/box_edge_;
  }

//calculate the fourier components and set the values
  if (!old_)
  {
    //Arrays to store per axis phases, as complex numbers:
    //  - even indexes are the real part and odd ones the imaginary
    //  - each row stores a different power, starting from zero
    //Trivial sin/cos calculations was roughly two times slower (derivatives taken away),
    // while the use of std::complex was just slightly slower.
    const unsigned size=2*NumAtom_*(1+n_max_);
    std::vector<double> x_axis(size);
    std::vector<double> y_axis(size);
    std::vector<double> z_axis(size);
    for(unsigned j=0; j<NumAtom_; j++)
    {
      x_axis[2*j]=1;
      y_axis[2*j]=1;
      z_axis[2*j]=1;
    }
    for(unsigned j=0; j<NumAtom_; j++)
    {
      const double x_arg=-1*k_const_*getPosition(j)[0];
      const double y_arg=-1*k_const_*getPosition(j)[1];
      const double z_arg=-1*k_const_*getPosition(j)[2];
      const unsigned index=2*(NumAtom_+j);
      x_axis[index]=cos(x_arg); x_axis[index+1]=sin(x_arg);
      y_axis[index]=cos(y_arg); y_axis[index+1]=sin(y_arg);
      z_axis[index]=cos(z_arg); z_axis[index+1]=sin(z_arg);
    }
    for (unsigned n=2; n<=n_max_; n++)
    {
      for(unsigned j=0; j<NumAtom_; j++) //FIXME a better cache handling would probably help...
      {
        x_axis[2*(n*NumAtom_+j)] = x_axis[2*(NumAtom_+j)]*x_axis[2*((n-1)*NumAtom_+j)]-x_axis[2*(NumAtom_+j)+1]*x_axis[2*((n-1)*NumAtom_+j)+1];
        x_axis[2*(n*NumAtom_+j)+1]=x_axis[2*(NumAtom_+j)]*x_axis[2*((n-1)*NumAtom_+j)+1]+x_axis[2*(NumAtom_+j)+1]*x_axis[2*((n-1)*NumAtom_+j)];
        y_axis[2*(n*NumAtom_+j)] = y_axis[2*(NumAtom_+j)]*y_axis[2*((n-1)*NumAtom_+j)]-y_axis[2*(NumAtom_+j)+1]*y_axis[2*((n-1)*NumAtom_+j)+1];
        y_axis[2*(n*NumAtom_+j)+1]=y_axis[2*(NumAtom_+j)]*y_axis[2*((n-1)*NumAtom_+j)+1]+y_axis[2*(NumAtom_+j)+1]*y_axis[2*((n-1)*NumAtom_+j)];
        z_axis[2*(n*NumAtom_+j)] = z_axis[2*(NumAtom_+j)]*z_axis[2*((n-1)*NumAtom_+j)]-z_axis[2*(NumAtom_+j)+1]*z_axis[2*((n-1)*NumAtom_+j)+1];
        z_axis[2*(n*NumAtom_+j)+1]=z_axis[2*(NumAtom_+j)]*z_axis[2*((n-1)*NumAtom_+j)+1]+z_axis[2*(NumAtom_+j)+1]*z_axis[2*((n-1)*NumAtom_+j)];
      }
    }
    //now can build the fourier components
    const double normalization=1./sqrt(NumAtom_);
    for (unsigned k=0; k<fwv_.size(); k++)
    {
      Re_rho_[k]=0;
      Im_rho_[k]=0;
      double weight_k=normalization;
      if (sigma2_!=0) //FIXME: might be slow, but I never use it...
        weight_k*=exp(-0.5*k_const_*fwv_[k].modulo2()*sigma2_);
      const int sign_x=((fwv_[k][0]<0) ? -1 : 1);
      const int sign_y=((fwv_[k][1]<0) ? -1 : 1);
      const int sign_z=((fwv_[k][2]<0) ? -1 : 1);
      const unsigned index_nx=sign_x*fwv_[k][0]*NumAtom_;
      const unsigned index_ny=sign_y*fwv_[k][1]*NumAtom_;
      const unsigned index_nz=sign_z*fwv_[k][2]*NumAtom_;
      for(unsigned j=0; j<NumAtom_; j++)
      {
        const double r_x=x_axis[2*(index_nx+j)];
        const double i_x=x_axis[2*(index_nx+j)+1]*sign_x;
        const double r_y=y_axis[2*(index_ny+j)];
        const double i_y=y_axis[2*(index_ny+j)+1]*sign_y;
        const double r_z=z_axis[2*(index_nz+j)];
        const double i_z=z_axis[2*(index_nz+j)+1]*sign_z;
        const double cos_KnRj=r_x*r_y*r_z-r_x*i_y*i_z-i_x*r_y*i_z-i_x*i_y*r_z;
        const double sin_KnRj=i_x*i_y*i_z-i_x*r_y*r_z-r_x*i_y*r_z-r_x*r_y*i_z;
        Re_rho_[k]+=cos_KnRj;
        Im_rho_[k]-=sin_KnRj;
        setAtomsDerivatives(valueReRho[k],j,(-1.)*weight_k*sin_KnRj*k_const_*fwv_[k]);
        setAtomsDerivatives(valueImRho[k],j,(-1.)*weight_k*cos_KnRj*k_const_*fwv_[k]);
      }
      Re_rho_[k]*=weight_k;
      Im_rho_[k]*=weight_k;
      valueReRho[k]->set(Re_rho_[k]);
      valueImRho[k]->set(Im_rho_[k]);
      if (!no_virial_)
      {
        setBoxDerivativesNoPbc(valueReRho[k]);
        setBoxDerivativesNoPbc(valueImRho[k]);
      }
    }
  }
//old implementation
  else if (old_ && !parallel_)
  {
    const double normalization=1./sqrt(NumAtom_);
    for (unsigned int k=0; k<fwv_.size(); k++)
    {
      Re_rho_[k]=0;
      Im_rho_[k]=0;
      double weight_k=normalization;
      if (sigma2_!=0)
        weight_k*=exp(-0.5*fwv_[k].modulo2()*sigma2_);
      for(unsigned int j=0; j<NumAtom_; j++)
      {
        const double dp_fwv_pos=dotProduct(fwv_[k],getPosition(j));
        const double cos_KnRj=cos(dp_fwv_pos);
        const double sin_KnRj=sin(dp_fwv_pos);
        Re_rho_[k]+=cos_KnRj;
        Im_rho_[k]-=sin_KnRj;
        setAtomsDerivatives(valueReRho[k],j,(-1.)*weight_k*sin_KnRj*fwv_[k]);
        setAtomsDerivatives(valueImRho[k],j,(-1.)*weight_k*cos_KnRj*fwv_[k]);
      }
      Re_rho_[k]*=weight_k;
      Im_rho_[k]*=weight_k;
      valueReRho[k]->set(Re_rho_[k]);
      valueImRho[k]->set(Im_rho_[k]);
      if (!no_virial_)
      {
        setBoxDerivativesNoPbc(valueReRho[k]);
        setBoxDerivativesNoPbc(valueImRho[k]);
      }
    }
  }
// parallel version
  else if (old_ && parallel_)
  {
    const double normalization=1./sqrt(NumAtom_);
    const unsigned int stride=comm.Get_size();
    const unsigned int rank=comm.Get_rank();
    std::vector<std::vector<Vector> > d_Re_rho(fwv_.size(),std::vector<Vector>(NumAtom_));
    std::vector<std::vector<Vector> > d_Im_rho(fwv_.size(),std::vector<Vector>(NumAtom_));
    for (unsigned int k=0; k<fwv_.size(); k++)
    {
      Re_rho_[k]=0;
      Im_rho_[k]=0;
      double weight_k=normalization;
      if (sigma2_!=0)
        weight_k*=exp(-0.5*fwv_[k].modulo2()*sigma2_);
      for(unsigned int j=rank; j<NumAtom_; j+=stride) //PARALLEL LOOP
      {
        const double dp_fwv_pos=dotProduct(fwv_[k],getPosition(j));
        const double cos_KnRj=cos(dp_fwv_pos);
        const double sin_KnRj=sin(dp_fwv_pos);
        Re_rho_[k] += cos_KnRj;
        Im_rho_[k] -= sin_KnRj;
        d_Re_rho[k][j]=(-1.)*weight_k*sin_KnRj*fwv_[k];
        d_Im_rho[k][j]=(-1.)*weight_k*cos_KnRj*fwv_[k];
      }
      comm.Sum(Re_rho_[k]);
      comm.Sum(Im_rho_[k]);
      comm.Sum(d_Re_rho[k]);
      comm.Sum(d_Im_rho[k]);

      Re_rho_[k]*=weight_k;
      Im_rho_[k]*=weight_k;
      valueReRho[k]->set(Re_rho_[k]);
      valueImRho[k]->set(Im_rho_[k]);
      for (unsigned int i=0; i<NumAtom_; i++)
      {
        setAtomsDerivatives(valueReRho[k],i,d_Re_rho[k][i]);//cannot stay in parallel loop
        setAtomsDerivatives(valueImRho[k],i,d_Im_rho[k][i]);
      }
      if (!no_virial_)
      {
        setBoxDerivativesNoPbc(valueReRho[k]);
        setBoxDerivativesNoPbc(valueImRho[k]);
      }
    }
  }
}

void DensityFourierComp::generate_int_fwv(int n2_max)
{
  const int max_int=floor(sqrt(n2_max));
  n_max_=max_int;
  int start;
  Vector new_point;
  for (int x=0; x<=max_int; x++)
  {
    if (x==0)
      start=0;
    else
      start=-1*max_int;
    for (int y=start; y<=max_int; y++)
    {
      if (x==0 && y==0)
        start=1; //this way it skips (0,0,0)
      else
        start=-1*max_int;
      if ((x*x+y*y)<=n2_max) //saves some calculations
      {
        for(int z=start; z<=max_int; z++)
        {
          if ((x*x+y*y+z*z)<=n2_max)
          {
            new_point[0]=x;
            new_point[1]=y;
            new_point[2]=z;
            fwv_.push_back(new_point);
          }
        }
      }
    }
  }
  std::stable_sort(fwv_.begin(),fwv_.end(),[](const Vector& a,const Vector& b)
  {return a.modulo2()<b.modulo2();}); //ordered is nicer
}

}
}

/***********************************************

  Here are the total number of fourier wave
  vectors k given different N2_MAX:

      n2_max  fwv_.size()
         1        3
         2        9
         3       13
         4       16
         5       28
         6       40
         7       40
         8       46
         9       61
        10       73

************************************************/

