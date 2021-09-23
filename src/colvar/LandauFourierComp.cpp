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
#include "LandauFourierComp.h"
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"

#include <cmath>
#include <algorithm> //std::stable_sort
#include <sstream>   //std::ostringstream

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR LANDAU_FOURIER_COMP
/*
This CV gives the fourier components of the order parameter psi=(rho-rho_c)/rho_0 in Landau's theory for a Lennard-Jones fluid
rho_c=N/V_c is the critical density given in LJ units.
rho_0=N/V   is the zero-th fourier amplitude, alias the average density.

\par Examples

label1: LANDAU_FOURIER_COMP N2_MAX=3
label2: LANDAU_FOURIER_COMP N2_MAX=3 ADD_RHO_IN_ZERO SIMPLE_NAME

*/
//+ENDPLUMEDOC

PLUMED_REGISTER_ACTION(LandauFourierComp,"LANDAU_FOURIER_COMP")

void LandauFourierComp::registerKeywords(Keywords& keys)
{
  Colvar::registerKeywords(keys);
  keys.add("compulsory","N2_MAX","the maximum square module of the indexing integer vector of the Fourier wave vectors (k) considered");
  keys.add("atoms","ATOMS","calculate fourier components using only these atoms. Default is to use all atoms");
  keys.add("optional","SIGMA","the reference density is a sum of gaussians with this sigma (LJ units). Default is zero, i.e. delta functions");

  keys.addFlag("ADD_RHO_IN_ZERO",false,"adds as component the value of the density in the real space point (0,0,0) for debugging");
  keys.addFlag("SIMPLE_NAME",false,"uses a simpler name for the components, without special characters");
  keys.addFlag("PARALLEL",false,"perform the calculation in parallel - CAUTION: gives worse performance for small N2_MAX");
  keys.addFlag("NO_VIRIAL",false,"skip the virial calculations, useful to speedup when a correct pressure is not needed (e.g. in NVT simulations)");

  keys.addOutputComponent("ReRho","default","the real part of the fuorier component of the order parameter");
  keys.addOutputComponent("ImRho","default","the imaginary part of the fuorier component of the order parameter");
  keys.addOutputComponent("RhoInZero","default","the value of the rebuilded density in the point (0,0,0). for debugging purposes");
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components
}

LandauFourierComp::LandauFourierComp(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao)
{
  first_run_=true;
  Vol_=-1;
//parse needed stuff
  int n2_max=0;
  parse("N2_MAX",n2_max);
  plumed_massert(n2_max>0,"N2_MAX should be an integer greater than zero");
  sigma_=0;
  parse("SIGMA",sigma_);
  add_rho_in_zero_=false;
  parseFlag("ADD_RHO_IN_ZERO",add_rho_in_zero_);
  bool simple_name=false;
  parseFlag("SIMPLE_NAME", simple_name);
  parallel_=false;
  parseFlag("PARALLEL",parallel_);
  no_virial_=false;
  parseFlag("NO_VIRIAL",no_virial_);

//generate all the fourier wave vectors
  generate_int_fwv(n2_max);
  Re_rho_.resize(fwv_.size());
  Im_rho_.resize(fwv_.size());

//add components
  int pos_count=1;
  if (add_rho_in_zero_)
  {
    addComponentWithDerivatives("RhoInZero"); //has derivatives only to avoid issues
    componentIsNotPeriodic("RhoInZero");
    valueRhoInZero=getPntrToComponent("RhoInZero");
    pos_count++;
  }
  valueReRho.resize(fwv_.size());
  valueImRho.resize(fwv_.size());
  std::ostringstream oss;
  for (unsigned int k=0; k<fwv_.size(); k++)
  {
    pos_count++;
    oss.str("");
    if (simple_name)
      oss<<"ReRho-"<<k;
    else
      oss<<"ReRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]~"<<pos_count;
    addComponentWithDerivatives(oss.str());
    componentIsNotPeriodic(oss.str());
    valueReRho[k]=getPntrToComponent(oss.str());

    pos_count++;
    oss.str("");
    if (simple_name)
      oss<<"ImRho-"<<k;
    else
      oss<<"ImRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]~"<<pos_count;
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
  log.printf("  calculate the Landau integrals using to the following wave vectors:\n");
  log.printf("    k=(2pi/L)*n  -->  n=(n1,n2,n3) such that |n|^2<=%d\n",n2_max);
  log.printf("    a total number of %d different k values will be used\n",2*fwv_.size());
  if (sigma_==0)
    log.printf("  the reference density is given by a sum of delta functions (sigma=0)\n");
  else
    log.printf("  the reference density is given by a sum of gaussians with sigma=%g\n",sigma_);
  if(add_rho_in_zero_)
    log.printf("  -- ADD_RHO_IN_ZERO: will give the value of the density in the point (0,0,0) of the real space\n");
  if(parallel_)
    log.printf("  -- PARALLEL: running with loop parallelization\n");
}

// calculator
void LandauFourierComp::calculate()
{
  if (first_run_) //this exists only because the function getBox() does not work before here
  {
    log.printf("\nFirst run:\n");
    first_run_=false;
    //assuming an orthorombic cubic box. up to now there is no need of a broader compatibility
    const bool is_cubic=(abs(getBox()[0][0]-getBox()[1][1])<1e-3 && abs(getBox()[1][1]-getBox()[2][2])<1e-3);
    plumed_massert(is_cubic,"ERROR: the simulation box must be orthorombic and cubic");

    Vol_=getBox().determinant();
    log.printf("  The simulation box (which should be cubic) has a volume V=%f\n",Vol_);
    const double k_const=2*PLMD::pi/pow(Vol_,1./3.);
    for (unsigned int k=0; k<fwv_.size(); k++)
      fwv_[k]*=k_const;
    //initialize the weight for each fourier component
    if (sigma_!=0)
    {
      sigma_weight_.resize(fwv_.size(),1./NumAtom_);
      for (unsigned int k=0; k<fwv_.size(); k++)
        sigma_weight_[k]*=exp(-0.5*fwv_[k].modulo2()*pow(sigma_,2));
    }
  }

//calculate the fourier components and set the values
  if (!parallel_)
  {
    double rho_in_zero=0;
    for (unsigned int k=0; k<fwv_.size(); k++)
    {
      Re_rho_[k]=0;
      Im_rho_[k]=0;
      double weight_k=1./NumAtom_;
      if (sigma_!=0)
        weight_k=sigma_weight_[k];

      for(unsigned int i=0; i<NumAtom_; i++)
      {
        const double dp_fwv_pos=dotProduct(fwv_[k],getPosition(i));
        const double cos_ki=cos(dp_fwv_pos);
        const double sin_ki=sin(dp_fwv_pos);
        Re_rho_[k] += cos_ki;
        Im_rho_[k] -= sin_ki;
        setAtomsDerivatives(valueReRho[k],i,(-1.)*weight_k*sin_ki*fwv_[k]);
        setAtomsDerivatives(valueImRho[k],i,(-1.)*weight_k*cos_ki*fwv_[k]);
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
      if (add_rho_in_zero_)
        rho_in_zero+=Re_rho_[k];
    }
    if (add_rho_in_zero_)
    {
      rho_in_zero*=2*NumAtom_/Vol_;
      rho_in_zero+=NumAtom_/Vol_;
      valueRhoInZero->set(rho_in_zero);
    }
  }

// parallel version
  else
  {
    const unsigned int stride=comm.Get_size();
    const unsigned int rank=comm.Get_rank();
    std::vector<std::vector<Vector> > d_Re_rho(fwv_.size(),std::vector<Vector>(NumAtom_));
    std::vector<std::vector<Vector> > d_Im_rho(fwv_.size(),std::vector<Vector>(NumAtom_));
    double rho_in_zero=0;
    for (unsigned int k=0; k<fwv_.size(); k++)
    {
      Re_rho_[k]=0;
      Im_rho_[k]=0;
      double weight_k=1./NumAtom_;
      if (sigma_!=0)
        weight_k=sigma_weight_[k];

      for(unsigned int i=rank; i<NumAtom_; i+=stride) //PARALLEL LOOP
      {
        const double dp_fwv_pos=dotProduct(fwv_[k],getPosition(i));
        const double cos_ki=cos(dp_fwv_pos);
        const double sin_ki=sin(dp_fwv_pos);
        Re_rho_[k] += cos_ki;
        Im_rho_[k] -= sin_ki;
        d_Re_rho[k][i]=(-1.)*weight_k*sin_ki*fwv_[k];
        d_Im_rho[k][i]=(-1.)*weight_k*cos_ki*fwv_[k];
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
      if (add_rho_in_zero_)
        rho_in_zero+=Re_rho_[k];
    }
    if (add_rho_in_zero_)
    {
      rho_in_zero*=2*NumAtom_/Vol_;
      rho_in_zero+=NumAtom_/Vol_;
      valueRhoInZero->set(rho_in_zero);
    }
  }
}

void LandauFourierComp::generate_int_fwv(int n2_max)
{
  const int max_int=floor(sqrt(n2_max));
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
  std::stable_sort(fwv_.begin(),fwv_.end(),Landau_VectorsCompare); //ordered is nicer
//  std::stable_sort(fwv_.begin(),fwv_.end(),[](const Vector& a,const Vector& b){return a.modulo2()<b.modulo2();}); //this requires c++11
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

