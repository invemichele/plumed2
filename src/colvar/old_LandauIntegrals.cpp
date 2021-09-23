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
#include "Colvar.h"
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "tools/Communicator.h"

#include <string>
#include <cmath>

using namespace std;

namespace PLMD {
namespace colvar {

//+PLUMEDOC COLVAR LANDAU_INTEGRALS_OLD
/*
This CV gives the value of the integrals of Landau theory for a special case.
  NOTE: the integrals are taken per unit volume
rho_c_=N/V_c is the critical density given in LJ units.

\par Examples

label1: LANDAU_INTEGRALS_OLD N2_MAX=3
label2: LANDAU_INTEGRALS_OLD ATOMS=1-512 N2_MAX=3 RHO_C=0.317

*/
//+ENDPLUMEDOC

struct indexes_pair {
  unsigned int k, kk;
  int sign_k, sign_kk;

  indexes_pair(unsigned int _k, int _sign_k, unsigned int _kk, int _sign_kk)
  { k=_k, sign_k=_sign_k, kk=_kk, sign_kk=_sign_kk;  }
};

class LandauIntegrals_OLD : public Colvar {

private:
  bool first_run_;
  bool parallel_;
  bool no_quartic_;
  bool add_f_comp_;

  unsigned int NumAtom_, fwv_size_;
  double Vol_;
  double rho_c_, psi_zero_;
  std::vector<Vector> fwv_; //fourier wave vector
  std::vector<double> k2_;
  std::vector<std::vector<indexes_pair> > conv_pair_; //indexes needed for the convolution

  Value* valueQuad;
  Value* valueKin;
  Value* valueQuart;
  Value* valueRhoInZero;
  std::vector<Value*> valueReRho;
  std::vector<Value*> valueImRho;

public:
  LandauIntegrals_OLD(const ActionOptions&);
  virtual void calculate();
  static void registerKeywords(Keywords& );
};

PLUMED_REGISTER_ACTION(LandauIntegrals_OLD,"LANDAU_INTEGRALS_OLD")

void LandauIntegrals_OLD::registerKeywords(Keywords& keys) {
  Colvar::registerKeywords(keys);
  keys.add("compulsory","N2_MAX","the maximum square module of the indexing integer vector of the Fourier wave vectors (k) considered");
  keys.add("atoms","ATOMS","calculate Landau integrals using only these atoms. Default is to use all atoms");
  keys.add("optional","RHO_C","the critical density of the system. By default (or if set to -1) the system is supposed to be at critical density");

  keys.addFlag("PARALLEL",false,"perform the calculation in parallel - CAUTION: gives worst performance for small N2_MAX");
  keys.addFlag("NO_QUARTIC",false,"skips the calculation of the quartic integral - for debug purpose");
  keys.addFlag("ADD_FOURIER_COMPONENTS",false,"adds as components the square module of the density fourier amplitudes, rho2_k");

  keys.addOutputComponent("quadratic","default","the Landau integral given by the quadratic terms");
  keys.addOutputComponent("kinetic","default","the Landau integral given by the gradient terms");
  keys.addOutputComponent("quartic","default","the Landau integral given by the quartic terms");

  keys.addOutputComponent("ReRho","default","the real part of the fuorier component of the order parameter");
  keys.addOutputComponent("ImRho","default","the imaginary part of the fuorier component of the order parameter");
  keys.addOutputComponent("RhoInZero","default","the value of the rebuilded density in the point (0,0,0). for debugging purposes");
  ActionWithValue::useCustomisableComponents(keys); //needed to have valueReRho and valueImRho
}

LandauIntegrals_OLD::LandauIntegrals_OLD(const ActionOptions&ao):
  PLUMED_COLVAR_INIT(ao),
  first_run_(true),
  parallel_(false),
  no_quartic_(false),
  add_f_comp_(false),
  NumAtom_(0.0),
  rho_c_(-1.),
  psi_zero_(0.) //default is to assume you're running at rho_c, so psi_zero=0
{
//adding components
  addComponentWithDerivatives("quadratic");
  componentIsNotPeriodic("quadratic");
  valueQuad=getPntrToComponent("quadratic");
  addComponentWithDerivatives("kinetic");
  componentIsNotPeriodic("kinetic");
  valueKin=getPntrToComponent("kinetic");
  addComponentWithDerivatives("quartic");
  componentIsNotPeriodic("quartic");
  valueQuart=getPntrToComponent("quartic");

//parsing needed stuff
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
  requestAtoms(atoms); //requestAtoms has to be done after addComponent
  //NumAtom_=getNumberOfAtoms();

  int n2_max=0;
  parse("N2_MAX",n2_max);
  if (n2_max<=0)
    error("N2_MAX should be an integer greater than zero");

  parse("RHO_C",rho_c_);
  if (rho_c_==-1)
    log.printf("  -- WARNING -- if not running at rho=rho_c then RHO_C must be specified. Using psi_zero=0\n");
  else
    log.printf("  using as critical density rho_c=%f\n",rho_c_);

  parseFlag("PARALLEL",parallel_);
  if(parallel_)
    log.printf("  -- PARALLEL: running with parallelization\n");
  parseFlag("NO_QUARTIC",no_quartic_);
  if(no_quartic_)
    log.printf("  -- NO_QUARTIC: skipping the calculation of the quartic integral\n");
  parseFlag("ADD_FOURIER_COMPONENTS",add_f_comp_);
  if(add_f_comp_)
    log.printf("  -- ADD_FOURIER_COMPONENTS: you will get a lot of output!\n");
  checkRead();

//generating all the fourier wave vectors
  int max_int=floor(2*sqrt(n2_max)); //triangle inequality gives this limitation
  int start;
  fwv_size_=0;
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
      if ((x*x+y*y)<=4*n2_max) //this saves some calc
      {
        for(int z=start; z<=max_int; z++)
        {
          if ((x*x+y*y+z*z)<=4*n2_max)
          {
            new_point[0]=x;
            new_point[1]=y;
            new_point[2]=z;
            if ((x*x+y*y+z*z)<=n2_max)
            {
              fwv_.insert(fwv_.begin(),new_point);
              fwv_size_++; //the considered k frequencies are at the beginning
            }
            else
              fwv_.push_back(new_point);
          }
        }
      }
    }
  }

//individuating the pairs of fwv needed for the convolution
  conv_pair_.resize(fwv_.size());
  for (unsigned int q=0; q<fwv_.size(); q++)
  {
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      for (unsigned int kk=k; kk<fwv_size_; kk++) //taking only half. multiplicity=2
      {
        if ((fwv_[q]-fwv_[k]-fwv_[kk]).modulo2()==0) // q-k=kk in fwv_
        {
          indexes_pair new_pair(k,+1,kk,+1);
          conv_pair_[q].push_back(new_pair);
        }
        else if ((fwv_[q]+fwv_[k]-fwv_[kk]).modulo2()==0) // q+k=kk in fwv_
        {
          indexes_pair new_pair(k,-1,kk,+1);
          conv_pair_[q].push_back(new_pair);
        }
        else if ((fwv_[q]-fwv_[k]+fwv_[kk]).modulo2()==0) // q-k=-kk in -fwv_
        {
          indexes_pair new_pair(k,+1,kk,-1);
          conv_pair_[q].push_back(new_pair);
        }
        else if ((fwv_[q]+fwv_[k]+fwv_[kk]).modulo2()==0) // q+k=-kk in -fwv_
        {
          indexes_pair new_pair(k,-1,kk,-1);
          conv_pair_[q].push_back(new_pair);
        }
      }
    }
  }
  fwv_.resize(fwv_size_);//the extra frequencies are not needed any more

//adding the fourier components
  if (add_f_comp_)
  {
    addComponent("RhoInZero");
    componentIsNotPeriodic("RhoInZero");
    valueRhoInZero=getPntrToComponent("RhoInZero");

    valueReRho.resize(fwv_size_);
    valueImRho.resize(fwv_size_);
    std::ostringstream oss;
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      oss.str("");
      oss<<"ReRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]";
      addComponent(oss.str());
      componentIsNotPeriodic(oss.str());
      valueReRho[k]=getPntrToComponent(oss.str());
      oss.str("");
      oss<<"ImRho-["<<fwv_[k][0]<<","<<fwv_[k][1]<<","<<fwv_[k][2]<<"]";
      addComponent(oss.str());
      componentIsNotPeriodic(oss.str());
      valueImRho[k]=getPntrToComponent(oss.str());
    }
  }

//printing some info in the log
  log.printf("  considering a number of atoms N=%d over a total of N_tot=%d\n",NumAtom_,plumed.getAtoms().getNatoms());
  log.printf("  calculate the Landau integrals using to the following wave vectors:\n");
  log.printf("    k=(2pi/L)*n  -->  n=(n1,n2,n3) such that |n|^2<=%d\n",n2_max);
  log.printf("    a total number of %d different k values will be used\n",fwv_size_);
  log.printf("    a total number of %d different q values will be used\n",conv_pair_.size());
}

// calculator
void LandauIntegrals_OLD::calculate() {

  if (first_run_) //this exists only because the function getBox() does not work before here
  {
    first_run_=false;
    Vol_=getBox().determinant();
    log.printf("  the volume is V=%f\n",Vol_);
    if (rho_c_==-1) //just a small comfort for an easier input
      psi_zero_=0;
    else
      psi_zero_=1-rho_c_*Vol_/NumAtom_;
    // Orthorhombic box so the box tensor is diagonal
    double kx_const=2.0*pi/getBox()[0][0]; //the box now is cubic but who knows...
    double ky_const=2.0*pi/getBox()[1][1];
    double kz_const=2.0*pi/getBox()[2][2];
    k2_.resize(fwv_size_);
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      fwv_[k][0]*=kx_const;
      fwv_[k][1]*=ky_const;
      fwv_[k][2]*=kz_const;
      k2_[k]=fwv_[k].modulo2();
    }
  }

// initializing parallel stuff
  unsigned stride=1;
  unsigned rank=0;
  if(parallel_)
  {
    stride=comm.Get_size();
    rank=comm.Get_rank();
  }

//needed variables
  double Quadratic_integral=0.5*psi_zero_*psi_zero_;
  double Kinetic_integral=0.;
  double Quartic_integral=0.;
  std::vector<Vector> d_Quartic_integral(NumAtom_); //needed because cannot set the derivatives inside a parallel loop

  std::vector<double> Re_rho(fwv_size_,0.);
  std::vector<double> Im_rho(fwv_size_,0.);
  std::vector<std::vector<Vector> > d_Re_rho(fwv_size_,std::vector<Vector>(NumAtom_));
  std::vector<std::vector<Vector> > d_Im_rho(fwv_size_,std::vector<Vector>(NumAtom_));

//first calculate Quadratic and Kinetic, that are similar
  double rho_in_zero=0;
  for (unsigned int k=0; k<fwv_size_; k++)
  {
    for(unsigned int i=rank; i<NumAtom_; i+=stride) //PARALLEL LOOP
    {
      double dp_fwv_pos=dotProduct(fwv_[k],getPosition(i));
      double cos_ki=cos(dp_fwv_pos);
      double sin_ki=sin(dp_fwv_pos);
      Re_rho[k] += cos_ki; //the density is a sum of delta functions
      Im_rho[k] -= sin_ki;
      d_Re_rho[k][i]-=sin_ki*fwv_[k];
      d_Im_rho[k][i]-=cos_ki*fwv_[k];
    }
    if(parallel_)
    {
      comm.Sum(Re_rho[k]);
      comm.Sum(Im_rho[k]);
      comm.Sum(d_Re_rho[k]);
      comm.Sum(d_Im_rho[k]);
    }
    Re_rho[k]/=NumAtom_;
    Im_rho[k]/=NumAtom_;
    double rho2_k=Re_rho[k]*Re_rho[k]+Im_rho[k]*Im_rho[k];

    Quadratic_integral+=rho2_k;
    Kinetic_integral+=k2_[k]*rho2_k; //if n2_max=1 and L=11.732851292 (N=512->V=Vc) then quadratic=kinetic*3.486963455
    if (add_f_comp_)
    {
      valueReRho[k]->set(Re_rho[k]);
      valueImRho[k]->set(Im_rho[k]);
      rho_in_zero+=Re_rho[k];
    }
  }
  Quadratic_integral*=2;
  Kinetic_integral*=2;
  valueQuad->set(Quadratic_integral);
  valueKin->set(Kinetic_integral);

  if (add_f_comp_)
  {
    rho_in_zero*=2*NumAtom_/Vol_;
    rho_in_zero+=NumAtom_/Vol_;
    valueRhoInZero->set(rho_in_zero);
  }

//now calculate the Quartic, through convolution
  if (!no_quartic_)
  {
    for (unsigned int q=rank; q<conv_pair_.size(); q+=stride) //PARALLEL LOOP the one over the atoms cannot be parallel due to comm.Sum restrictions
    {
      double Re_psi2_q=0.; //actually these are rescaled: V*psi2_q, V*d_psi2_q
      double Im_psi2_q=0.;
      std::vector<Vector> d_Re_psi2_q(NumAtom_);
      std::vector<Vector> d_Im_psi2_q(NumAtom_);
      if (q<fwv_size_)//there are always two zero cases: k->0, q-k->0
      {
        unsigned int k=q; //just to avoid confusion: is below fwv_size_
        Re_psi2_q+=2*psi_zero_*Re_rho[k];
        Im_psi2_q+=2*psi_zero_*Im_rho[k];
        for (unsigned int i=0; i<NumAtom_; i++)
        {
          d_Re_psi2_q[i]+=2*psi_zero_*d_Re_rho[k][i];
          d_Im_psi2_q[i]+=2*psi_zero_*d_Im_rho[k][i];
        }
      }
      for (unsigned int j=0; j<conv_pair_[q].size(); j++)
      {
        //to be less verbose, saving needed stuff in local variables
        unsigned int k=conv_pair_[q][j].k;
        unsigned int kk=conv_pair_[q][j].kk;
        int sign_k=conv_pair_[q][j].sign_k;
        int sign_kk=conv_pair_[q][j].sign_kk;
        int mlt=2; //multiplicity
        if (k==kk)
          mlt=1;

        Re_psi2_q+=mlt*(Re_rho[k]*Re_rho[kk]-sign_k*Im_rho[k]*sign_kk*Im_rho[kk]);
        Im_psi2_q+=mlt*(Re_rho[k]*sign_kk*Im_rho[kk]+sign_k*Im_rho[k]*Re_rho[kk]);
        for (unsigned int i=0; i<NumAtom_; i++) //TODO implement the other method to get the derivatives faster
        {
          d_Re_psi2_q[i]+=mlt*(d_Re_rho[k][i]*Re_rho[kk]+Re_rho[k]*d_Re_rho[kk][i]);
          d_Re_psi2_q[i]-=mlt*sign_k*sign_kk*(d_Im_rho[k][i]*Im_rho[kk]+Im_rho[k]*d_Im_rho[kk][i]);

          d_Im_psi2_q[i]+=mlt*sign_kk*(d_Re_rho[k][i]*Im_rho[kk]+Re_rho[k]*d_Im_rho[kk][i]);
          d_Im_psi2_q[i]+=mlt*sign_k*(d_Im_rho[k][i]*Re_rho[kk]+Im_rho[k]*d_Re_rho[kk][i]);
        }
      }
      Quartic_integral+=Re_psi2_q*Re_psi2_q+Im_psi2_q*Im_psi2_q;
      for (unsigned int i=0; i<NumAtom_; i++)
        d_Quartic_integral[i]+=Re_psi2_q*d_Re_psi2_q[i]+Im_psi2_q*d_Im_psi2_q[i];
    }
    if(parallel_)
    {
      comm.Sum(Quartic_integral);
      comm.Sum(d_Quartic_integral);
    }
    Quartic_integral*=2;
    Quartic_integral+=Quadratic_integral*Quadratic_integral;//the q=(0,0,0) component is equal to Q2
  }
  valueQuart->set(Quartic_integral);

//Setting the value of all the derivatives
  for (unsigned int i=0; i<NumAtom_; i++)
  {
    Vector d_Quadratic_integral_i;
    Vector d_Kinetic_integral_i;
    for (unsigned int k=rank; k<fwv_size_; k+=stride) //PARALLEL LOOP
    {
      Vector d_rho2_ki=Re_rho[k]*d_Re_rho[k][i]+Im_rho[k]*d_Im_rho[k][i];
      d_Quadratic_integral_i+=d_rho2_ki;
      d_Kinetic_integral_i+=k2_[k]*d_rho2_ki;
    }
    if(parallel_)
    {
      comm.Sum(d_Quadratic_integral_i);
      comm.Sum(d_Kinetic_integral_i);
    }
    d_Quadratic_integral_i*=4;
    d_Kinetic_integral_i*=4;
    d_Quartic_integral[i]*=4;
    d_Quartic_integral[i]+=2*Quadratic_integral*d_Quadratic_integral_i; //q=(0,0,0)

    setAtomsDerivatives(valueQuad,i,d_Quadratic_integral_i); //these cannot stay in a parallel loop!
    setAtomsDerivatives(valueKin,i,d_Kinetic_integral_i);
    setAtomsDerivatives(valueQuart,i,d_Quartic_integral[i]);
  }
  setBoxDerivativesNoPbc(valueQuad);
  setBoxDerivativesNoPbc(valueKin);
  setBoxDerivativesNoPbc(valueQuart);
}

}
}

/********************************************************************************************

Here are the total number of fourier wave vectors k given different N2_MAX:

  n2_max  fwv_size_  conv_pair_.size()    4*n2_max
    1         3             16               4
    2         9             46               8
    3        13             89              12
    4        16            128              16
    5        28            194              20
    6        40            242              24
    7        40            309              28
    8        46            375              32
    9        61            462              36

**********************************************************************************************/

