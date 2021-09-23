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
#include "Bias.h"
#include "../colvar/LandauFourierComp.h"
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/ActionSet.h"
#include "core/Atoms.h"
#include "tools/Communicator.h"

#include <algorithm> // std::stable_sort
#include <sstream>   // std::ostringstream

using namespace std;

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS VARIATIONAL_LANDAU_DIS
/*
 * Implementation of the variationally enhanced method to obtain the optimal estimate of the phenomenological parameters
 * in the Ginzburg-Landau free-energy model for a Lennard-Jonnes fluid close to the critical point.
 *
 * The used collective variables are the Fourier amplitudes of the order parameter (rescaled density) up to a maximum wave vector.
 * Supports only N2_MAX<=max_n2_max_ but an higher number is not physically meaningfull.
*/
//+ENDPLUMEDOC

//a struct needed for the convolution to calculate I4
namespace // this struct is only needed in this .cpp file
{
struct indexes_pair //FIXME for sure this can be made faster
{
  unsigned int k, kk;
  int sign_k, sign_kk;

  indexes_pair(unsigned int _k, int _sign_k, unsigned int _kk, int _sign_kk)
  { k=_k, sign_k=_sign_k, kk=_kk, sign_kk=_sign_kk;  }
};
}

//now the bias class
class VariationalLandau_DIS : public Bias {

private:
  static const unsigned int max_n2_max_=20; //N2_MAX can't be larger than this (unless you add entries in fwv_sizes_)
  static const unsigned int fwv_sizes_[max_n2_max_];
  static const unsigned int dim_=3; //number of Landau Integrals (LI). you cannot change it unless you change other parts of the code

  bool first_run_;
  bool extra_arg_;
  bool amplitude_cutoff_;
  bool no_bias_;
  bool no_quartic_;
  bool parallel_;
  unsigned int walkers_num_;

  double Vol_;
  double rho_c_;
  double psi_zero_; //psi_0=(1-rho_c/rho_0)=(1-V/V_c)
  unsigned int n2_max_;
  unsigned int fwv_size_;
  unsigned m_max_, pos_size_;
  std::vector<double> k2_;
  std::vector<double> cos_km_;
  std::vector<double> sin_km_;
  std::vector<std::vector<indexes_pair> > conv_pair_; //indexes needed for the convolution
  const double* Re_;
  const double* Im_;
  std::vector<double> max_ampl2_;

  double beta_;
  double inv_gamma_;
  std::vector<double> minimization_step_;
  std::vector<double> coeff_, mean_coeff_, fes_coeff_;
  std::vector<double> target_av_LI_;  //'target' -> 'target distribution', p(s)
  std::vector<double> av_LI_, av_prod_LI_;//'av' -> 'ensemble average'
  double av_exp_bias_;
  unsigned int av_counter_, av_stride_;
  unsigned int target_stride_;
  unsigned int mean_counter_;
  unsigned int mean_weight_tau_;
  std::vector<double> manual_target_coeff_;

  Value* valueIg;
  Value* valueI2;
  Value* valueI4;

  Value* valueForceTot2;
  Value* valueOmega;
  Value* valueGradOmega[dim_];

  Value* valueCoeff[dim_];
  Value* valueMeanCoeff[dim_];
  Value* valueFesCoeff[dim_];

//class private methods
  bool check_arg_and_set_n2_max(unsigned int);
  void initialize_conv_pair(unsigned int);
  void calc_landau_int_and_set_forces(double*);
  void update_ensemble_av(double*);
  void update_target_av();
  void update_omega_and_coeffs();
  inline unsigned int get_index(unsigned int, unsigned int) const;

public:
  VariationalLandau_DIS(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

const unsigned int VariationalLandau_DIS::fwv_sizes_[max_n2_max_]= {3,9,13,16,28,40,40,46,61,73,85,89,101,125,125,128,152,170,182,194}; //not a problem if two have same size

PLUMED_REGISTER_ACTION(VariationalLandau_DIS,"VARIATIONAL_LANDAU_DIS")

void VariationalLandau_DIS::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.addOutputComponent("Ig","default","the Landau integral given by the gradient terms");
  keys.addOutputComponent("I2","default","the Landau integral given by the quadratic terms");
  keys.addOutputComponent("I4","default","the Landau integral given by the quartic terms");

  keys.addOutputComponent("force2","default","the instantaneous value of the squared total force");
  keys.addOutputComponent("omega","default","the value of the functional Omega");

  keys.addOutputComponent("GradOmega","default","gradient of Omega, the functional to be minimized");
  keys.addOutputComponent("InstCoeff","default","istantaneous coefficient of the bias potential");
  keys.addOutputComponent("MeanCoeff","default","mean coefficient of the bias potential");
  keys.addOutputComponent("FesCoeff","default","Landau coefficient of the free energy");
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components

  keys.use("ARG");
  keys.add("optional","RHO_C","the critical densiy of the system, used to define psi_0=1-rho_c/rho_0, default is psi_0=0");
  keys.add("optional","PSI_ZERO","the zero-th fourier component: psi_0=1-rho_c/rho_0, default is zero");
  keys.add("optional","TEMP","temperature");
  keys.add("optional","MAX_AMPL","maximum amplitude of the fourier components, can be a different one for each different |k|^2 (N2_MAX in total). default is 1");
  keys.add("optional","GAMMA","bias factor for the semi-well-tempered sampling");
  keys.add("optional","MANUAL_TG_COEFFS","set the averages over the target distribution through fixed coefficients");
  keys.add("optional","AV_STRIDE","the number of steps between updating coeffs");
  keys.add("optional","TG_STRIDE","tune how often the average distribution is updated in semi-WT scheme. Default is each time");
  keys.add("optional","MEAN_COUNTER","avoid initial kink to the mean coefficient value when continuing a run");
  keys.add("optional","TAU_MEAN","exponentially decaing average for the mean coefficients: all old points count as (tau-1) points. rescaled using av_stride");
  keys.add("optional","M_STEP","the step used for the minimization of the functional");
  keys.add("optional","INITIAL_FES_COEFFS","an initial guess for the Free Energy Surface coefficients");

  keys.addFlag("NO_MULTIPLE_WALKERS",false,"do not use Multiple Walkers, even if there are multiple simulations running");
  keys.addFlag("NO_BIAS",false,"do not apply any bias. MAX_AMPL still applies");
  keys.addFlag("NO_QUARTIC",false,"do not include the quartic term in the biasing. Still calculates its value");
  keys.addFlag("PARALLEL",false,"perform the calculation in parallel - CAUTION: could give worse performance");
}

VariationalLandau_DIS::VariationalLandau_DIS(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao)
{
  unsigned int arg_num=getNumberOfArguments();
//check for the extra argument at the beginning of ARG
  const char extra_argument[]="RhoInZero";
  extra_arg_=(getPntrToArgument(0)->getName().find(extra_argument)!=std::string::npos);
  if (extra_arg_)
    arg_num--;
//initialize n2_max_ and check for consistency with the used colvar
  const bool successful=check_arg_and_set_n2_max(arg_num);
  plumed_massert(successful,"the number of arguments dosen't match any of the predetermined. You are either trying to bias the wrong colvar or using a N2_MAX>max_n2_max_");

//initialize other variables
  first_run_=true;
  fwv_size_=fwv_sizes_[n2_max_-1];
  initialize_conv_pair(n2_max_);
  coeff_.resize(dim_);
  mean_coeff_.resize(dim_);
  target_av_LI_.resize(dim_);
  av_LI_.resize(dim_);
  av_prod_LI_.resize(dim_*(dim_+1)/2); //diagonal matrix mapped into a vector (convenient for parallel loops)
  av_exp_bias_=0;
  av_counter_=0;

//parsing
  rho_c_=-1;
  psi_zero_=-7e70; //FIXME hugly, but should always work
  parse("RHO_C",rho_c_);
  parse("PSI_ZERO",psi_zero_);
  plumed_massert(rho_c_==-1 || psi_zero_==-7e70,"ERROR: cannot set both RHO_C and PSI_ZERO, choose one");
  if (rho_c_==-1 && psi_zero_==-7e70)
    psi_zero_=0; //default. if rho_c_ is set psi_zero_ will be updated as soon as we have rho_0=N/V

  double temp=0;
  parse("TEMP",temp);
  double KbT=plumed.getAtoms().getKBoltzmann()*temp;
  if(KbT<=0)
  {
    KbT=plumed.getAtoms().getKbT();
    plumed_massert(KbT>0,"ERROR: unless the MD engine passes the temperature to plumed, you must specify it using TEMP");
  }
  beta_=1.0/KbT; //remember: with LJ use NATURAL UNITS

  amplitude_cutoff_=true;
  std::vector<double> max_ampl;
  parseVector("MAX_AMPL",max_ampl);
  if(max_ampl.size()==0)
  {
    amplitude_cutoff_=false;
    max_ampl2_.resize(fwv_size_,1);
  }
  else if(max_ampl.size()==1)
    max_ampl2_.resize(fwv_size_,max_ampl[0]*max_ampl[0]);
  else
  {
    plumed_massert(max_ampl.size()==n2_max_,"ERROR: MAX_AMPL should have either one or N2_MAX arguments");
    for (unsigned int l=0; l<n2_max_; l++)
      max_ampl2_.resize(fwv_sizes_[l],max_ampl[l]*max_ampl[l]);
  }

  double gamma=0;
  parse("GAMMA",gamma);
  if (gamma==0)
    inv_gamma_=0; //default or GAMMA=0 -> inv_gamma_=0 that gives non semi-WT sampling
  else
  {
    plumed_massert(gamma>=1, "ERROR: gamma has to be greater than one for the semi-well-tempered sampling to work");
    inv_gamma_=1./gamma;
  }

  parseVector("MANUAL_TG_COEFFS",manual_target_coeff_);
  if (manual_target_coeff_.size()!=0)
  {
    plumed_massert(manual_target_coeff_.size()==2,"ERROR: number of MANUAL_TG_COEFFS must be 2, the kinetic and the quadratic");
    plumed_massert(inv_gamma_==0,"ERROR: semi-well-tempered scheme is not compatible with fixed target average. Set GAMMA or MANUAL_TG_COEFFS, not both");
  }
  else
    manual_target_coeff_.resize(2,0);

  av_stride_=2000;
  parse("AV_STRIDE",av_stride_);
  target_stride_=1;
  parse("TG_STRIDE",target_stride_);
  mean_counter_=1;
  parse("MEAN_COUNTER",mean_counter_);
  mean_weight_tau_=0;
  parse("TAU_MEAN",mean_weight_tau_);
  plumed_massert((mean_weight_tau_==0 || mean_weight_tau_>av_stride_),"ERROR: TAU_MEAN is rescaled with AV_STRIDE, so it has to be greater");
  mean_weight_tau_/=av_stride_; //this way you can look at the number of simulation steps to choose TAU_MEAN

  parseVector("M_STEP",minimization_step_);
  double m_step=0.1;
  if (minimization_step_.size()==1)
    m_step=minimization_step_[0];
  if (minimization_step_.size()<=1)
    minimization_step_.resize(dim_,m_step);
  else
    plumed_massert(minimization_step_.size()==dim_,"ERROR: number of arguments for M_STEP must be 1 or match number of Landau Integrals");

  parseVector("INITIAL_FES_COEFFS",fes_coeff_);
  if (fes_coeff_.size()==0)
    fes_coeff_.resize(dim_);
  else
    plumed_massert(fes_coeff_.size()==dim_,"ERROR: number of INITIAL_FES_COEFFS must match number of Landau Integrals");
//now can initialize the coefficients
  for (unsigned int i=0; i<dim_; i++)
  {
    if (i!=2) //the original condition is (i==0 || i==1)
      coeff_[i]=(inv_gamma_-1)*fes_coeff_[i]+manual_target_coeff_[i];
    else
      coeff_[i]=-1*fes_coeff_[i];
    mean_coeff_[i]=coeff_[i];
  }

  bool no_multiple_walkers=false;
  parseFlag("NO_MULTIPLE_WALKERS",no_multiple_walkers);
  if (no_multiple_walkers)
    walkers_num_=1;
  else
  {
    walkers_num_=0;
    if (comm.Get_rank()==0)//multi_sim_comm works well on first rank only
      walkers_num_=multi_sim_comm.Get_size();
    if (comm.Get_size()>1) //if each walker has more than one processor update them all
      comm.Bcast(walkers_num_,0);
  }

  no_bias_=false;
  parseFlag("NO_BIAS",no_bias_);
  no_quartic_=false;
  parseFlag("NO_QUARTIC",no_quartic_);
  parallel_=false;
  parseFlag("PARALLEL",parallel_);
  checkRead();

//add all the output components
  addComponent("Ig"); componentIsNotPeriodic("Ig"); valueIg=getPntrToComponent("Ig");
  addComponent("I2"); componentIsNotPeriodic("I2"); valueI2=getPntrToComponent("I2");
  addComponent("I4"); componentIsNotPeriodic("I4"); valueI4=getPntrToComponent("I4");
  addComponent("force2"); componentIsNotPeriodic("force2"); valueForceTot2=getPntrToComponent("force2");
  addComponent("omega");  componentIsNotPeriodic("omega");  valueOmega=getPntrToComponent("omega");
  std::ostringstream oss;
  for (unsigned int i=0; i<dim_; i++)
  {
    oss.str(""); oss<<"GradOmega-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueGradOmega[i]=getPntrToComponent(oss.str());
    oss.str(""); oss<<"InstCoeff-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueCoeff[i]=getPntrToComponent(oss.str());
    valueCoeff[i]->set(coeff_[i]);
    oss.str(""); oss<<"MeanCoeff-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueMeanCoeff[i]=getPntrToComponent(oss.str());
    valueMeanCoeff[i]->set(mean_coeff_[i]);
    oss.str(""); oss<<"FesCoeff-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueFesCoeff[i]=getPntrToComponent(oss.str());
    valueFesCoeff[i]->set(fes_coeff_[i]);
  }

//printing some info
  log.printf("  Bolzman Temperature (Kb*T): %g\n",1./beta_);
  log.printf("  Beta (1/Kb*T): %g\n",beta_);
  log.printf("  The considered Landau integrals are %d\n",dim_);
  log.printf("  The maximum possile value of N2_MAX is %d, the requested one is %d\n",max_n2_max_,n2_max_);
  if (rho_c_==-1)
    log.printf("  Using PSI_ZERO=%f\n",psi_zero_);
  else
    log.printf("  Using PSI_ZERO=1-RHO_C*V/N, where RHO_C=%f\n",rho_c_);
  log.printf("  The maximum square amplitudes are MAX_AMPL^2, where\n  ");
  for(unsigned int l=0; l<n2_max_; l++)
    log.printf("\t|k|=%d->max_ampl^2=%g",l+1,max_ampl2_[fwv_sizes_[l]-1]);
  if (inv_gamma_==0)
  {
    if(manual_target_coeff_[0]!=0 || manual_target_coeff_[1]!=0)
      log.printf("\n  MANUAL_TG_COEFFS=%g,%g will be used to determine fixed target averages\n",manual_target_coeff_[0],manual_target_coeff_[1]);
    else
      log.printf("\n  GAMMA not set, uniform target distribution will be used\n");
  }
  else
    log.printf("\n  The target distribution uses a factor GAMMA=%g\n", gamma);
  log.printf("  Initial guess for the Free Energy Surface coefficients:\n");
  for(unsigned int i=0; i<dim_; i++)
    log.printf("    FesCoeff%d: %g\t",i+1,fes_coeff_[i]);
  log.printf("\n  Steps for the minimization algorithm:\n");
  for(unsigned int i=0; i<dim_; i++)
    log.printf("    Coeff%d: %g",i+1,minimization_step_[i]);
  log.printf("\n  Stride for the ensemble average: %d\n",av_stride_);
  if (inv_gamma_!=0)
    log.printf("  Stride for the target average: the one above times %d\n",target_stride_);
  if (mean_counter_>1)
    log.printf("  Mean coefficient calculated as if mean_counter*av_stride=%d steps were already performed\n",mean_counter_*av_stride_);
  if (mean_weight_tau_>0)
    log.printf("  Exponentially decaing average with weight=tau/av_stride=%d\n",mean_weight_tau_);
  if(walkers_num_>1)
  {
    log.printf("  Using multiple walkers\n");
    log.printf("    number of walkers: %d\n",walkers_num_);
    log.printf("    walker rank: %d\n",multi_sim_comm.Get_rank()); //only comm.Get_rank()=0 will print, so is fine
  }
  if (no_multiple_walkers)
    log.printf("  -- NO_MULTIPLE_WALKERS: multiple simulations will not communicate\n");
  if(parallel_)
    log.printf("  -- PARALLEL: running with loop parallelization\n");
  if(no_bias_)
    log.printf("  -- NO_BIAS: no bias will be applied (cut off at MAX_AMPL is still peformed)\n");
}

void VariationalLandau_DIS::calculate()
{
  if (first_run_) //this exists only because the function getBox() does not work before here (see LandauFourierComp)
  {
    first_run_=false;
    //a trick to get some info directly from the colvar. It also avoids wrong uses of this bias
    std::vector<colvar::LandauFourierComp*> pFourierComp;
    pFourierComp=plumed.getActionSet().select<colvar::LandauFourierComp*>();
    plumed_massert(pFourierComp.size()==1,"This bias can only be used with one colvar \"LandauFourierComp\"");
    plumed_massert(fwv_size_==pFourierComp[0]->getFwvSize(),"Number of ARG not maching the used colvar. All the colvar output should be used as ARG");
    //get the address of the fourier components (avoids useless copies)
    Re_=pFourierComp[0]->getRe_adr();
    Im_=pFourierComp[0]->getIm_adr();
    //get the volume
    Vol_=pFourierComp[0]->getVol();
    if (rho_c_!=-1)
    {
      psi_zero_=1-rho_c_*Vol_/pFourierComp[0]->getNumAtom();
      log.printf("  Using psi_0 calculated from rho_c: PSI_ZERO=%f\n",psi_zero_);
    }
    //initialize k2_
    const double k2_const=4*PLMD::pi*PLMD::pi/pow(Vol_,2./3.);
    for (unsigned int l=0; l<n2_max_; l++)
      k2_.resize(fwv_sizes_[l],(l+1)*k2_const);//not the fastest way of doing it, but it is done only once
    //if needed, initialize target_av_LI_ (needs Vol_)
    if (inv_gamma_==0) //FIXME to simplify the semi-WT scheme might be taken away...
    {
      if(manual_target_coeff_[0]!=0 || manual_target_coeff_[1]!=0)
      { //use the manual_target_coeff_ to initialize target_av_LI_
        const double temp0=fes_coeff_[0];
        const double temp1=fes_coeff_[1];
        fes_coeff_[0]=manual_target_coeff_[0];
        fes_coeff_[1]=manual_target_coeff_[1];
        inv_gamma_=1;
        update_target_av();
        inv_gamma_=0;
        fes_coeff_[0]=temp0;
        fes_coeff_[1]=temp1;
      }
      else
        update_target_av();
    }
    log.printf("\n");
  }

  if (amplitude_cutoff_) //check if out of range only if a range was set
  {
    //check if the updated fourier amplitudes are in range. if not, set the forces to zero and stop.
    //  this approach speeds up convergence (no extra forces are added if the CVs are big) and,
    //  if MAX_AMPL has a reasonable value, once close to convergence no data should be discarded
    int out_of_range=0; //cannot use bool with communicators. 0=false, !0=true
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      if (Re_[k]*Re_[k]+Im_[k]*Im_[k]>max_ampl2_[k])//the physical quantity is the module
      {
        out_of_range=1; //if one k is out, skip this point
        log.printf("  --- WARNING --- k=%d out of range\n",k);
        unsigned int p=2*k;//set to zero only the forces on this k
        if (extra_arg_)
          p++;
        setOutputForce(p,0.);
        setOutputForce(p+1,0.);
      }
    }
    if (walkers_num_>1)//all the walkers must be syncronized
    {
      if (comm.Get_rank()==0)
        multi_sim_comm.Sum(out_of_range);//if one walker is out, skip this point also for the others
      if (comm.Get_size()>1)
        comm.Bcast(out_of_range,0);
    }
    if(out_of_range)
    {
      valueForceTot2->set(0.); //this to remind that something is not good
      return;
    }
  }

//else go haed and calculate landau integrals and forces
  double Landau_int[dim_]; //integrals are rescaled over the volume
  calc_landau_int_and_set_forces(Landau_int);
  valueIg->set(Landau_int[0]);
  valueI2->set(Landau_int[1]);
  valueI4->set(Landau_int[2]);

//update stuff. has to be done after the forces are setted
  update_ensemble_av(Landau_int);
  if (av_counter_==av_stride_)
  {
    if (inv_gamma_!=0 && (mean_counter_<target_stride_ || mean_counter_%target_stride_==0)) //update only if semi-WT and with a certain stride
      update_target_av();//if is uniform does not change
    update_omega_and_coeffs();
    //reset the ensamble averages
    av_counter_=0;
    av_exp_bias_=0;
    for (unsigned int i=0; i<dim_; i++)
      av_LI_[i]=0;
    for (unsigned int ij=0; ij<av_prod_LI_.size(); ij++)
      av_prod_LI_[ij]=0;
    //update other components values
    for (unsigned int i=0; i<dim_; i++)
    {
      valueCoeff[i]->set(coeff_[i]);
      valueMeanCoeff[i]->set(mean_coeff_[i]);
      valueFesCoeff[i]->set(fes_coeff_[i]);
    }
  }
}

bool VariationalLandau_DIS::check_arg_and_set_n2_max(unsigned int arg_number) //FIXME maybe here could put the automatic calculation of fwv_sizes_, so that any n2_max can be given
{
//check if arg_number corresponds to any of the known sizes
  for (unsigned int l=0; l<max_n2_max_; l++)
  {
    if (arg_number==2*fwv_sizes_[l])
    {
      n2_max_=l+1;
      return true;
    }
  }
  return false;
}

void VariationalLandau_DIS::initialize_conv_pair(unsigned int n2_max)
{
//generate all the fourier wave vectors up to m2_max
  std::vector<Vector> fwv;
  std::vector<Vector> pos;
  const int m2_max=4*n2_max; //triangle inequality gives this upper bound
  const int max_int=floor(sqrt(m2_max));
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
      if ((x*x+y*y)<=m2_max) //this saves some calc
      {
        for(int z=start; z<=max_int; z++)
        {
          if ((x*x+y*y+z*z)<=m2_max)
          {
            new_point[0]=x;
            new_point[1]=y;
            new_point[2]=z;
            fwv.push_back(new_point);
          }
        }
      }
    }
  }
  std::stable_sort(fwv.begin(),fwv.end(),Landau_VectorsCompare); //ordered is nicer

//individuating the pairs of fwv needed for the convolution:
//  for any given q there are conv_pair_[q].size() different pairs of indexes (k -> \vec_{k} and kk -> \vec_{q}-\vec_{k})
//  that give a non-zero contribution to the convolution. complicances are due to the fact that in fwv there are only half of the frequencies.
  conv_pair_.resize(fwv.size());
  for (unsigned int q=0; q<conv_pair_.size(); q++)
  {
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      for (unsigned int kk=k; kk<fwv_size_; kk++) //taking only ordered pairs k<=kk
      {
        if ((fwv[q]-fwv[k]-fwv[kk]).modulo2()==0) // q-k=kk in fwv
        {
          indexes_pair new_pair(k,+1,kk,+1);
          conv_pair_[q].push_back(new_pair);
        }
        else if ((fwv[q]+fwv[k]-fwv[kk]).modulo2()==0) // q+k=kk in fwv
        {
          indexes_pair new_pair(k,-1,kk,+1);
          conv_pair_[q].push_back(new_pair);
        }
        else if ((fwv[q]-fwv[k]+fwv[kk]).modulo2()==0) // q-k=-kk in -fwv
        {
          indexes_pair new_pair(k,+1,kk,-1);
          conv_pair_[q].push_back(new_pair);
        }
        // the case "q+k=-kk in -fwv" cannot happen because the first non-zero element of any k is always positive
      }
    }
  }

  m_max_=1+4*(floor(sqrt(n2_max)));
  for (unsigned x=0; x<m_max_; x++)
  {
    for (unsigned y=0; y<m_max_; y++)
    {
      for(unsigned z=0; z<m_max_; z++)
      {
        new_point[0]=x;
        new_point[1]=y;
        new_point[2]=z;
        pos.push_back(new_point);
      }
    }
  }
  pos_size_=pos.size();//FIXME: pos_size_==pow(m_max_,3)
  cos_km_.resize(fwv_size_*pos_size_);
  sin_km_.resize(fwv_size_*pos_size_);

  for (unsigned m=0; m<pos_size_; m++)
  {
    for (unsigned k=0; k<fwv_size_; k++)
    {
      double xi_km=2*PLMD::pi/m_max_*dotProduct(fwv[k],pos[m]);
      cos_km_[k+m*fwv_size_]=cos(xi_km);
      sin_km_[k+m*fwv_size_]=sin(xi_km);
    }
  }
}

void VariationalLandau_DIS::update_target_av()
{
  for (unsigned int i=0; i<dim_; i++)
    target_av_LI_[i]=0;
//first calculate the moments over p(s) of the square modulus of the order parameter
  std::vector<double> av_r2(fwv_size_); // (r_k)^2=(Re[rho_k])^2+(Im[rho_k])^2
  std::vector<double> av_r4(fwv_size_); // <(r_k)^2>, <(r_k)^4>
//the average is done in a uniform way or semi-WT dropping the I4 term
  if(inv_gamma_==0 || (fes_coeff_[0]==0 && fes_coeff_[1]==0))
  { //uniform
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      av_r2[k]=1./2.*max_ampl2_[k];
      av_r4[k]=1./3.*max_ampl2_[k]*max_ampl2_[k];
    }
  }
  else
  { //semi-well-tempered with Fes'=g*Ig+a*I2 instead of Fes=g*Ig+a*I2+b*I4
    for (unsigned int k=0; k<fwv_size_; k++)
    {
      const double weight_k=beta_*inv_gamma_*2*Vol_*(fes_coeff_[0]*k2_[k]+fes_coeff_[1]);
      const double exp_den_k=exp(weight_k*max_ampl2_[k])-1;
      av_r2[k]=1./weight_k-max_ampl2_[k]/exp_den_k;
      av_r4[k]=2./(weight_k*weight_k)-max_ampl2_[k]*(max_ampl2_[k]+2./weight_k)/exp_den_k;
    }
  }
//now use them to evaluate the averages of the integrals
  for (unsigned int q=0; q<conv_pair_.size(); q++)
  {
    if (q<fwv_size_)
    {
      const unsigned int k=q;
      target_av_LI_[0]+=2*k2_[k]*av_r2[k];
      target_av_LI_[1]+=2*av_r2[k];
      target_av_LI_[2]+=4*av_r4[k];
      for (unsigned int kk=k+1; kk<fwv_size_; kk++)
        target_av_LI_[2]+=8*av_r2[k]*av_r2[kk];
    }
    for (unsigned int h=0; h<conv_pair_[q].size(); h++)
    {
      const unsigned int k=conv_pair_[q][h].k;
      const unsigned int kk=conv_pair_[q][h].kk;
      if(k==kk)
        target_av_LI_[2]+=2*av_r4[k];
      else
        target_av_LI_[2]+=8*av_r2[k]*av_r2[kk];
    }
  }
  target_av_LI_[1]+=pow(psi_zero_,2);
  target_av_LI_[2]+=pow(psi_zero_,2)*(6*target_av_LI_[1]-5*pow(psi_zero_,2));

  log.printf("  Averages over the uniform target distribution: <Ig>=%f\t<I2>=%f\t<I4>=%f\n",target_av_LI_[0],target_av_LI_[1],target_av_LI_[2]);
}

void VariationalLandau_DIS::calc_landau_int_and_set_forces(double* lan_int)
{
//initialize to zero needed variables (important if parallel)
  double Kin_int=0.;
  double Quad_int=0.;
  double Quart_int=0.;
  std::vector<double> psiTilde(pos_size_,0); //is the anti-Fourier transform evaluated on the discrete points

  for (unsigned k=0; k<fwv_size_; k++)
  {
    Kin_int+=2*k2_[k]*(Re_[k]*Re_[k]+Im_[k]*Im_[k]);
    Quad_int+=2*(Re_[k]*Re_[k]+Im_[k]*Im_[k]);
  }
  Quad_int+=psi_zero_*psi_zero_;
//build the quartic
  for (unsigned m=0; m<pos_size_; m++)
  {
    for (unsigned k=0; k<fwv_size_; k++)
      psiTilde[m]+=2*(cos_km_[k+m*fwv_size_]*Re_[k]-sin_km_[k+m*fwv_size_]*Im_[k]);
    psiTilde[m]+=psi_zero_;
    Quart_int+=pow(psiTilde[m],4.);
  }
  Quart_int/=pow(m_max_,3);

//  unsigned int rank=0;
//  unsigned int stride=1;
//  if (parallel_)
//  {
//    rank=comm.Get_rank();
//    stride=comm.Get_size();
//  }
//  for (unsigned int q=rank; q<conv_pair_.size(); q+=stride) //PARALLEL LOOP
//  {
//    double Re_psi2_q=0.; //actually these are rescaled: V*psi2_q, V*d_psi2_q
//    double Im_psi2_q=0.;
//    if (q<fwv_size_)
//    {
//      unsigned int k=q; //just to avoid confusion: is below fwv_size_
//      Kin_int+=2*k2_[k]*(Re_[k]*Re_[k]+Im_[k]*Im_[k]);
//      Quad_int+=2*(Re_[k]*Re_[k]+Im_[k]*Im_[k]);
//
//      Re_psi2_q+=2*psi_zero_*Re_[k]; //there are always two zero cases: k->0, q-k->0
//      Im_psi2_q+=2*psi_zero_*Im_[k];
//    }
//    for (unsigned int h=0; h<conv_pair_[q].size(); h++)
//    {
//    //just to be less verbose. hopefully the compiler is smart enough to optimize
//      unsigned int k=conv_pair_[q][h].k;
//      unsigned int kk=conv_pair_[q][h].kk;
//      int sign_k=conv_pair_[q][h].sign_k;
//      int sign_kk=conv_pair_[q][h].sign_kk;
//      int mlt=2; //multiplicity, because the sum is over ordered pairs
//      if (k==kk)
//        mlt=1;
//
//      Re_psi2_q+=mlt*(Re_[k]*Re_[kk]-sign_k*Im_[k]*sign_kk*Im_[kk]);
//      Im_psi2_q+=mlt*(Re_[k]*sign_kk*Im_[kk]+sign_k*Im_[k]*Re_[kk]);
//    }
//    Quart_int+=2*(Re_psi2_q*Re_psi2_q+Im_psi2_q*Im_psi2_q);
//  //now with psi2_q can start building the derivatives
//    if (!no_quartic_)
//    {
//      if (q<fwv_size_)
//      {
//        unsigned int k=q; //just to avoid confusion: is below fwv_size_
//        dQuart_dRe[k]+=psi_zero_*Re_psi2_q;
//        dQuart_dIm[k]+=psi_zero_*Im_psi2_q;
//      }
//      for (unsigned int h=0; h<conv_pair_[q].size(); h++)
//      {
//        unsigned int k=conv_pair_[q][h].k;
//        unsigned int kk=conv_pair_[q][h].kk;
//        int sign_k=conv_pair_[q][h].sign_k;
//        int sign_kk=conv_pair_[q][h].sign_kk;
//           // k==kk -> sign_k==sign_kk==+1 since (0,0,0) is not in fwv_
//        dQuart_dRe[k]+=(Re_psi2_q*Re_[kk]+Im_psi2_q*Im_[kk]*sign_kk);
//        dQuart_dIm[k]+=(Im_psi2_q*Re_[kk]-Re_psi2_q*Im_[kk]*sign_kk)*sign_k;
//        if (k!=kk) //otherwise the mlt would not be compensated
//        {
//           dQuart_dRe[kk]+=(Re_psi2_q*Re_[k]+Im_psi2_q*Im_[k]*sign_k);
//           dQuart_dIm[kk]+=(Im_psi2_q*Re_[k]-Re_psi2_q*Im_[k]*sign_k)*sign_kk;
//        }
//      }
//    }
//  }
//  if (parallel_)//CAUTION: not tested with multi walkers
//  {
//    comm.Sum(Kin_int);
//    comm.Sum(Quad_int);
//    comm.Sum(Quart_int);
//    comm.Sum(dQuart_dRe);
//    comm.Sum(dQuart_dIm);
//  }
//  Quad_int+=psi_zero_*psi_zero_;
//  Quart_int+=Quad_int*Quad_int;//the q=(0,0,0) component is equal to Quad_int

//set the forces
//  if (!no_bias_)
//  {
  double tot_force2=0;
  for (unsigned int k=0; k<fwv_size_; k++)
  {
    unsigned int Rk=2*k;
    unsigned int Ik=2*k+1;
    if (extra_arg_)
    {
      Rk++;
      Ik++;
    }
    double dKin_dRe_k=4*Vol_*k2_[k]*Re_[k];
    double dKin_dIm_k=4*Vol_*k2_[k]*Im_[k];
    double dQuad_dRe_k=4*Vol_*Re_[k];
    double dQuad_dIm_k=4*Vol_*Im_[k];
    double dQuart_dRe_k=0;
    double dQuart_dIm_k=0;
    for (unsigned m=0; m<pos_size_; m++)
    {
      dQuart_dRe_k+=pow(psiTilde[m],3)*cos_km_[k+m*fwv_size_];
      dQuart_dIm_k-=pow(psiTilde[m],3)*sin_km_[k+m*fwv_size_];
    }
    dQuart_dRe_k*=8*Vol_/pow(m_max_,3);
    dQuart_dIm_k*=8*Vol_/pow(m_max_,3);

//    //corrections to prime ones
//      dQuart_dRe[k]+=psi_zero_*(6*psi_zero_*dQuad_dRe_k+4*dCubic_dRe[k]);
//      dQuart_dIm[k]+=psi_zero_*(6*psi_zero_*dQuad_dIm_k+4*dCubic_dIm[k]);

    double force_Rk=-mean_coeff_[0]*dKin_dRe_k-mean_coeff_[1]*dQuad_dRe_k-mean_coeff_[2]*dQuart_dRe_k;
    double force_Ik=-mean_coeff_[0]*dKin_dIm_k-mean_coeff_[1]*dQuad_dIm_k-mean_coeff_[2]*dQuart_dIm_k;
    if (!no_bias_)
    {
      setOutputForce(Rk,force_Rk);
      setOutputForce(Ik,force_Ik);
      tot_force2+=pow(force_Rk,2)+pow(force_Ik,2);
    }
    else
      tot_force2=dQuart_dRe_k+dQuart_dIm_k; //XXX for debug only!!
  }
  valueForceTot2->set(tot_force2);
//  }
//store the values of the integrals
  lan_int[0]=Kin_int;
  lan_int[1]=Quad_int;
  lan_int[2]=Quart_int;
}

void VariationalLandau_DIS::update_ensemble_av(double *lan_int)
{
//evaluate bias potential
  double bias_pot=0.;
  for (unsigned int i=0; i<dim_; i++)
    bias_pot+=mean_coeff_[i]*Vol_*lan_int[i];
  setBias(bias_pot);
//update ensable averages
  av_counter_++;
  for (unsigned int i=0; i<dim_; i++)
  {
    av_exp_bias_+=(exp(beta_*bias_pot)-av_exp_bias_)/av_counter_;
    av_LI_[i]+=(lan_int[i]-av_LI_[i])/av_counter_;
    for (unsigned int j=i; j<dim_; j++)
      av_prod_LI_[get_index(i,j)]+=(lan_int[i]*lan_int[j]-av_prod_LI_[get_index(i,j)])/av_counter_;
  }
}

void VariationalLandau_DIS::update_omega_and_coeffs()
{
//combining the averages of multiple walkers
  if(walkers_num_>1)
  {
    if(comm.Get_rank()==0) //sum only once: in the first rank of each walker
    {
      multi_sim_comm.Sum(av_exp_bias_);
      multi_sim_comm.Sum(av_LI_);
      multi_sim_comm.Sum(av_prod_LI_);
      av_exp_bias_/=walkers_num_;
      for(unsigned int i=0; i<dim_; i++)
        av_LI_[i]/=walkers_num_;
      for(unsigned int ij=0; ij<av_prod_LI_.size(); ij++)
        av_prod_LI_[ij]/=walkers_num_; //WARNING: is this the best way to implement mw into this algorithm? some theoretical work should be done...
    }
    if (comm.Get_size()>1)//if there are more ranks for each walker, everybody has to know
    {
      comm.Bcast(av_exp_bias_,0);
      comm.Bcast(av_LI_,0);
      comm.Bcast(av_prod_LI_,0);
    }
  }
//a small trik needed for NO_QUARTIC keyword
  unsigned int special_dim=dim_;
  if (no_quartic_)
    special_dim--;

//update Omega. NOTICE that is actually rescaled to Omega/Vol
  double omega=-1*std::log(av_exp_bias_)/beta_/Vol_; //there is also a PLMD::log function
  for (unsigned int i=0; i<special_dim; i++)
    omega+=mean_coeff_[i]*target_av_LI_[i];
  valueOmega->set(omega);

//build the gradient and the hessian of the functional
  std::vector<double> grad_omega(dim_);
  std::vector<double> hess_omega_increm(dim_);//inner product between the hessian and the increment
  mean_counter_++;
  unsigned int mean_weight=mean_counter_;
  if (mean_weight_tau_>0 && mean_weight_tau_<mean_counter_)
    mean_weight=mean_weight_tau_;
  for (unsigned int i=0; i<special_dim; i++)
  {
    grad_omega[i]=target_av_LI_[i]-av_LI_[i];
    for(unsigned int j=0; j<special_dim; j++)
      hess_omega_increm[i]+=(av_prod_LI_[get_index(i,j)]-av_LI_[i]*av_LI_[j])*(coeff_[j]-mean_coeff_[j]);
    hess_omega_increm[i]*=beta_*Vol_;
  }
  for (unsigned int i=0; i<special_dim; i++)
  {
//update all the coefficients
    coeff_[i]-=minimization_step_[i]*(grad_omega[i]+hess_omega_increm[i]);
    mean_coeff_[i]+=(coeff_[i]-mean_coeff_[i])/mean_weight;
    if (i!=2) //the original condition is (i==0 || i==1)
      fes_coeff_[i]=inv_gamma_*fes_coeff_[i]+manual_target_coeff_[i]-mean_coeff_[i]; //always inv_gamma_ or manual_target_coeff_ will be equal to zero
    else
      fes_coeff_[i]=-1*mean_coeff_[i];
    //update also GradOmega
    valueGradOmega[i]->set(grad_omega[i]);
  }
}

//since the communincators use std::vector is more convenient to avoid multiple indexes, that's why this function exists
inline unsigned int VariationalLandau_DIS::get_index(unsigned int i, unsigned int j) const //mapping of a (dim_)x(dim_) symmetric matrix into a vector
{
  if (i<=j)
    return j+i*(dim_-1)-i*(i-1)/2;
  else
    return get_index(j,i);
}

}
}


/***********************************************
  Here are the total number of fourier wave
  vectors k given different N2_MAX:

  n2_max_  fwv_size_  multiplicity
     1         3           3
     2         9           6
     3        13           4
     4        16           3
     5        28          12
     6        40          12
     7        40           0
     8        46           6
     9        61          15
    10        73          12
    11        85          12
    12        89           4
    13       101          12
    14       125          24
    15       125           0
    16       128           3
    17       152          24
    18       170          18
    19       182          12
    20       194          12
    21       218          24
    22       230          12
    23       230           0
    24       242          12
    25       257          15
    26       293          36
    27       309          16
    28       309           0
    29       345          36
    30       369          24
    31       369           0
    32       375           6
    33       399          24
    34       423          24
    35       447          24
    36       462          15
    37       474          12
    38       510          36
    39       510           0
    40       522          12

************************************************/
