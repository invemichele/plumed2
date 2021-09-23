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
#include "../colvar/DensityFourierComp.h"
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/ActionSet.h"
#include "core/Atoms.h"
#include "tools/Communicator.h"

#include <sstream>   // std::ostringstream

using namespace std;

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS DENSITY_FOURIER_VAR
/*
 * ...
 * ...
 *
 * The used collective variables are the Fourier amplitudes of the rescaled density up to a maximum wave vector.
 *
 * NB: regular expression can be used in plumed.dat, thus if the label is   var: DENSITY_FOURIER_VAR ....
 *     we can print in this way:    PRINT FMT=%g FILE=RhoSquare.data ARG=(var\.RhoSquare-[0-9]*)
*/
//+ENDPLUMEDOC

//now the bias class
class DensityFourierVar : public Bias {

private:
  bool first_run_;
  bool amplitude_cutoff_;
  bool no_bias_;
//  bool parallel_;
  unsigned walkers_num_;

  double Vol_;
  unsigned n2_max_;
  unsigned fwv_size_;
  std::vector<unsigned> fwv_mult_; //the multiplicity for each k-shell
  std::vector<unsigned> i_index_; //to go from k to i
  const double* Re_;
  const double* Im_;
  std::vector<double> max_ampl2_;

  double beta_;
  double inv_gamma_;
  std::vector<double> minimization_step_;
  std::vector<double> inst_coeff_, mean_coeff_, fes_coeff_;
  std::vector<double> target_av_LI_;  //'target' -> 'target distribution', p(s)
  std::vector<double> av_LI_, av_prod_LI_;//'av' -> 'ensemble average'
  double av_exp_bias_;
  unsigned av_counter_, av_stride_;
  unsigned target_stride_;
  unsigned mean_counter_;
  unsigned mean_weight_tau_;
  std::vector<double> manual_target_coeff_;
  bool using_manual_tg_;

  Value* valueForceTot2;
  Value* valueOmega;
  std::vector<Value*> valueGradOmega;
  std::vector<Value*> valueRhoSquare;
  std::vector<Value*> valueInstCoeff;
  std::vector<Value*> valueMeanCoeff;
  std::vector<Value*> valueFesCoeff;

//class private methods
  bool check_arg_and_set_n2_max(unsigned,std::vector<unsigned>&);
  void set_RhoSquare_and_Forces(std::vector<double>&);
  void update_ensemble_av(std::vector<double>&);
  void update_target_av();
  void update_omega_and_coeffs();
  inline unsigned get_index(unsigned, unsigned) const;

public:
  DensityFourierVar(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(DensityFourierVar,"DENSITY_FOURIER_VAR")

void DensityFourierVar::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.addOutputComponent("force2","default","the instantaneous value of the squared total force");
  keys.addOutputComponent("omega","default","the value of the functional Omega");

  keys.addOutputComponent("RhoSquare","default","the square modulus of the CVs averaged over a k-shell");
  keys.addOutputComponent("GradOmega","default","gradient of Omega, the functional to be minimized");
  keys.addOutputComponent("InstCoeff","default","istantaneous coefficient of the bias potential");
  keys.addOutputComponent("MeanCoeff","default","mean coefficient of the bias potential");
  keys.addOutputComponent("FesCoeff","default","Landau coefficient of the free energy");
  ActionWithValue::useCustomisableComponents(keys); //needed to have an unknown number of components

  keys.use("ARG");
  keys.add("optional","TEMP","temperature");
  keys.add("optional","MAX_AMPL","maximum amplitude of the fourier components, can be a different one for each different |k|^2 (N2_MAX in total). default is 1");
  keys.add("optional","GAMMA","bias factor for the well-tempered sampling");
  keys.add("optional","MANUAL_TG_COEFFS","set the averages over the target distribution through fixed coefficients");
  keys.add("optional","AV_STRIDE","the number of steps between updating coeffs");
  keys.add("optional","TG_STRIDE","tune how often the average distribution is updated in WT scheme. Default is each time");
  keys.add("optional","MEAN_COUNTER","avoid initial kink to the mean coefficient value when continuing a run");
  keys.add("optional","TAU_MEAN","exponentially decaing average for the mean coefficients: all old points count as (tau-1) points. rescaled using av_stride");
  keys.add("optional","M_STEP","the step used for the minimization of the functional");
  keys.add("optional","INITIAL_FES_COEFFS","an initial guess for the Free Energy Surface coefficients");

  keys.addFlag("NO_MULTIPLE_WALKERS",false,"do not use Multiple Walkers, even if there are multiple simulations running");
  keys.addFlag("NO_BIAS",false,"do not apply any bias. MAX_AMPL still applies");
  keys.addFlag("PARALLEL",false,"perform the calculation in parallel - CAUTION: could give worse performance");
}

DensityFourierVar::DensityFourierVar(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao)
{
//initialize n2_max_ and check for consistency with the used colvar
  std::vector<unsigned> fwv_sizes{3,9,13,16,28,40,40,46,61,73,85,89,101,125,125,128,152,170,182,194};//leftover from the past: a few possible sizes are given. Not a problem if two have same size
  const unsigned arg_num=getNumberOfArguments();
  const bool successful=check_arg_and_set_n2_max(arg_num,fwv_sizes);
  plumed_massert(successful,"the number of arguments dosen't match any of the predetermined. You are either trying to bias the wrong colvar or not giving as ARG all the components of the \"DensityFourierComp\" colvar");

//initialize other variables
  first_run_=true;
  av_exp_bias_=0;
  av_counter_=0;

  fwv_size_=fwv_sizes[n2_max_-1];
  for (unsigned i=0; i<n2_max_; i++)
    i_index_.resize(fwv_sizes[i],i);
  fwv_mult_.resize(n2_max_);
  fwv_mult_[0]=fwv_sizes[0];
  for (unsigned ii=1; ii<n2_max_; ii++)
    fwv_mult_[ii]=fwv_sizes[ii]-fwv_sizes[ii-1];
  inst_coeff_.resize(n2_max_);
  mean_coeff_.resize(n2_max_);
  target_av_LI_.resize(n2_max_);
  av_LI_.resize(n2_max_);
  av_prod_LI_.resize(n2_max_*(n2_max_+1)/2); //diagonal matrix mapped into a vector (convenient for parallel loops)

  valueRhoSquare.resize(n2_max_);
  valueGradOmega.resize(n2_max_);
  valueInstCoeff.resize(n2_max_);
  valueMeanCoeff.resize(n2_max_);
  valueFesCoeff.resize(n2_max_);

//parsing
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
    max_ampl2_.resize(n2_max_,1);
  }
  else if(max_ampl.size()==1)
    max_ampl2_.resize(n2_max_,pow(max_ampl[0],2));
  else
  {
    plumed_massert(max_ampl.size()==n2_max_,"ERROR: MAX_AMPL should have either one or N2_MAX arguments");
    max_ampl2_.resize(n2_max_);
    for (unsigned i=0; i<n2_max_; i++)
      max_ampl2_[i]=pow(max_ampl[i],2);
  }

  double gamma=0;
  parse("GAMMA",gamma);
  if (gamma==0)
    inv_gamma_=0; //default or GAMMA=0 -> inv_gamma_=0 that gives non WT sampling
  else
  {
    plumed_massert(gamma>=1, "ERROR: gamma has to be greater than one for the well-tempered sampling to work");
    inv_gamma_=1./gamma;
  }

  using_manual_tg_=false;
  parseVector("MANUAL_TG_COEFFS",manual_target_coeff_);
  if (manual_target_coeff_.size()!=0)
  {
    if (manual_target_coeff_.size()==1)
      manual_target_coeff_.resize(n2_max_,manual_target_coeff_[0]);
    plumed_massert(manual_target_coeff_.size()==n2_max_,"ERROR: number of MANUAL_TG_COEFFS must be 1 or N2_MAX");
    plumed_massert(inv_gamma_==0,"ERROR: Well-Tempered scheme is not compatible with fixed target average. Set GAMMA or MANUAL_TG_COEFFS, not both");
    using_manual_tg_=true;
    for (unsigned i=0; i<n2_max_; i++)
    {
      if (fwv_mult_[i]==0 && manual_target_coeff_[i]!=0)
      {
        log.printf("  - WARNING: setting to zero the manual_target_coeff_ related to the empy k-shell n_M=%d\n",i);
        manual_target_coeff_[i]=0;
      }
    }
  }
  else
    manual_target_coeff_.resize(n2_max_,0);

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
    minimization_step_.resize(n2_max_,m_step);
  else
    plumed_massert(minimization_step_.size()==n2_max_,"ERROR: number of arguments for M_STEP must be 1 or match number of coefficients N2_MAX");

  parseVector("INITIAL_FES_COEFFS",fes_coeff_);
  if (fes_coeff_.size()!=0)
  {
    if (fes_coeff_.size()==1)
      fes_coeff_.resize(n2_max_,fes_coeff_[0]);
    plumed_massert(fes_coeff_.size()==n2_max_,"ERROR: number of INITIAL_FES_COEFFS must match number of coefficients N2_MAX");
    for (unsigned i=0; i<n2_max_; i++)
    {
      if (fwv_mult_[i]==0 && fes_coeff_[i]!=0)
      {
        log.printf("  - WARNING: setting to zero the fes_coeff_ related to the empy k-shell n_M=%d\n",i);
        fes_coeff_[i]=0;
      }
    }
  }
  else
    fes_coeff_.resize(n2_max_,0);
//now can initialize the coefficients
  for (unsigned i=0; i<n2_max_; i++)
  {
    inst_coeff_[i]=(inv_gamma_-1)*fes_coeff_[i]+manual_target_coeff_[i];
    mean_coeff_[i]=inst_coeff_[i];
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
//  parallel_=false;
//  parseFlag("PARALLEL",parallel_);
  checkRead();

//add all the output components
  addComponent("force2"); componentIsNotPeriodic("force2"); valueForceTot2=getPntrToComponent("force2");
  addComponent("omega"); componentIsNotPeriodic("omega"); valueOmega=getPntrToComponent("omega");
  std::ostringstream oss;
  for (unsigned i=0; i<n2_max_; i++)
  {
    oss.str(""); oss<<"RhoSquare-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueRhoSquare[i]=getPntrToComponent(oss.str());
    oss.str(""); oss<<"GradOmega-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueGradOmega[i]=getPntrToComponent(oss.str());
    oss.str(""); oss<<"InstCoeff-"<<i+1;
    addComponent(oss.str()); componentIsNotPeriodic(oss.str()); valueInstCoeff[i]=getPntrToComponent(oss.str());
    valueInstCoeff[i]->set(inst_coeff_[i]);
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
  log.printf("  The considered coefficients are %d\n",n2_max_);
  log.printf("  The maximum square amplitudes are MAX_AMPL^2, where |n|^2->max_ampl^2\n  ");
  for(unsigned i=0; i<n2_max_; i++)
    log.printf("  %d-> %g",i+1,max_ampl2_[i]);
  if (inv_gamma_==0)
  {
    if(using_manual_tg_)
      log.printf("\n  MANUAL_TG_COEFFS will be used to determine fixed target averages\n");
    else
      log.printf("\n  GAMMA not set, uniform target distribution will be used\n");
  }
  else
    log.printf("\n  The target distribution uses a factor GAMMA=%g\n", gamma);
  log.printf("  Initial guess for the Free Energy Surface coefficients:\n  ");
  for(unsigned i=0; i<n2_max_; i++)
    log.printf("  FesCoeff%d=%g  ",i+1,fes_coeff_[i]);
  log.printf("\n  Steps for the minimization algorithm:\n  ");
  for(unsigned i=0; i<n2_max_; i++)
    log.printf("  m_step%d=%g",i+1,minimization_step_[i]);
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
//  if(parallel_)
//    log.printf("  -- PARALLEL: running with loop parallelization\n");
  if(no_bias_)
    log.printf("  -- NO_BIAS: no bias will be applied (cut off at MAX_AMPL is still peformed)\n");
}

void DensityFourierVar::calculate()
{
  if (first_run_) //this exists only because the function getBox() does not work before here (see DensityFourierComp)
  {
    first_run_=false;
    //a trick to get some info directly from the colvar. It also avoids wrong uses of this bias
    std::vector<colvar::DensityFourierComp*> pFourierComp;
    pFourierComp=plumed.getActionSet().select<colvar::DensityFourierComp*>();
    plumed_massert(pFourierComp.size()==1,"This bias can only be used with one colvar \"DensityFourierComp\"");
    plumed_massert(fwv_size_==pFourierComp[0]->getFwvSize(),"Number of ARG not maching the used colvar. All the colvar output should be used as ARG");
    //get the address of the fourier components (avoids useless copies)
    Re_=pFourierComp[0]->getRe_adr();
    Im_=pFourierComp[0]->getIm_adr();
    //get the volume
    Vol_=pow(pFourierComp[0]->getBoxEdge(),3);
    //if needed, initialize target_av_LI_ (needs Vol_)
    if (inv_gamma_==0)
    {
      if(using_manual_tg_)
      { //use the manual_target_coeff_ to initialize target_av_LI_
        std::vector<double> temp=fes_coeff_;
        for (unsigned i=0; i<n2_max_; i++)
          fes_coeff_[i]=manual_target_coeff_[i];
        inv_gamma_=1;
        update_target_av();
        inv_gamma_=0;
        for (unsigned i=0; i<n2_max_; i++)
          fes_coeff_[i]=temp[i];
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
    for (unsigned k=0; k<fwv_size_; k++)
    {
      if (Re_[k]*Re_[k]+Im_[k]*Im_[k]>max_ampl2_[i_index_[k]])//the physical quantity is the module
      {
        out_of_range=1; //if one k is out, skip this point
        log.printf("  --- WARNING --- k=%d out of range\n",k);
        setOutputForce(2*k,0.);//set to zero only the forces on this k
        setOutputForce(2*k+1,0.);
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

//else go haed and set the forces
  std::vector<double> square_rho(n2_max_,0);
  set_RhoSquare_and_Forces(square_rho);

//update stuff. has to be done after the forces are setted
  update_ensemble_av(square_rho);
  if (av_counter_==av_stride_)
  {
    if (inv_gamma_!=0 && (mean_counter_<target_stride_ || mean_counter_%target_stride_==0)) //update only if WT and with a certain stride
      update_target_av();//if is uniform does not change
    update_omega_and_coeffs();
    //reset the ensamble averages
    av_counter_=0;
    av_exp_bias_=0;
    for (unsigned i=0; i<n2_max_; i++)
      av_LI_[i]=0;
    for (unsigned ij=0; ij<av_prod_LI_.size(); ij++)
      av_prod_LI_[ij]=0;
    //update other components values
    for (unsigned i=0; i<n2_max_; i++)
    {
      valueInstCoeff[i]->set(inst_coeff_[i]);
      valueMeanCoeff[i]->set(mean_coeff_[i]);
      valueFesCoeff[i]->set(fes_coeff_[i]);
    }
  }
}

bool DensityFourierVar::check_arg_and_set_n2_max(unsigned arg_number, std::vector<unsigned>& fwv_sizes)
{
//check if arg_number corresponds to any of the known sizes
  for (unsigned l=0; l<fwv_sizes.size(); l++)
  {
    if (arg_number==2*fwv_sizes[l])
    {
      n2_max_=l+1;
      return true;
    }
  }
//an inefficient but working way of extending beyond fwv_sizes.size()
  log.printf("  - WARNING: default fwv_sizes.size() exeded, going ahead calculating new possible sizes\n");
  unsigned size=fwv_sizes.back();
  unsigned n2_pivot=fwv_sizes.size();
  while (arg_number>2*size)
  {
    size=0;
    n2_pivot++;
    const int max_int=floor(sqrt(n2_pivot));
    int start;
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
        if ((x*x+y*y)<=int(n2_pivot)) //saves some calculations
        {
          for(int z=start; z<=max_int; z++)
          {
            if ((x*x+y*y+z*z)<=int(n2_pivot))
              size++;
          }
        }
      }
    }
    fwv_sizes.push_back(size);
    if (arg_number==2*size)
    {
      n2_max_=n2_pivot;
      return true;
    }
  }
//if there is no match then something is wrong
  return false;
}

void DensityFourierVar::update_target_av()
{
//first calculate the moments over p(s) of the square modulus of the order parameter
// they are only function of the modulus of the k fwv
// target_av_LI_[i]=<(r_k)^2>=<(Re[rho_k])^2+(Im[rho_k])^2>
  log.printf("  Averages over the uniform target distribution: ");
  for (unsigned i=0; i<n2_max_; i++)
  {
    if (fwv_mult_[i]==0) //some k-shells are empty!
      target_av_LI_[i]=0;
    else if(inv_gamma_==0 || fes_coeff_[i]==0) //extra check: should never get here if inv_gamma_==0
      target_av_LI_[i]=1./2.*max_ampl2_[i];
    else //Well-Tempered (or manual if inv_gamma_==1)
    {
      const double weight_i=beta_*inv_gamma_*Vol_*fes_coeff_[i];
      const double exp_den_i=exp(weight_i*max_ampl2_[i])-1;
      target_av_LI_[i]=1./weight_i-max_ampl2_[i]/exp_den_i;
    }
    log.printf("target_av[%d]=%g  ",i,target_av_LI_[i]);
  }
  log.printf("\n");
}

void DensityFourierVar::set_RhoSquare_and_Forces(std::vector<double>& square_rho) //the vector is supposed to be all zeros
{
  double tot_force2=0;
  for (unsigned k=0; k<fwv_size_; k++)
  {
    const unsigned i=i_index_[k];
    square_rho[i]+=(pow(Re_[k],2)+pow(Im_[k],2))/fwv_mult_[i];

    if (!no_bias_)
    {
      const double force_Rk=-mean_coeff_[i]*Vol_*2*Re_[k]/fwv_mult_[i];
      const double force_Ik=-mean_coeff_[i]*Vol_*2*Im_[k]/fwv_mult_[i];
      setOutputForce(2*k,force_Rk);
      setOutputForce(2*k+1,force_Ik);
      tot_force2+=pow(force_Rk,2)+pow(force_Ik,2);
    }
  }
  valueForceTot2->set(tot_force2);
  for (unsigned i=0; i<n2_max_; i++)
    valueRhoSquare[i]->set(square_rho[i]);
}

void DensityFourierVar::update_ensemble_av(std::vector<double>& square_rho)
{
//evaluate bias potential
  double bias_pot=0.;
  for (unsigned i=0; i<n2_max_; i++)
    bias_pot+=mean_coeff_[i]*Vol_*square_rho[i];
  setBias(bias_pot);
//update ensable averages
  av_counter_++;
  for (unsigned i=0; i<n2_max_; i++)
  {
    av_exp_bias_+=(exp(beta_*bias_pot)-av_exp_bias_)/av_counter_;
    av_LI_[i]+=(square_rho[i]-av_LI_[i])/av_counter_;
    for (unsigned j=i; j<n2_max_; j++)
      av_prod_LI_[get_index(i,j)]+=(square_rho[i]*square_rho[j]-av_prod_LI_[get_index(i,j)])/av_counter_;
  }
}

void DensityFourierVar::update_omega_and_coeffs()
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
      for(unsigned i=0; i<n2_max_; i++)
        av_LI_[i]/=walkers_num_;
      for(unsigned ij=0; ij<av_prod_LI_.size(); ij++)
        av_prod_LI_[ij]/=walkers_num_; //WARNING: is this the best way to implement mw into this algorithm? some theoretical work should be done...
    }
    if (comm.Get_size()>1)//if there are more ranks for each walker, everybody has to know
    {
      comm.Bcast(av_exp_bias_,0);
      comm.Bcast(av_LI_,0);
      comm.Bcast(av_prod_LI_,0);
    }
  }

//update Omega. NOTICE that is actually rescaled to Omega/Vol
  double omega=-1*std::log(av_exp_bias_)/beta_/Vol_; //there is also a PLMD::log function
  for (unsigned i=0; i<n2_max_; i++)
    omega+=mean_coeff_[i]*target_av_LI_[i];
  valueOmega->set(omega);

//build the gradient and the hessian of the functional
  std::vector<double> grad_omega(n2_max_);
  std::vector<double> hess_omega_increm(n2_max_);//inner product between the hessian and the increment
  mean_counter_++;
  unsigned mean_weight=mean_counter_;
  if (mean_weight_tau_>0 && mean_weight_tau_<mean_counter_)
    mean_weight=mean_weight_tau_;
  for (unsigned i=0; i<n2_max_; i++)
  {
    grad_omega[i]=target_av_LI_[i]-av_LI_[i];
    for(unsigned j=0; j<n2_max_; j++)
      hess_omega_increm[i]+=(av_prod_LI_[get_index(i,j)]-av_LI_[i]*av_LI_[j])*(inst_coeff_[j]-mean_coeff_[j]);
    hess_omega_increm[i]*=beta_*Vol_;
    //update all the coefficients
    inst_coeff_[i]-=minimization_step_[i]*(grad_omega[i]+hess_omega_increm[i]);
    mean_coeff_[i]+=(inst_coeff_[i]-mean_coeff_[i])/mean_weight;
    fes_coeff_[i]=inv_gamma_*fes_coeff_[i]+manual_target_coeff_[i]-mean_coeff_[i]; //always inv_gamma_ or manual_target_coeff_ will be equal to zero
    //update also GradOmega
    valueGradOmega[i]->set(grad_omega[i]);
  }
}

//since the communincators use std::vector is more convenient to avoid multiple indexes, that's why this function exists
inline unsigned DensityFourierVar::get_index(unsigned i, unsigned j) const //mapping of a (n2_max_)x(n2_max_) symmetric matrix into a vector
{
  if (i<=j)
    return j+i*(n2_max_-1)-i*(i-1)/2;
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
