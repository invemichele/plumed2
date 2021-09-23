/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2013 The plumed team
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
#include "ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include <cmath>
#include <algorithm> // std::min_element
#include <sstream>   // std::ostringstream

using namespace std;

namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS VARIATIONAL_LINEAR
/*
  Implementation of the variational method with a bias potential linear in the CVs.
  The target distribution is builded following the Well Tempered sheme.
*/
//+ENDPLUMEDOC

class Variational_Linear : public Bias {

private:
  unsigned int dim_; //number of CVs and of coefficients
  std::vector<double> coeffs_, mean_coeffs_, target_coeffs_;
  std::vector<double> av_CV_;        //'av' -> 'ensemble average'
  std::vector<double> av_two_CV_;    //caution: is a matrix (not a great name...)
  std::vector<double> target_av_CV_; //'target' -> 'target distribution', p(s)
  std::vector<double> min_CV_, max_CV_;
  std::vector<double> minimization_step_;

  double beta_, inv_gamma_;
  unsigned int mean_counter_, av_stride_;
  std::vector<unsigned int> av_multi_counter_;
  bool use_mwalkers_;

  Value* valueBias;
  Value* valueForceTot2;
  std::vector<Value*> valueGradOmega;
  std::vector<Value*> valueCoeffs, valueMeanCoeffs;
  std::vector<Value*> valueFesCoeffs, valueTargetCoeffs;

//class private methods
  void updateEnsemble_av(std::vector<double>);
  void updateCoeffs();
  void updateTarget_av();
  unsigned int get_index(unsigned int, unsigned int) const;

public:
  Variational_Linear(const ActionOptions&);
  void calculate();
  static void registerKeywords(Keywords& keys);
};

PLUMED_REGISTER_ACTION(Variational_Linear,"VARIATIONAL_LINEAR")

void Variational_Linear::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.addOutputComponent("bias","default","the instantaneous value of the bias potential");
  keys.addOutputComponent("force2","default","the instantaneous value of the squared total force");
  ActionWithValue::useCustomisableComponents(keys); //Need this because the number of ARG is not given a priori

  keys.use("ARG");
  keys.add("compulsory","MAX_CV","maximum value the collective variables can assume");
  keys.add("compulsory","MIN_CV","minimum value the collective variables can assume");

  keys.add("optional","TEMP","temperature");
  keys.add("optional","GAMMA","biasfactor for the well-tempered sampling");
  keys.add("optional","M_STEP","the step used for the minimization of the functional");
  keys.add("optional","AV_STRIDE","the number of steps between updating coeffs");
  keys.add("optional","INITIAL_TG_COEFFS","an initial guess for the free energy coefficients");

  keys.addFlag("NO_WT",false,"no well-tempered, use instead a uniform target distribution p(s)=1/(max_CV-min_CV)");
  keys.addFlag("MULTIPLE_WALKERS",false,"use Multiple Walkers");
}

Variational_Linear::Variational_Linear(const ActionOptions&ao):
  PLUMED_BIAS_INIT(ao)
{
  dim_=getNumberOfArguments();
//resizing all the vectors
  coeffs_.resize(dim_);
  mean_coeffs_.resize(dim_);
  av_CV_.resize(dim_);
  av_two_CV_.resize(dim_*(dim_+1)/2); //diagonal matrix mapped into a vector
  target_av_CV_.resize(dim_);
  min_CV_.resize(dim_);
  max_CV_.resize(dim_);
  av_multi_counter_.resize(av_two_CV_.size());
  mean_counter_=0;

  parseVector("MAX_CV",max_CV_);
  parseVector("MIN_CV",min_CV_);
  if (max_CV_.size()!=dim_ || min_CV_.size()!=dim_)
    error("ERROR: number of arguments for MAX_CV and MIN_CV must match number of ARG");

  double gamma=10.;
  bool no_WT=false;
  parse("GAMMA",gamma);
  if (gamma==-1) //for the lazy: setting GAMMA=-1 is equivalent to NO_WT
    no_WT=true;
  else
  {
    if (gamma<=1)
      error("ERROR: gamma has to be greater than one for the well-tempered sampling to work");
    parseFlag("NO_WT",no_WT);
  }
  if (no_WT)
    inv_gamma_=0;
  else
    inv_gamma_=1./gamma;

  double temp=0.;
  parse("TEMP",temp);
  double KbT=plumed.getAtoms().getKBoltzmann()*temp;
  if(KbT<=0.0)
  {
    KbT=plumed.getAtoms().getKbT();
    if(KbT<=0.0)
      error("ERROR: unless the MD engine passes the temperature to plumed, you must specify it using TEMP");
  }
  beta_=1.0/KbT; //remember: with LJ use NATURAL UNITS

  use_mwalkers_=false;
  parseFlag("MULTIPLE_WALKERS",use_mwalkers_);
  av_stride_=1000;
  parse("AV_STRIDE",av_stride_);

  parseVector("M_STEP",minimization_step_);
  double m_step=0.01;
  if (minimization_step_.size()==1)
    m_step=minimization_step_[0];
  if (minimization_step_.size()<=1)
    minimization_step_.resize(dim_,m_step);
  else
  {
    if (minimization_step_.size()!=dim_)
      error("ERROR: number of arguments for M_STEP must be 1 or match number of ARG");
  }

  parseVector("INITIAL_TG_COEFFS",target_coeffs_);
  if (target_coeffs_.size()==0)
    target_coeffs_.resize(dim_);
  else
  {
    if (target_coeffs_.size()!=dim_)
      error("ERROR: number of arguments for INITIAL_TG_COEFFS must match number of ARG");
    mean_counter_=1;
    for (unsigned int i=0; i<dim_; i++)
    {
      coeffs_[i]=((inv_gamma_-1.)*target_coeffs_[i]);
      mean_coeffs_[i]=coeffs_[i];
    }
  }
  checkRead();

//adding all the output components
  addComponent("bias");
  componentIsNotPeriodic("bias");
  valueBias=getPntrToComponent("bias");
  addComponent("force2");
  componentIsNotPeriodic("force2");
  valueForceTot2=getPntrToComponent("force2");

  valueCoeffs.resize(dim_);
  valueMeanCoeffs.resize(dim_);
  valueTargetCoeffs.resize(dim_);
  valueFesCoeffs.resize(dim_);
  valueGradOmega.resize(dim_);
  std::ostringstream oss;
  for (unsigned int i=0; i<dim_; i++)
  {
    oss.str("");
    oss<<"InstCoeffs"<<i+1;
    addComponent(oss.str());
    componentIsNotPeriodic(oss.str());
    valueCoeffs[i]=getPntrToComponent(oss.str());
    valueCoeffs[i]->set(coeffs_[i]);
    oss.str("");
    oss<<"MeanCoeffs"<<i+1;
    addComponent(oss.str());
    componentIsNotPeriodic(oss.str());
    valueMeanCoeffs[i]=getPntrToComponent(oss.str());
    valueMeanCoeffs[i]->set(mean_coeffs_[i]);
    oss.str("");
    oss<<"TargetCoeffs"<<i+1;
    addComponent(oss.str());
    componentIsNotPeriodic(oss.str());
    valueTargetCoeffs[i]=getPntrToComponent(oss.str());
    valueTargetCoeffs[i]->set(target_coeffs_[i]);
    oss.str("");
    oss<<"FesCoeffs"<<i+1;
    addComponent(oss.str());
    componentIsNotPeriodic(oss.str());
    valueFesCoeffs[i]=getPntrToComponent(oss.str());
    valueFesCoeffs[i]->set(mean_coeffs_[i]/(inv_gamma_-1.));
    oss.str("");
    oss<<"GradOmega"<<i+1;
    addComponent(oss.str());
    componentIsNotPeriodic(oss.str());
    valueGradOmega[i]=getPntrToComponent(oss.str());
  }

//initializing target_av_CV_
  for (unsigned int i=0; i<dim_; i++)
    target_av_CV_[i]=0.5*(max_CV_[i]+min_CV_[i]);

//printing some info
  log.printf("  Bolzman Temperature (Kb*T): %f\n",1./beta_);
  log.printf("  Beta (1/Kb*T): %f\n",beta_);
  if (no_WT)
    log.printf("  No well-tempering: a uniform target distribution will be used. 1/gamma=0");
  else
    log.printf("  Well-tempering with bias factor gamma: %f\n",1./inv_gamma_);
  log.printf("  Total number of CVs and coefficents: %d\n",dim_);
  log.printf("  Collective Variables used:\n");
  for(unsigned int i=0; i<dim_; i++)
  {
    log.printf("    CV%d: %s",i+1,getPntrToArgument(i)->getName().c_str());
    log.printf("\tmax=%f\tmin=%f\n",max_CV_[i],min_CV_[i]);
  }
  log.printf("  Initial guess for the free energy coefficients:\n");
  for(unsigned int i=0; i<dim_; i++)
    log.printf("    FesCoeff%d: %f\n",i+1,target_coeffs_[i]);
  log.printf("  Step for the minimization algorithm:");
  for(unsigned int i=0; i<dim_; i++)
    log.printf("\t%f",minimization_step_[i]);
  log.printf("\n  Stride for the ensemble average: %d\n",av_stride_);
  if(use_mwalkers_)
  {
    log.printf("  Using multiple walkers\n");
    log.printf("    number of walkers: %d\n",multi_sim_comm.Get_size());
    log.printf("    walker number: %d\n",multi_sim_comm.Get_rank());
  }
}

void Variational_Linear::calculate()
{
  double bias_pot=0.;
  double tot_force2=0.;
  std::vector<double> current_CV(dim_);
//setting the forces and resizing the CV. bias_force=-grad V_{mean_alpha}(s)
  for (unsigned int i=0; i<dim_; i++)
  { //for values outside the range the bias potential is flat
    current_CV[i]=getArgument(i);
    if (current_CV[i]>max_CV_[i])
    {
      setOutputForce(i,0.);
      bias_pot+=mean_coeffs_[i]*max_CV_[i];
    }
    else if(current_CV[i]<min_CV_[i])
    {
      setOutputForce(i,0.);
      bias_pot+=mean_coeffs_[i]*min_CV_[i];
    }
    else
    {
      setOutputForce(i,(-1.)*mean_coeffs_[i]);
      bias_pot+=mean_coeffs_[i]*current_CV[i];
      tot_force2+=mean_coeffs_[i]*mean_coeffs_[i];
    }
  }
  valueBias->set(bias_pot);
  valueForceTot2->set(tot_force2);

//updating stuff. has to be after the force has been set
  updateEnsemble_av(current_CV);
  if (*std::min_element(av_multi_counter_.begin(),av_multi_counter_.end())==av_stride_) //fancy way to be sure that all the averages have at least av_stride_ point
  {
    if (inv_gamma_!=0 && mean_counter_!=0)
      updateTarget_av();
    updateCoeffs();
    //resetting the ensamble averages
    for (unsigned int i=0; i<dim_; i++)
      av_CV_[i]=0;
    for (unsigned int ij=0; ij<av_two_CV_.size(); ij++)
    {
      av_two_CV_[ij]=0;
      av_multi_counter_[ij]=0;
    }
    //getting the updated components values
    for (unsigned int i=0; i<dim_; i++)
    {
      valueCoeffs[i]->set(coeffs_[i]);
      valueMeanCoeffs[i]->set(mean_coeffs_[i]);
      valueTargetCoeffs[i]->set(target_coeffs_[i]);
      valueFesCoeffs[i]->set(mean_coeffs_[i]/(inv_gamma_-1.)); //should become equal to target_coeffs_
    }
  }
}

void Variational_Linear::updateEnsemble_av(std::vector<double> col_var)
{
//updating the ensemble averages only if col_var is in the given range
  for (unsigned int i=0; i<dim_; i++)
  {
    if(col_var[i]>=min_CV_[i] && col_var[i]<=max_CV_[i])
    {
      av_multi_counter_[get_index(i,i)]++;
      av_CV_[i]+=(col_var[i]-av_CV_[i])/av_multi_counter_[get_index(i,i)];
      av_two_CV_[get_index(i,i)]+=(col_var[i]*col_var[i]-av_two_CV_[get_index(i,i)])/av_multi_counter_[get_index(i,i)];
      for (unsigned int j=i+1; j<dim_; j++)
      {
        if(col_var[j]>=min_CV_[j] && col_var[j]<=max_CV_[j])
        {
          av_multi_counter_[get_index(i,j)]++;
          av_two_CV_[get_index(i,j)]+=(col_var[i]*col_var[j]-av_two_CV_[get_index(i,j)])/av_multi_counter_[get_index(i,j)];
        }
      }
    }
  }
}

void Variational_Linear::updateCoeffs()
{
//combining the averages of multiple walkers
  if(use_mwalkers_)
  {
    if(comm.Get_rank()==0) //multi_sim_comm is defined only in the first rank
    {
      int num_walkers=multi_sim_comm.Get_size();
      multi_sim_comm.Sum(av_CV_);
      multi_sim_comm.Sum(av_two_CV_);
      for(unsigned int i=0; i<dim_; i++)
        av_CV_[i] /= num_walkers;
      for(unsigned int ij=0; ij<av_two_CV_.size(); ij++)
        av_two_CV_[ij] /= num_walkers;
    }
    comm.Bcast(av_CV_,0);//the idea is that everybody needs to know
    comm.Bcast(av_two_CV_,0);
  }
//build the gradient and the hessian of the functional
  std::vector<double> grad_omega(dim_);
  std::vector<double> hess_omega_increm(dim_);//inner product between the hessian and the increment
  mean_counter_++;
  for (unsigned int i=0; i<dim_; i++)//NOTE: probably there is room for optimization, but dim_ is very small...
  {
    grad_omega[i]=target_av_CV_[i]-av_CV_[i];
    for(unsigned int j=0; j<dim_; j++)
      hess_omega_increm[i]+=beta_*(av_two_CV_[get_index(i,j)]-av_CV_[i]*av_CV_[j])*(coeffs_[j]-mean_coeffs_[j]);
  }
  for (unsigned int i=0; i<dim_; i++)
  {
//update all the coefficients
    coeffs_[i]-=minimization_step_[i]*(grad_omega[i]+hess_omega_increm[i]);
    mean_coeffs_[i]+=(coeffs_[i]-mean_coeffs_[i])/mean_counter_;
    target_coeffs_[i]=inv_gamma_*target_coeffs_[i]-mean_coeffs_[i];
    //update also the gradOmega value (can't do it elsewhere)
    valueGradOmega[i]->set(grad_omega[i]);
  }
}

void Variational_Linear::updateTarget_av()
{
  for (unsigned int i=0; i<dim_; i++)
  {
    double CV_weight=beta_*inv_gamma_*target_coeffs_[i];
    double exp_factor=exp(CV_weight*(max_CV_[i]-min_CV_[i]));
    if (exp_factor==1) //just an extra check, usually never mached
      target_av_CV_[i]=0.5*(max_CV_[i]+min_CV_[i]);
    else
      target_av_CV_[i]=1.0/CV_weight+(max_CV_[i]-min_CV_[i]*exp_factor)/(1.0-exp_factor);
  }
}

unsigned int Variational_Linear::get_index(unsigned int i, unsigned int j) const //mapping of a (dim_)x(dim_) symmetric matrix into a vector
{
  if (i<=j)
    return j+i*(dim_-1)-i*(i-1)/2;
  else
    return get_index(j,i);
}

}
}
