/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2011-2019 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

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
#include "core/ActionSet.h"
#include "tools/Grid.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/Exception.h"
#include "core/FlexibleBin.h"
#include "tools/Matrix.h"
#include "tools/Random.h"
#include <string>
#include <cstring>
#include "tools/File.h"
#include <iostream>
#include <limits>
#include <ctime>
#include <memory>

#define DP2CUTOFF 6.25

using namespace std;


namespace PLMD {
namespace bias {

//+PLUMEDOC BIAS EXTERNAL_PROTOCOL
/*
 * For lazyness all the MetaD keyworks have been kept, even though they are not actually used
 * Just keep the same input as the master simulation
*/
//+ENDPLUMEDOC

class ExternalProtocol : public Bias {

private:
  struct Gaussian {
    vector<double> center;
    vector<double> sigma;
    double height;
    bool   multivariate; // this is required to discriminate the one dimensional case
    vector<double> invsigma;
    Gaussian(const vector<double> & center,const vector<double> & sigma,double height, bool multivariate ):
      center(center),sigma(sigma),height(height),multivariate(multivariate),invsigma(sigma) {
      // to avoid troubles from zero element in flexible hills
      for(unsigned i=0; i<invsigma.size(); ++i) abs(invsigma[i])>1.e-20?invsigma[i]=1.0/invsigma[i]:0.;
    }
  };
  struct TemperingSpecs {
    bool is_active;
    std::string name_stem;
    std::string name;
    double biasf;
    double threshold;
    double alpha;
    inline TemperingSpecs(bool is_active, const std::string &name_stem, const std::string &name, double biasf, double threshold, double alpha) :
      is_active(is_active), name_stem(name_stem), name(name), biasf(biasf), threshold(threshold), alpha(alpha)
    {}
  };
  vector<double> sigma0_;
  vector<double> sigma0min_;
  vector<double> sigma0max_;
  vector<Gaussian> hills_;
  vector<Gaussian> current_hills_;
  OFile hillsOfile_;
  OFile gridfile_;
  std::unique_ptr<GridBase> BiasGrid_;
  bool storeOldGrids_;
  int wgridstride_;
  bool grid_;
  double height0_;
  double biasf_;
  static const size_t n_tempering_options_ = 1;
  static const string tempering_names_[1][2];
  double dampfactor_;
  struct TemperingSpecs tt_specs_;
  std::string targetfilename_;
  std::unique_ptr<GridBase> TargetGrid_;
  double kbt_;
  int stride_;
  bool welltemp_;
  std::unique_ptr<double[]> dp_;
  int adaptive_;
  std::unique_ptr<FlexibleBin> flexbin;
  int mw_n_;
  string mw_dir_;
  int mw_id_;
  int mw_rstride_;
  bool walkers_mpi;
  unsigned mpi_nw_;
  unsigned mpi_mw_;
  bool acceleration;
  double acc;
  double acc_restart_mean_;
  bool calc_max_bias_;
  double max_bias_;
  bool calc_transition_bias_;
  double transition_bias_;
  vector<vector<double> > transitionwells_;
  vector<std::unique_ptr<IFile>> ifiles;
//-----------------------------------------------------------
  std::unique_ptr<IFile> iprotocol;
  unsigned nhills_;
  bool inverted_protocol_;
//-----------------------------------------------------------
  vector<string> ifilesnames;
  double uppI_;
  double lowI_;
  bool doInt_;
  bool isFirstStep;
  bool calc_rct_;
  double reweight_factor_;
  unsigned rct_ustride_;
  double work_;
  double inst_work_;
  long int last_step_warn_grid;

  static void   registerTemperingKeywords(const std::string &name_stem, const std::string &name, Keywords &keys);
  void   readTemperingSpecs(TemperingSpecs &t_specs);
  void   logTemperingSpecs(const TemperingSpecs &t_specs);
  void   readGaussians(IFile*);
  void   writeGaussian(const Gaussian&,OFile&);
  void   addGaussian(const Gaussian&);
  double getBiasAndDerivatives(const vector<double>&,double* der=NULL);
  double evaluateGaussian(const vector<double>&, const Gaussian&,double* der=NULL);
  vector<unsigned> getGaussianSupport(const Gaussian&);
  bool   scanOneHill(IFile *ifile,  vector<Value> &v, vector<double> &center, vector<double>  &sigma, double &height, bool &multivariate);
  void   computeReweightingFactor();
  double getTransitionBarrierBias();
  string fmt;

public:
  explicit ExternalProtocol(const ActionOptions&);
  void calculate();
  void update();
  static void registerKeywords(Keywords& keys);
  bool checkNeedsGradients()const {if(adaptive_==FlexibleBin::geometry) {return true;} else {return false;}}
};

PLUMED_REGISTER_ACTION(ExternalProtocol,"EXTERNAL_PROTOCOL")

void ExternalProtocol::registerKeywords(Keywords& keys) {
  Bias::registerKeywords(keys);
  keys.addOutputComponent("rbias","CALC_RCT","the instantaneous value of the bias normalized using the \\f$c(t)\\f$ reweighting factor [rbias=bias-rct]."
                          "This component can be used to obtain a reweighted histogram.");
  keys.addOutputComponent("rct","CALC_RCT","the reweighting factor \\f$c(t)\\f$.");
  keys.addOutputComponent("work","default","accumulator for work");
  keys.addOutputComponent("instwork","default","work done in a pace step");
  keys.addOutputComponent("acc","ACCELERATION","the metadynamics acceleration factor");
  keys.addOutputComponent("maxbias", "CALC_MAX_BIAS", "the maximum of the metadynamics V(s, t)");
  keys.addOutputComponent("transbias", "CALC_TRANSITION_BIAS", "the metadynamics transition bias V*(t)");
  keys.use("ARG");
  keys.add("compulsory","SIGMA","the widths of the Gaussian hills");
  keys.add("compulsory","PACE","the frequency for hill addition");
  keys.add("compulsory","FILE","HILLS","a file in which the list of added hills is stored");
//--------------------------------------------------------------------------------------------------
  keys.add("compulsory","PROTOCOL","Protocol.data","a Hills file of a previous MetaD simulation, to be used as protocol");
  keys.add("optional","NHILLS","how many hills to read from protocol per each step");
  keys.addFlag("INVERTED_PROTOCOL", false, "run as inverted protocol, first load it all and then delete it one gaussian at the time. Inverted hills file should be provided");
//--------------------------------------------------------------------------------------------------
  keys.add("optional","HEIGHT","the heights of the Gaussian hills. Compulsory unless TAU and either BIASFACTOR or DAMPFACTOR are given");
  keys.add("optional","FMT","specify format for HILLS files (useful for decrease the number of digits in regtests)");
  keys.add("optional","BIASFACTOR","use well tempered metadynamics and use this bias factor.  Please note you must also specify temp");
  keys.add("optional","RECT","list of bias factors for all the replicas");
  keys.add("optional","DAMPFACTOR","damp hills with exp(-max(V)/(\\f$k_B\\f$T*DAMPFACTOR)");
  for (size_t i = 0; i < n_tempering_options_; i++) {
    registerTemperingKeywords(tempering_names_[i][0], tempering_names_[i][1], keys);
  }
  keys.add("optional","TARGET","target to a predefined distribution");
  keys.add("optional","TEMP","the system temperature - this is only needed if you are doing well-tempered metadynamics");
  keys.add("optional","TAU","in well tempered metadynamics, sets height to (\\f$k_B \\Delta T\\f$*pace*timestep)/tau");
  keys.add("optional","GRID_MIN","the lower bounds for the grid");
  keys.add("optional","GRID_MAX","the upper bounds for the grid");
  keys.add("optional","GRID_BIN","the number of bins for the grid");
  keys.add("optional","GRID_SPACING","the approximate grid spacing (to be used as an alternative or together with GRID_BIN)");
  keys.addFlag("CALC_RCT",false,"calculate the \\f$c(t)\\f$ reweighting factor and use that to obtain the normalized bias [rbias=bias-rct]."
               "This method is not compatible with metadynamics not on a grid.");
  keys.add("optional","RCT_USTRIDE","the update stride for calculating the \\f$c(t)\\f$ reweighting factor."
           "The default 1, so \\f$c(t)\\f$ is updated every time the bias is updated.");
  keys.addFlag("GRID_SPARSE",false,"use a sparse grid to store hills");
  keys.addFlag("GRID_NOSPLINE",false,"don't use spline interpolation with grids");
  keys.add("optional","GRID_WSTRIDE","write the grid to a file every N steps");
  keys.add("optional","GRID_WFILE","the file on which to write the grid");
  keys.add("optional","GRID_RFILE","a grid file from which the bias should be read at the initial step of the simulation");
  keys.addFlag("STORE_GRIDS",false,"store all the grid files the calculation generates. They will be deleted if this keyword is not present");
  keys.add("optional","ADAPTIVE","use a geometric (=GEOM) or diffusion (=DIFF) based hills width scheme. Sigma is one number that has distance units or time step dimensions");
  keys.add("optional","WALKERS_ID", "walker id");
  keys.add("optional","WALKERS_N", "number of walkers");
  keys.add("optional","WALKERS_DIR", "shared directory with the hills files from all the walkers");
  keys.add("optional","WALKERS_RSTRIDE","stride for reading hills files");
  keys.add("optional","INTERVAL","one dimensional lower and upper limits, outside the limits the system will not feel the biasing force.");
  keys.add("optional","SIGMA_MAX","the upper bounds for the sigmas (in CV units) when using adaptive hills. Negative number means no bounds ");
  keys.add("optional","SIGMA_MIN","the lower bounds for the sigmas (in CV units) when using adaptive hills. Negative number means no bounds ");
  keys.addFlag("WALKERS_MPI",false,"Switch on MPI version of multiple walkers - not compatible with WALKERS_* options other than WALKERS_DIR");
  keys.addFlag("ACCELERATION",false,"Set to TRUE if you want to compute the metadynamics acceleration factor.");
  keys.add("optional","ACCELERATION_RFILE","a data file from which the acceleration should be read at the initial step of the simulation");
  keys.addFlag("CALC_MAX_BIAS", false, "Set to TRUE if you want to compute the maximum of the metadynamics V(s, t)");
  keys.addFlag("CALC_TRANSITION_BIAS", false, "Set to TRUE if you want to compute a metadynamics transition bias V*(t)");
  keys.add("numbered", "TRANSITIONWELL", "This keyword appears multiple times as TRANSITIONWELL followed by an integer. Each specifies the coordinates for one well as in transition-tempered metadynamics. At least one must be provided.");
  keys.use("RESTART");
  keys.use("UPDATE_FROM");
  keys.use("UPDATE_UNTIL");
}

const std::string ExternalProtocol::tempering_names_[1][2] = {{"TT", "transition tempered"}};

void ExternalProtocol::registerTemperingKeywords(const std::string &name_stem, const std::string &name, Keywords &keys) {
  keys.add("optional", name_stem + "BIASFACTOR", "use " + name + " metadynamics with this biasfactor.  Please note you must also specify temp");
  keys.add("optional", name_stem + "BIASTHRESHOLD", "use " + name + " metadynamics with this bias threshold.  Please note you must also specify " + name_stem + "BIASFACTOR");
  keys.add("optional", name_stem + "ALPHA", "use " + name + " metadynamics with this hill size decay exponent parameter.  Please note you must also specify " + name_stem + "BIASFACTOR");
}

ExternalProtocol::ExternalProtocol(const ActionOptions& ao):
  PLUMED_BIAS_INIT(ao),
// Grid stuff initialization
  wgridstride_(0), grid_(false),
// Metadynamics basic parameters
  height0_(std::numeric_limits<double>::max()), biasf_(-1.0), dampfactor_(0.0),
  tt_specs_(false, "TT", "Transition Tempered", -1.0, 0.0, 1.0),
  kbt_(0.0),
  stride_(0), welltemp_(false),
// Other stuff
  adaptive_(FlexibleBin::none),
// Multiple walkers initialization
  mw_n_(1), mw_dir_(""), mw_id_(0), mw_rstride_(1),
  walkers_mpi(false), mpi_nw_(0), mpi_mw_(0),
  acceleration(false), acc(0.0), acc_restart_mean_(0.0),
  calc_max_bias_(false), max_bias_(0.0),
  calc_transition_bias_(false), transition_bias_(0.0),
// Interval initialization
  uppI_(-1), lowI_(-1), doInt_(false),
  isFirstStep(true),
  calc_rct_(false),
  reweight_factor_(0.0),
  rct_ustride_(1),
  work_(0),
  inst_work_(0),
  last_step_warn_grid(0)
{
  // parse the flexible hills
  string adaptiveoption;
  adaptiveoption="NONE";
  parse("ADAPTIVE",adaptiveoption);
  if(adaptiveoption=="GEOM") {
    log.printf("  Uses Geometry-based hills width: sigma must be in distance units and only one sigma is needed\n");
    adaptive_=FlexibleBin::geometry;
  } else if(adaptiveoption=="DIFF") {
    log.printf("  Uses Diffusion-based hills width: sigma must be in time steps and only one sigma is needed\n");
    adaptive_=FlexibleBin::diffusion;
  } else if(adaptiveoption=="NONE") {
    adaptive_=FlexibleBin::none;
  } else {
    error("I do not know this type of adaptive scheme");
  }

  parse("FMT",fmt);

  // parse the sigma
  parseVector("SIGMA",sigma0_);
  if(adaptive_==FlexibleBin::none) {
    // if you use normal sigma you need one sigma per argument
    if( sigma0_.size()!=getNumberOfArguments() ) error("number of arguments does not match number of SIGMA parameters");
  } else {
    // if you use flexible hills you need one sigma
    if(sigma0_.size()!=1) {
      error("If you choose ADAPTIVE you need only one sigma according to your choice of type (GEOM/DIFF)");
    }
    // if adaptive then the number must be an integer
    if(adaptive_==FlexibleBin::diffusion) {
      if(int(sigma0_[0])-sigma0_[0]>1.e-9 || int(sigma0_[0])-sigma0_[0] <-1.e-9 || int(sigma0_[0])<1 ) {
        error("In case of adaptive hills with diffusion, the sigma must be an integer which is the number of time steps\n");
      }
    }
    // here evtl parse the sigma min and max values
    parseVector("SIGMA_MIN",sigma0min_);
    if(sigma0min_.size()>0 && sigma0min_.size()!=getNumberOfArguments()) {
      error("the number of SIGMA_MIN values be the same of the number of the arguments");
    } else if(sigma0min_.size()==0) {
      sigma0min_.resize(getNumberOfArguments());
      for(unsigned i=0; i<getNumberOfArguments(); i++) {sigma0min_[i]=-1.;}
    }

    parseVector("SIGMA_MAX",sigma0max_);
    if(sigma0max_.size()>0 && sigma0max_.size()!=getNumberOfArguments()) {
      error("the number of SIGMA_MAX values be the same of the number of the arguments");
    } else if(sigma0max_.size()==0) {
      sigma0max_.resize(getNumberOfArguments());
      for(unsigned i=0; i<getNumberOfArguments(); i++) {sigma0max_[i]=-1.;}
    }

    flexbin.reset(new FlexibleBin(adaptive_,this,sigma0_[0],sigma0min_,sigma0max_));
  }
  // note: HEIGHT is not compulsory, since one could use the TAU keyword, see below
  parse("HEIGHT",height0_);
  parse("PACE",stride_);
  if(stride_<=0 ) error("frequency for hill addition is nonsensical");
  string hillsfname="HILLS";
  parse("FILE",hillsfname);
//--------------------------------------------------------------------------------------------------
  string protocolfname="Protocol.data";
  parse("PROTOCOL",protocolfname);
  nhills_=1;
  parse("NHILLS",nhills_);
  inverted_protocol_=false;
  parseFlag("INVERTED_PROTOCOL",inverted_protocol_);
//--------------------------------------------------------------------------------------------------

  // Manually set to calculate special bias quantities
  // throughout the course of simulation. (These are chosen due to
  // relevance for tempering and event-driven logic as well.)
  parseFlag("CALC_MAX_BIAS", calc_max_bias_);
  parseFlag("CALC_TRANSITION_BIAS", calc_transition_bias_);

  std::vector<double> rect_biasf_;
  parseVector("RECT",rect_biasf_);
  if(rect_biasf_.size()>0) {
    int r=0;
    if(comm.Get_rank()==0) r=multi_sim_comm.Get_rank();
    comm.Bcast(r,0);
    biasf_=rect_biasf_[r];
    log<<"  You are using RECT\n";
  } else {
    parse("BIASFACTOR",biasf_);
  }
  if( biasf_<1.0  && biasf_!=-1.0) error("well tempered bias factor is nonsensical");
  parse("DAMPFACTOR",dampfactor_);
  double temp=0.0;
  parse("TEMP",temp);
  if(temp>0.0) kbt_=plumed.getAtoms().getKBoltzmann()*temp;
  else kbt_=plumed.getAtoms().getKbT();
  if(biasf_>=1.0) {
    if(kbt_==0.0) error("Unless the MD engine passes the temperature to plumed, with well-tempered metad you must specify it using TEMP");
    welltemp_=true;
  }
  if(dampfactor_>0.0) {
    if(kbt_==0.0) error("Unless the MD engine passes the temperature to plumed, with damped metad you must specify it using TEMP");
  }

  // Set transition tempering parameters.
  // Transition wells are read later via calc_transition_bias_.
  readTemperingSpecs(tt_specs_);
  if (tt_specs_.is_active) calc_transition_bias_ = true;

  // If any previous option specified to calculate a transition bias,
  // now read the transition wells for that quantity.
  if (calc_transition_bias_) {
    vector<double> tempcoords(getNumberOfArguments());
    for (unsigned i = 0; ; i++) {
      if (!parseNumberedVector("TRANSITIONWELL", i, tempcoords) ) break;
      if (tempcoords.size() != getNumberOfArguments()) {
        error("incorrect number of coordinates for transition tempering well");
      }
      transitionwells_.push_back(tempcoords);
    }
  }

  parse("TARGET",targetfilename_);
  if(targetfilename_.length()>0 && kbt_==0.0)  error("with TARGET temperature must be specified");
  double tau=0.0;
  parse("TAU",tau);
  if(tau==0.0) {
    if(height0_==std::numeric_limits<double>::max()) error("At least one between HEIGHT and TAU should be specified");
    // if tau is not set, we compute it here from the other input parameters
    if(welltemp_) tau=(kbt_*(biasf_-1.0))/height0_*getTimeStep()*stride_;
    else if(dampfactor_>0.0) tau=(kbt_*dampfactor_)/height0_*getTimeStep()*stride_;
  } else {
    if(height0_!=std::numeric_limits<double>::max()) error("At most one between HEIGHT and TAU should be specified");
    if(welltemp_) {
      if(biasf_!=1.0) height0_=(kbt_*(biasf_-1.0))/tau*getTimeStep()*stride_;
      else           height0_=kbt_/tau*getTimeStep()*stride_; // special case for gamma=1
    }
    else if(dampfactor_>0.0) height0_=(kbt_*dampfactor_)/tau*getTimeStep()*stride_;
    else error("TAU only makes sense in well-tempered or damped metadynamics");
  }

  // Grid Stuff
  vector<std::string> gmin(getNumberOfArguments());
  parseVector("GRID_MIN",gmin);
  if(gmin.size()!=getNumberOfArguments() && gmin.size()!=0) error("not enough values for GRID_MIN");
  vector<std::string> gmax(getNumberOfArguments());
  parseVector("GRID_MAX",gmax);
  if(gmax.size()!=getNumberOfArguments() && gmax.size()!=0) error("not enough values for GRID_MAX");
  vector<unsigned> gbin(getNumberOfArguments());
  vector<double>   gspacing;
  parseVector("GRID_BIN",gbin);
  if(gbin.size()!=getNumberOfArguments() && gbin.size()!=0) error("not enough values for GRID_BIN");
  parseVector("GRID_SPACING",gspacing);
  if(gspacing.size()!=getNumberOfArguments() && gspacing.size()!=0) error("not enough values for GRID_SPACING");
  if(gmin.size()!=gmax.size()) error("GRID_MAX and GRID_MIN should be either present or absent");
  if(gspacing.size()!=0 && gmin.size()==0) error("If GRID_SPACING is present also GRID_MIN should be present");
  if(gbin.size()!=0     && gmin.size()==0) error("If GRID_SPACING is present also GRID_MIN should be present");
  if(gmin.size()!=0) {
    if(gbin.size()==0 && gspacing.size()==0) {
      if(adaptive_==FlexibleBin::none) {
        log<<"  Binsize not specified, 1/5 of sigma will be be used\n";
        plumed_assert(sigma0_.size()==getNumberOfArguments());
        gspacing.resize(getNumberOfArguments());
        for(unsigned i=0; i<gspacing.size(); i++) gspacing[i]=0.2*sigma0_[i];
      } else {
        // with adaptive hills and grid a sigma min must be specified
        for(unsigned i=0; i<sigma0min_.size(); i++) if(sigma0min_[i]<=0) error("When using Adaptive Gaussians on a grid SIGMA_MIN must be specified");
        log<<"  Binsize not specified, 1/5 of sigma_min will be be used\n";
        gspacing.resize(getNumberOfArguments());
        for(unsigned i=0; i<gspacing.size(); i++) gspacing[i]=0.2*sigma0min_[i];
      }
    } else if(gspacing.size()!=0 && gbin.size()==0) {
      log<<"  The number of bins will be estimated from GRID_SPACING\n";
    } else if(gspacing.size()!=0 && gbin.size()!=0) {
      log<<"  You specified both GRID_BIN and GRID_SPACING\n";
      log<<"  The more conservative (highest) number of bins will be used for each variable\n";
    }
    if(gbin.size()==0) gbin.assign(getNumberOfArguments(),1);
    if(gspacing.size()!=0) for(unsigned i=0; i<getNumberOfArguments(); i++) {
        double a,b;
        Tools::convert(gmin[i],a);
        Tools::convert(gmax[i],b);
        unsigned n=((b-a)/gspacing[i])+1;
        if(gbin[i]<n) gbin[i]=n;
      }
  }
  bool sparsegrid=false;
  parseFlag("GRID_SPARSE",sparsegrid);
  bool nospline=false;
  parseFlag("GRID_NOSPLINE",nospline);
  bool spline=!nospline;
  if(gbin.size()>0) {grid_=true;}
  parse("GRID_WSTRIDE",wgridstride_);
  string gridfilename_;
  parse("GRID_WFILE",gridfilename_);
  parseFlag("STORE_GRIDS",storeOldGrids_);
  if(grid_ && gridfilename_.length()>0) {
    if(wgridstride_==0 ) error("frequency with which to output grid not specified use GRID_WSTRIDE");
  }

  if(grid_ && wgridstride_>0) {
    if(gridfilename_.length()==0) error("grid filename not specified use GRID_WFILE");
  }
  string gridreadfilename_;
  parse("GRID_RFILE",gridreadfilename_);

  if(!grid_&&gridfilename_.length()> 0) error("To write a grid you need first to define it!");
  if(!grid_&&gridreadfilename_.length()>0) error("To read a grid you need first to define it!");

  // Reweighting factor rct
  parseFlag("CALC_RCT",calc_rct_);
  if (calc_rct_)
    plumed_massert(grid_,"CALC_RCT is supported only if bias is on a grid");
  parse("RCT_USTRIDE",rct_ustride_);

  if(dampfactor_>0.0) {
    if(!grid_) error("With DAMPFACTOR you should use grids");
  }

  // Multiple walkers
  parse("WALKERS_N",mw_n_);
  parse("WALKERS_ID",mw_id_);
  if(mw_n_<=mw_id_) error("walker ID should be a numerical value less than the total number of walkers");
  parse("WALKERS_DIR",mw_dir_);
  parse("WALKERS_RSTRIDE",mw_rstride_);

  // MPI version
  parseFlag("WALKERS_MPI",walkers_mpi);

  // Inteval keyword
  vector<double> tmpI(2);
  parseVector("INTERVAL",tmpI);
  if(tmpI.size()!=2&&tmpI.size()!=0) error("both a lower and an upper limits must be provided with INTERVAL");
  else if(tmpI.size()==2) {
    lowI_=tmpI.at(0);
    uppI_=tmpI.at(1);
    if(getNumberOfArguments()!=1) error("INTERVAL limits correction works only for monodimensional metadynamics!");
    if(uppI_<lowI_) error("The Upper limit must be greater than the Lower limit!");
    if(getPntrToArgument(0)->isPeriodic()) error("INTERVAL cannot be used with periodic variables!");
    doInt_=true;
  }

  acceleration=false;
  parseFlag("ACCELERATION",acceleration);
  // Check for a restart acceleration if acceleration is active.
  string acc_rfilename;
  if (acceleration) {
    parse("ACCELERATION_RFILE", acc_rfilename);
  }

  checkRead();

//--------------------------------------------------------------------------------------------------
  log.printf("  Protocol taken from file %s\n",protocolfname.c_str());
  log.printf("  Reading %d hills at the time\n",nhills_);
  if (inverted_protocol_)
    log.printf(" -- INVERTED_PROTOCOL: protocol will be first loaded and then deleted one gaussian at the time\n");
//--------------------------------------------------------------------------------------------------
  log.printf("  Gaussian width ");
  if (adaptive_==FlexibleBin::diffusion)log.printf(" (Note: The units of sigma are in timesteps) ");
  if (adaptive_==FlexibleBin::geometry)log.printf(" (Note: The units of sigma are in dist units) ");
  for(unsigned i=0; i<sigma0_.size(); ++i) log.printf(" %f",sigma0_[i]);
  log.printf("  Gaussian height %f\n",height0_);
  log.printf("  Gaussian deposition pace %d\n",stride_);
  log.printf("  Gaussian file %s\n",hillsfname.c_str());
  if(welltemp_) {
    log.printf("  Well-Tempered Bias Factor %f\n",biasf_);
    log.printf("  Hills relaxation time (tau) %f\n",tau);
    log.printf("  KbT %f\n",kbt_);
  }
  // Transition tempered metadynamics options
  if (tt_specs_.is_active) {
    logTemperingSpecs(tt_specs_);
    // Check that the appropriate transition bias quantity is calculated.
    // (Should never trip, given that the flag is automatically set.)
    if (!calc_transition_bias_) {
      error(" transition tempering requires calculation of a transition bias");
    }
  }

  // Overall tempering sanity check (this gets tricky when multiple are active).
  // When multiple temperings are active, it's fine to have one tempering attempt
  // to increase hill size with increasing bias, so long as the others can shrink
  // the hills faster than it increases their size in the long-time limit.
  // This set of checks ensures that the hill sizes eventually decay to zero as c(t)
  // diverges to infinity.
  // The alpha parameter allows hills to decay as 1/t^alpha instead of 1/t,
  // a slower decay, so as t -> infinity, only the temperings with the largest
  // alphas govern the final asymptotic decay. (Alpha helps prevent false convergence.)
  if (welltemp_ || dampfactor_ > 0.0 || tt_specs_.is_active) {
    // Determine the number of active temperings.
    int n_active = 0;
    if (welltemp_) n_active++;
    if (dampfactor_ > 0.0) n_active++;
    if (tt_specs_.is_active) n_active++;
    // Find the greatest alpha.
    double greatest_alpha = 0.0;
    if (welltemp_) greatest_alpha = max(greatest_alpha, 1.0);
    if (dampfactor_ > 0.0) greatest_alpha = max(greatest_alpha, 1.0);
    if (tt_specs_.is_active) greatest_alpha = max(greatest_alpha, tt_specs_.alpha);
    // Find the least alpha.
    double least_alpha = 1.0;
    if (welltemp_) least_alpha = min(least_alpha, 1.0);
    if (dampfactor_ > 0.0) least_alpha = min(least_alpha, 1.0);
    if (tt_specs_.is_active) least_alpha = min(least_alpha, tt_specs_.alpha);
    // Find the inverse harmonic average of the delta T parameters for all
    // of the temperings with the greatest alpha values.
    double total_governing_deltaT_inv = 0.0;
    if (welltemp_ && 1.0 == greatest_alpha && biasf_ != 1.0) total_governing_deltaT_inv += 1.0 / (biasf_ - 1.0);
    if (dampfactor_ > 0.0 && 1.0 == greatest_alpha) total_governing_deltaT_inv += 1.0 / (dampfactor_);
    if (tt_specs_.is_active && tt_specs_.alpha == greatest_alpha) total_governing_deltaT_inv += 1.0 / (tt_specs_.biasf - 1.0);
    // Give a newbie-friendly error message for people using one tempering if
    // only one is active.
    if (n_active == 1 && total_governing_deltaT_inv < 0.0) {
      error("for stable tempering, the bias factor must be greater than one");
      // Give a slightly more complex error message to users stacking multiple
      // tempering options at a time, but all with uniform alpha values.
    } else if (total_governing_deltaT_inv < 0.0 && greatest_alpha == least_alpha) {
      error("for stable tempering, the sum of the inverse Delta T parameters must be greater than zero!");
      // Give the most technical error message to users stacking multiple tempering
      // options with different alpha parameters.
    } else if (total_governing_deltaT_inv < 0.0 && greatest_alpha != least_alpha) {
      error("for stable tempering, the sum of the inverse Delta T parameters for the greatest asymptotic hill decay exponents must be greater than zero!");
    }
  }

  if(doInt_) log.printf("  Upper and Lower limits boundaries for the bias are activated at %f - %f\n", lowI_, uppI_);
  if(grid_) {
    log.printf("  Grid min");
    for(unsigned i=0; i<gmin.size(); ++i) log.printf(" %s",gmin[i].c_str() );
    log.printf("\n");
    log.printf("  Grid max");
    for(unsigned i=0; i<gmax.size(); ++i) log.printf(" %s",gmax[i].c_str() );
    log.printf("\n");
    log.printf("  Grid bin");
    for(unsigned i=0; i<gbin.size(); ++i) log.printf(" %u",gbin[i]);
    log.printf("\n");
    if(spline) {log.printf("  Grid uses spline interpolation\n");}
    if(sparsegrid) {log.printf("  Grid uses sparse grid\n");}
    if(wgridstride_>0) {log.printf("  Grid is written on file %s with stride %d\n",gridfilename_.c_str(),wgridstride_);}
  }

  if(mw_n_>1) {
    if(walkers_mpi) error("MPI version of multiple walkers is not compatible with filesystem version of multiple walkers");
    log.printf("  %d multiple walkers active\n",mw_n_);
    log.printf("  walker id %d\n",mw_id_);
    log.printf("  reading stride %d\n",mw_rstride_);
    if(mw_dir_!="")log.printf("  directory with hills files %s\n",mw_dir_.c_str());
  } else {
    if(walkers_mpi) {
      log.printf("  Multiple walkers active using MPI communnication\n");
      if(mw_dir_!="")log.printf("  directory with hills files %s\n",mw_dir_.c_str());
      if(comm.Get_rank()==0) {
        // Only root of group can communicate with other walkers
        mpi_nw_=multi_sim_comm.Get_size();
        mpi_mw_=multi_sim_comm.Get_rank();
      }
      // Communicate to the other members of the same group
      // info abount number of walkers and walker index
      comm.Bcast(mpi_nw_,0);
      comm.Bcast(mpi_mw_,0);
    }
  }

  if(calc_rct_) {
    addComponent("rbias"); componentIsNotPeriodic("rbias");
    addComponent("rct"); componentIsNotPeriodic("rct");
    log.printf("  The c(t) reweighting factor will be calculated every %u hills\n",rct_ustride_);
    getPntrToComponent("rct")->set(reweight_factor_);
  }
  addComponent("work"); componentIsNotPeriodic("work");
  addComponent("instwork"); componentIsNotPeriodic("instwork");

  if(acceleration) {
    if (kbt_ == 0.0) {
      error("The calculation of the acceleration works only if simulation temperature has been defined");
    }
    log.printf("  calculation on the fly of the acceleration factor\n");
    addComponent("acc"); componentIsNotPeriodic("acc");
    // Set the initial value of the the acceleration.
    // If this is not a restart, set to 1.0.
    if (acc_rfilename.length() == 0) {
      getPntrToComponent("acc")->set(1.0);
      if(getRestart()) {
        log.printf("  WARNING: calculating the acceleration factor in a restarted run without reading in the previous value will most likely lead to incorrect results. You should use the ACCELERATION_RFILE keyword.\n");
      }
      // Otherwise, read and set the restart value.
    } else {
      // Restart of acceleration does not make sense if the restart timestep is zero.
      //if (getStep() == 0) {
      //  error("Restarting calculation of acceleration factors works only if simulation timestep is restarted correctly");
      //}
      // Open the ACCELERATION_RFILE.
      IFile acc_rfile;
      acc_rfile.link(*this);
      if(acc_rfile.FileExist(acc_rfilename)) {
        acc_rfile.open(acc_rfilename);
      } else {
        error("The ACCELERATION_RFILE file you want to read: " + acc_rfilename + ", cannot be found!");
      }
      // Read the file to find the restart acceleration.
      double acc_rmean;
      double acc_rtime;
      std::string acclabel = getLabel() + ".acc";
      acc_rfile.allowIgnoredFields();
      while(acc_rfile.scanField("time", acc_rtime)) {
        acc_rfile.scanField(acclabel, acc_rmean);
        acc_rfile.scanField();
      }
      acc_restart_mean_ = acc_rmean;
      // Set component based on the read values.
      getPntrToComponent("acc")->set(acc_rmean);
      log.printf("  initial acceleration factor read from file %s: value of %f at time %f\n",acc_rfilename.c_str(),acc_rmean,acc_rtime);
    }
  }
  if (calc_max_bias_) {
    if (!grid_) error("Calculating the maximum bias on the fly works only with a grid");
    log.printf("  calculation on the fly of the maximum bias max(V(s,t)) \n");
    addComponent("maxbias");
    componentIsNotPeriodic("maxbias");
  }
  if (calc_transition_bias_) {
    if (!grid_) error("Calculating the transition bias on the fly works only with a grid");
    log.printf("  calculation on the fly of the transition bias V*(t)\n");
    addComponent("transbias");
    componentIsNotPeriodic("transbias");
    log.printf("  Number of transition wells %d\n", transitionwells_.size());
    if (transitionwells_.size() == 0) error("Calculating the transition bias on the fly requires definition of at least one transition well");
    // Check that a grid is in use.
    if (!grid_) error(" transition barrier finding requires a grid for the bias");
    // Log the wells and check that they are in the grid.
    for (unsigned i = 0; i < transitionwells_.size(); i++) {
      // Log the coordinate.
      log.printf("  Transition well %d at coordinate ", i);
      for (unsigned j = 0; j < getNumberOfArguments(); j++) log.printf("%f ", transitionwells_[i][j]);
      log.printf("\n");
      // Check that the coordinate is in the grid.
      for (unsigned j = 0; j < getNumberOfArguments(); j++) {
        double max, min;
        Tools::convert(gmin[j], min);
        Tools::convert(gmax[j], max);
        if (transitionwells_[i][j] < min || transitionwells_[i][j] > max) error(" transition well is not in grid");
      }
    }
  }

  // for performance
  dp_.reset( new double[getNumberOfArguments()] );

  // initializing and checking grid
  if(grid_) {
    // check for mesh and sigma size
    for(unsigned i=0; i<getNumberOfArguments(); i++) {
      double a,b;
      Tools::convert(gmin[i],a);
      Tools::convert(gmax[i],b);
      double mesh=(b-a)/((double)gbin[i]);
      if(adaptive_==FlexibleBin::none) {
        if(mesh>0.5*sigma0_[i]) log<<"  WARNING: Using a METAD with a Grid Spacing larger than half of the Gaussians width can produce artifacts\n";
      } else {
        if(mesh>0.5*sigma0min_[i]||sigma0min_[i]<0.) log<<"  WARNING: to use a METAD with a GRID and ADAPTIVE you need to set a Grid Spacing larger than half of the Gaussians \n";
      }
    }
    std::string funcl=getLabel() + ".bias";
    if(!sparsegrid) {BiasGrid_.reset(new Grid(funcl,getArguments(),gmin,gmax,gbin,spline,true));}
    else {BiasGrid_.reset(new SparseGrid(funcl,getArguments(),gmin,gmax,gbin,spline,true));}
    std::vector<std::string> actualmin=BiasGrid_->getMin();
    std::vector<std::string> actualmax=BiasGrid_->getMax();
    for(unsigned i=0; i<getNumberOfArguments(); i++) {
      std::string is;
      Tools::convert(i,is);
      if(gmin[i]!=actualmin[i]) error("GRID_MIN["+is+"] must be adjusted to "+actualmin[i]+" to fit periodicity");
      if(gmax[i]!=actualmax[i]) error("GRID_MAX["+is+"] must be adjusted to "+actualmax[i]+" to fit periodicity");
    }
  }

  // restart from external grid
  bool restartedFromGrid=false;
  if(gridreadfilename_.length()>0) {
    // read the grid in input, find the keys
    IFile gridfile;
    gridfile.link(*this);
    if(gridfile.FileExist(gridreadfilename_)) {
      gridfile.open(gridreadfilename_);
    } else {
      error("The GRID file you want to read: " + gridreadfilename_ + ", cannot be found!");
    }
    std::string funcl=getLabel() + ".bias";
    BiasGrid_=GridBase::create(funcl, getArguments(), gridfile, gmin, gmax, gbin, sparsegrid, spline, true);
    if(BiasGrid_->getDimension()!=getNumberOfArguments()) error("mismatch between dimensionality of input grid and number of arguments");
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      if( getPntrToArgument(i)->isPeriodic()!=BiasGrid_->getIsPeriodic()[i] ) error("periodicity mismatch between arguments and input bias");
      double a, b;
      Tools::convert(gmin[i],a);
      Tools::convert(gmax[i],b);
      double mesh=(b-a)/((double)gbin[i]);
      if(mesh>0.5*sigma0_[i]) log<<"  WARNING: Using a METAD with a Grid Spacing larger than half of the Gaussians width can produce artifacts\n";
    }
    log.printf("  Restarting from %s:",gridreadfilename_.c_str());
    if(getRestart()) restartedFromGrid=true;
  }

  // initializing and checking grid
  if(grid_&&!(gridreadfilename_.length()>0)) {
    // check for adaptive and sigma_min
    if(sigma0min_.size()==0&&adaptive_!=FlexibleBin::none) error("When using Adaptive Gaussians on a grid SIGMA_MIN must be specified");
    // check for mesh and sigma size
    for(unsigned i=0; i<getNumberOfArguments(); i++) {
      double a,b;
      Tools::convert(gmin[i],a);
      Tools::convert(gmax[i],b);
      double mesh=(b-a)/((double)gbin[i]);
      if(mesh>0.5*sigma0_[i]) log<<"  WARNING: Using a METAD with a Grid Spacing larger than half of the Gaussians width can produce artifacts\n";
    }
    std::string funcl=getLabel() + ".bias";
    if(!sparsegrid) {BiasGrid_.reset(new Grid(funcl,getArguments(),gmin,gmax,gbin,spline,true));}
    else {BiasGrid_.reset(new SparseGrid(funcl,getArguments(),gmin,gmax,gbin,spline,true));}
    std::vector<std::string> actualmin=BiasGrid_->getMin();
    std::vector<std::string> actualmax=BiasGrid_->getMax();
    for(unsigned i=0; i<getNumberOfArguments(); i++) {
      if(gmin[i]!=actualmin[i]) log<<"  WARNING: GRID_MIN["<<i<<"] has been adjusted to "<<actualmin[i]<<" to fit periodicity\n";
      if(gmax[i]!=actualmax[i]) log<<"  WARNING: GRID_MAX["<<i<<"] has been adjusted to "<<actualmax[i]<<" to fit periodicity\n";
    }
  }

  // creating vector of ifile* for hills reading
  // open all files at the beginning and read Gaussians if restarting
  for(int i=0; i<mw_n_; ++i) {
    string fname;
    if(mw_dir_!="") {
      if(mw_n_>1) {
        stringstream out; out << i;
        fname = mw_dir_+"/"+hillsfname+"."+out.str();
      } else if(walkers_mpi) {
        fname = mw_dir_+"/"+hillsfname;
      } else {
        fname = hillsfname;
      }
    } else {
      if(mw_n_>1) {
        stringstream out; out << i;
        fname = hillsfname+"."+out.str();
      } else {
        fname = hillsfname;
      }
    }
    ifiles.emplace_back(new IFile());
    // this is just a shortcut pointer to the last element:
    IFile *ifile = ifiles.back().get();
    ifilesnames.push_back(fname);
    ifile->link(*this);
    if(ifile->FileExist(fname)) {
      ifile->open(fname);
      if(getRestart()&&!restartedFromGrid) {
        log.printf("  Restarting from %s:",ifilesnames[i].c_str());
        readGaussians(ifiles[i].get());
      }
      ifiles[i]->reset(false);
      // close only the walker own hills file for later writing
      if(i==mw_id_) ifiles[i]->close();
    } else {
      // in case a file does not exist and we are restarting, complain that the file was not found
      if(getRestart()) log<<"  WARNING: restart file "<<fname<<" not found\n";
    }
  }
//-------------------------------------------------------------------
  if (inverted_protocol_)
  {
    IFile tmp_iprotocol;
    tmp_iprotocol.link(*this);
    if(tmp_iprotocol.FileExist(protocolfname)) {
      tmp_iprotocol.open(protocolfname);
      readGaussians(&tmp_iprotocol);
      tmp_iprotocol.reset(false);
      tmp_iprotocol.close();
      computeReweightingFactor();
    }
    else
      error(" Hills protocol file '"+protocolfname+"' not found");
  }

  if (comm.Get_rank()==0 && multi_sim_comm.Get_rank()==0)
  {
    iprotocol=std::unique_ptr<IFile>(new IFile());
    iprotocol->link(*this);
    if(iprotocol->FileExist(protocolfname)) {
      iprotocol->open(protocolfname);
      iprotocol->reset(false);
    }
    else
      error(" Hills protocol file '"+protocolfname+"' not found");
  }
//-------------------------------------------------------------------

  comm.Barrier();
  // this barrier is needed when using walkers_mpi
  // to be sure that all files have been read before
  // backing them up
  // it should not be used when walkers_mpi is false otherwise
  // it would introduce troubles when using replicas without METAD
  // (e.g. in bias exchange with a neutral replica)
  // see issue #168 on github
  if(comm.Get_rank()==0 && walkers_mpi) multi_sim_comm.Barrier();
  if(targetfilename_.length()>0) {
    IFile gridfile; gridfile.open(targetfilename_);
    std::string funcl=getLabel() + ".target";
    TargetGrid_=GridBase::create(funcl,getArguments(),gridfile,false,false,true);
    if(TargetGrid_->getDimension()!=getNumberOfArguments()) error("mismatch between dimensionality of input grid and number of arguments");
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {
      if( getPntrToArgument(i)->isPeriodic()!=TargetGrid_->getIsPeriodic()[i] ) error("periodicity mismatch between arguments and input bias");
    }
  }

  // Calculate the Tiwary-Parrinello reweighting factor if we are restarting from previous hills
  if(getRestart() && calc_rct_) computeReweightingFactor();
  // Calculate all special bias quantities desired if restarting with nonzero bias.
  if(getRestart() && calc_max_bias_) {
    max_bias_ = BiasGrid_->getMaxValue();
    getPntrToComponent("maxbias")->set(max_bias_);
  }
  if(getRestart() && calc_transition_bias_) {
    transition_bias_ = getTransitionBarrierBias();
    getPntrToComponent("transbias")->set(transition_bias_);
  }

  // open grid file for writing
  if(wgridstride_>0) {
    gridfile_.link(*this);
    if(walkers_mpi) {
      int r=0;
      if(comm.Get_rank()==0) r=multi_sim_comm.Get_rank();
      comm.Bcast(r,0);
      if(r>0) gridfilename_="/dev/null";
      gridfile_.enforceSuffix("");
    }
    if(mw_n_>1) gridfile_.enforceSuffix("");
    gridfile_.open(gridfilename_);
  }

  // open hills file for writing
  hillsOfile_.link(*this);
  if(walkers_mpi) {
    int r=0;
    if(comm.Get_rank()==0) r=multi_sim_comm.Get_rank();
    comm.Bcast(r,0);
    if(r>0) ifilesnames[mw_id_]="/dev/null";
    hillsOfile_.enforceSuffix("");
  }
  if(mw_n_>1) hillsOfile_.enforceSuffix("");
  hillsOfile_.open(ifilesnames[mw_id_]);
  if(fmt.length()>0) hillsOfile_.fmtField(fmt);
  hillsOfile_.addConstantField("multivariate");
  hillsOfile_.addConstantField("kerneltype");
  if(doInt_) {
    hillsOfile_.addConstantField("lower_int").printField("lower_int",lowI_);
    hillsOfile_.addConstantField("upper_int").printField("upper_int",uppI_);
  }
  hillsOfile_.setHeavyFlush();
  // output periodicities of variables
  for(unsigned i=0; i<getNumberOfArguments(); ++i) hillsOfile_.setupPrintValue( getPntrToArgument(i) );

  bool concurrent=false;
  const ActionSet&actionSet(plumed.getActionSet());
  for(const auto & p : actionSet) if(dynamic_cast<ExternalProtocol*>(p.get())) { concurrent=true; break; }
  if(concurrent) log<<"  You are using concurrent metadynamics\n";
  if(rect_biasf_.size()>0) {
    if(walkers_mpi) {
      log<<"  You are using RECT in its 'altruistic' implementation\n";
    }{
      log<<"  You are using RECT\n";
    }
  }

  log<<"  Bibliography "<<plumed.cite("Laio and Parrinello, PNAS 99, 12562 (2002)");
  if(welltemp_) log<<plumed.cite(
                       "Barducci, Bussi, and Parrinello, Phys. Rev. Lett. 100, 020603 (2008)");
  if(tt_specs_.is_active) {
    log << plumed.cite("Dama, Rotskoff, Parrinello, and Voth, J. Chem. Theory Comput. 10, 3626 (2014)");
    log << plumed.cite("Dama, Parrinello, and Voth, Phys. Rev. Lett. 112, 240602 (2014)");
  }
  if(mw_n_>1||walkers_mpi) log<<plumed.cite(
                                  "Raiteri, Laio, Gervasio, Micheletti, and Parrinello, J. Phys. Chem. B 110, 3533 (2006)");
  if(adaptive_!=FlexibleBin::none) log<<plumed.cite(
                                          "Branduardi, Bussi, and Parrinello, J. Chem. Theory Comput. 8, 2247 (2012)");
  if(doInt_) log<<plumed.cite(
                    "Baftizadeh, Cossio, Pietrucci, and Laio, Curr. Phys. Chem. 2, 79 (2012)");
  if(acceleration) log<<plumed.cite(
                          "Pratyush and Parrinello, Phys. Rev. Lett. 111, 230602 (2013)");
  if(calc_rct_) log<<plumed.cite(
                       "Pratyush and Parrinello, J. Phys. Chem. B, 119, 736 (2015)");
  if(concurrent || rect_biasf_.size()>0) log<<plumed.cite(
          "Gil-Ley and Bussi, J. Chem. Theory Comput. 11, 1077 (2015)");
  if(rect_biasf_.size()>0 && walkers_mpi) log<<plumed.cite(
          "Hosek, Toulcova, Bortolato, and Spiwok, J. Phys. Chem. B 120, 2209 (2016)");
  if(targetfilename_.length()>0) {
    log<<plumed.cite("White, Dama, and Voth, J. Chem. Theory Comput. 11, 2451 (2015)");
    log<<plumed.cite("Marinelli and Faraldo-Gómez,  Biophys. J. 108, 2779 (2015)");
    log<<plumed.cite("Gil-Ley, Bottaro, and Bussi, J. Chem. Theory Comput. 12, 2790 (2016)");
  }
  log<<"\n";
}

void ExternalProtocol::readTemperingSpecs(TemperingSpecs &t_specs) {
  // Set global tempering parameters.
  parse(t_specs.name_stem + "BIASFACTOR", t_specs.biasf);
  if (t_specs.biasf != -1.0) {
    if (kbt_ == 0.0) {
      error("Unless the MD engine passes the temperature to plumed, with tempered metad you must specify it using TEMP");
    }
    if (t_specs.biasf == 1.0) {
      error("A bias factor of 1 corresponds to zero delta T and zero hill size, so it is not allowed.");
    }
    t_specs.is_active = true;
    parse(t_specs.name_stem + "BIASTHRESHOLD", t_specs.threshold);
    if (t_specs.threshold < 0.0) {
      error(t_specs.name + " bias threshold is nonsensical");
    }
    parse(t_specs.name_stem + "ALPHA", t_specs.alpha);
    if (t_specs.alpha <= 0.0 || t_specs.alpha > 1.0) {
      error(t_specs.name + " decay shape parameter alpha is nonsensical");
    }
  }
}

void ExternalProtocol::logTemperingSpecs(const TemperingSpecs &t_specs) {
  log.printf("  %s bias factor %f\n", t_specs.name.c_str(), t_specs.biasf);
  log.printf("  KbT %f\n", kbt_);
  if (t_specs.threshold != 0.0) log.printf("  %s bias threshold %f\n", t_specs.name.c_str(), t_specs.threshold);
  if (t_specs.alpha != 1.0) log.printf("  %s decay shape parameter alpha %f\n", t_specs.name.c_str(), t_specs.alpha);
}

void ExternalProtocol::readGaussians(IFile *ifile)
{
  unsigned ncv=getNumberOfArguments();
  vector<double> center(ncv);
  vector<double> sigma(ncv);
  double height;
  int nhills=0;
  bool multivariate=false;

  std::vector<Value> tmpvalues;
  for(unsigned j=0; j<getNumberOfArguments(); ++j) tmpvalues.push_back( Value( this, getPntrToArgument(j)->getName(), false ) );

  while(scanOneHill(ifile,tmpvalues,center,sigma,height,multivariate)) {
    ;
    nhills++;
// note that for gamma=1 we store directly -F
    if(welltemp_ && biasf_>1.0) {height*=(biasf_-1.0)/biasf_;}
    addGaussian(Gaussian(center,sigma,height,multivariate));
  }
  log.printf("      %d Gaussians read\n",nhills);
}

void ExternalProtocol::writeGaussian(const Gaussian& hill, OFile&file)
{
  unsigned ncv=getNumberOfArguments();
  file.printField("time",getTimeStep()*getStep());
  for(unsigned i=0; i<ncv; ++i) {
    file.printField(getPntrToArgument(i),hill.center[i]);
  }
  hillsOfile_.printField("kerneltype","gaussian");
  if(hill.multivariate) {
    hillsOfile_.printField("multivariate","true");
    Matrix<double> mymatrix(ncv,ncv);
    unsigned k=0;
    for(unsigned i=0; i<ncv; i++) {
      for(unsigned j=i; j<ncv; j++) {
        // recompose the full inverse matrix
        mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k];
        k++;
      }
    }
    // invert it
    Matrix<double> invmatrix(ncv,ncv);
    Invert(mymatrix,invmatrix);
    // enforce symmetry
    for(unsigned i=0; i<ncv; i++) {
      for(unsigned j=i; j<ncv; j++) {
        invmatrix(i,j)=invmatrix(j,i);
      }
    }

    // do cholesky so to have a "sigma like" number
    Matrix<double> lower(ncv,ncv);
    cholesky(invmatrix,lower);
    // loop in band form
    for(unsigned i=0; i<ncv; i++) {
      for(unsigned j=0; j<ncv-i; j++) {
        file.printField("sigma_"+getPntrToArgument(j+i)->getName()+"_"+getPntrToArgument(j)->getName(),lower(j+i,j));
      }
    }
  } else {
    hillsOfile_.printField("multivariate","false");
    for(unsigned i=0; i<ncv; ++i)
      file.printField("sigma_"+getPntrToArgument(i)->getName(),hill.sigma[i]);
  }
  double height=hill.height;
// note that for gamma=1 we store directly -F
  if(welltemp_ && biasf_>1.0) height*=biasf_/(biasf_-1.0);
  file.printField("height",height).printField("biasf",biasf_);
  if(mw_n_>1) file.printField("clock",int(std::time(0)));
  file.printField();
}

void ExternalProtocol::addGaussian(const Gaussian& hill)
{
  if(!grid_) hills_.push_back(hill);
  else {
    unsigned ncv=getNumberOfArguments();
    vector<unsigned> nneighb=getGaussianSupport(hill);
    vector<GridBase::index_t> neighbors=BiasGrid_->getNeighbors(hill.center,nneighb);
    vector<double> der(ncv);
    vector<double> xx(ncv);
    if(comm.Get_size()==1) {
      for(unsigned i=0; i<neighbors.size(); ++i) {
        GridBase::index_t ineigh=neighbors[i];
        for(unsigned j=0; j<ncv; ++j) der[j]=0.0;
        BiasGrid_->getPoint(ineigh,xx);
        double bias=evaluateGaussian(xx,hill,&der[0]);
        BiasGrid_->addValueAndDerivatives(ineigh,bias,der);
      }
    } else {
      unsigned stride=comm.Get_size();
      unsigned rank=comm.Get_rank();
      vector<double> allder(ncv*neighbors.size(),0.0);
      vector<double> allbias(neighbors.size(),0.0);
      for(unsigned i=rank; i<neighbors.size(); i+=stride) {
        GridBase::index_t ineigh=neighbors[i];
        BiasGrid_->getPoint(ineigh,xx);
        allbias[i]=evaluateGaussian(xx,hill,&allder[ncv*i]);
      }
      comm.Sum(allbias);
      comm.Sum(allder);
      for(unsigned i=0; i<neighbors.size(); ++i) {
        GridBase::index_t ineigh=neighbors[i];
        for(unsigned j=0; j<ncv; ++j) {der[j]=allder[ncv*i+j];}
        BiasGrid_->addValueAndDerivatives(ineigh,allbias[i],der);
      }
    }
  }
}

vector<unsigned> ExternalProtocol::getGaussianSupport(const Gaussian& hill)
{
  vector<unsigned> nneigh;
  vector<double> cutoff;
  unsigned ncv=getNumberOfArguments();

  // traditional or flexible hill?
  if(hill.multivariate) {
    unsigned k=0;
    Matrix<double> mymatrix(ncv,ncv);
    for(unsigned i=0; i<ncv; i++) {
      for(unsigned j=i; j<ncv; j++) {
        // recompose the full inverse matrix
        mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k];
        k++;
      }
    }
    // Reinvert so to have the ellipses
    Matrix<double> myinv(ncv,ncv);
    Invert(mymatrix,myinv);
    Matrix<double> myautovec(ncv,ncv);
    vector<double> myautoval(ncv); //should I take this or their square root?
    diagMat(myinv,myautoval,myautovec);
    double maxautoval=0.;
    unsigned ind_maxautoval; ind_maxautoval=ncv;
    for(unsigned i=0; i<ncv; i++) {
      if(myautoval[i]>maxautoval) {maxautoval=myautoval[i]; ind_maxautoval=i;}
    }
    for(unsigned i=0; i<ncv; i++) {
      cutoff.push_back(sqrt(2.0*DP2CUTOFF)*abs(sqrt(maxautoval)*myautovec(i,ind_maxautoval)));
    }
  } else {
    for(unsigned i=0; i<ncv; ++i) {
      cutoff.push_back(sqrt(2.0*DP2CUTOFF)*hill.sigma[i]);
    }
  }

  if(doInt_) {
    if(hill.center[0]+cutoff[0] > uppI_ || hill.center[0]-cutoff[0] < lowI_) {
      // in this case, we updated the entire grid to avoid problems
      return BiasGrid_->getNbin();
    } else {
      nneigh.push_back( static_cast<unsigned>(ceil(cutoff[0]/BiasGrid_->getDx()[0])) );
      return nneigh;
    }
  } else {
    for(unsigned i=0; i<ncv; i++) {
      nneigh.push_back( static_cast<unsigned>(ceil(cutoff[i]/BiasGrid_->getDx()[i])) );
    }
  }

  return nneigh;
}

double ExternalProtocol::getBiasAndDerivatives(const vector<double>& cv, double* der)
{
  double bias=0.0;
  if(!grid_) {
    if(hills_.size()>10000 && (getStep()-last_step_warn_grid)>10000) {
      std::string msg;
      Tools::convert(hills_.size(),msg);
      msg="You have accumulated "+msg+" hills, you should enable GRIDs to avoid serious performance hits";
      warning(msg);
      last_step_warn_grid=getStep();
    }
    unsigned stride=comm.Get_size();
    unsigned rank=comm.Get_rank();
    for(unsigned i=rank; i<hills_.size(); i+=stride) {
      bias+=evaluateGaussian(cv,hills_[i],der);
    }
    comm.Sum(bias);
    if(der) comm.Sum(der,getNumberOfArguments());
  } else {
    if(der) {
      vector<double> vder(getNumberOfArguments());
      bias=BiasGrid_->getValueAndDerivatives(cv,vder);
      for(unsigned i=0; i<getNumberOfArguments(); ++i) {der[i]=vder[i];}
    } else {
      bias = BiasGrid_->getValue(cv);
    }
  }

  return bias;
}

double ExternalProtocol::evaluateGaussian(const vector<double>& cv, const Gaussian& hill, double* der)
{
  double dp2=0.0;
  double bias=0.0;
  // I use a pointer here because cv is const (and should be const)
  // but when using doInt it is easier to locally replace cv[0] with
  // the upper/lower limit in case it is out of range
  const double *pcv=NULL; // pointer to cv
  double tmpcv[1]; // tmp array with cv (to be used with doInt_)
  if(cv.size()>0) pcv=&cv[0];
  if(doInt_) {
    plumed_assert(cv.size()==1);
    tmpcv[0]=cv[0];
    if(cv[0]<lowI_) tmpcv[0]=lowI_;
    if(cv[0]>uppI_) tmpcv[0]=uppI_;
    pcv=&(tmpcv[0]);
  }
  if(hill.multivariate) {
    unsigned k=0;
    unsigned ncv=cv.size();
    // recompose the full sigma from the upper diag cholesky
    Matrix<double> mymatrix(ncv,ncv);
    for(unsigned i=0; i<ncv; i++) {
      for(unsigned j=i; j<ncv; j++) {
        mymatrix(i,j)=mymatrix(j,i)=hill.sigma[k]; // recompose the full inverse matrix
        k++;
      }
    }
    for(unsigned i=0; i<cv.size(); ++i) {
      double dp_i=difference(i,hill.center[i],pcv[i]);
      dp_[i]=dp_i;
      for(unsigned j=i; j<cv.size(); ++j) {
        if(i==j) {
          dp2+=dp_i*dp_i*mymatrix(i,j)*0.5;
        } else {
          double dp_j=difference(j,hill.center[j],pcv[j]);
          dp2+=dp_i*dp_j*mymatrix(i,j);
        }
      }
    }
    if(dp2<DP2CUTOFF) {
      bias=hill.height*exp(-dp2);
      if(der) {
        for(unsigned i=0; i<cv.size(); ++i) {
          double tmp=0.0;
          for(unsigned j=0; j<cv.size(); ++j) {
            tmp += dp_[j]*mymatrix(i,j)*bias;
          }
          der[i]-=tmp;
        }
      }
    }
  } else {
    for(unsigned i=0; i<cv.size(); ++i) {
      double dp=difference(i,hill.center[i],pcv[i])*hill.invsigma[i];
      dp2+=dp*dp;
      dp_[i]=dp;
    }
    dp2*=0.5;
    if(dp2<DP2CUTOFF) {
      bias=hill.height*exp(-dp2);
      if(der) {
        for(unsigned i=0; i<cv.size(); ++i) {der[i]+=-bias*dp_[i]*hill.invsigma[i];}
      }
    }
  }

  if(doInt_ && der) {
    if(cv[0]<lowI_ || cv[0]>uppI_) for(unsigned i=0; i<cv.size(); ++i) der[i]=0;
  }

  return bias;
}

void ExternalProtocol::calculate()
{
  // this is because presently there is no way to properly pass information
  // on adaptive hills (diff) after exchanges:
  if(adaptive_==FlexibleBin::diffusion && getExchangeStep()) error("ADAPTIVE=DIFF is not compatible with replica exchange");

  const unsigned ncv=getNumberOfArguments();
  vector<double> cv(ncv);
  std::unique_ptr<double[]> der(new double[ncv]);
  for(unsigned i=0; i<ncv; ++i) {
    cv[i]=getArgument(i);
    der[i]=0.;
  }
  double ene = getBiasAndDerivatives(cv,der.get());
// special case for gamma=1.0
  if(biasf_==1.0) {
    ene=0.0;
    for(unsigned i=0; i<getNumberOfArguments(); ++i) {der[i]=0.0;}
  }

  setBias(ene);

  //update inst_work_
  if (isFirstStep)
  {
    for (unsigned i=0; i<nhills_; i++)
      current_hills_.emplace_back(cv,sigma0_,0,false);
  }
  for (unsigned i=0; i<nhills_; i++)
    inst_work_+=evaluateGaussian(cv,current_hills_[i],NULL);
  if(calc_rct_) getPntrToComponent("rbias")->set(ene - reweight_factor_);
  // calculate the acceleration factor
  if(acceleration&&!isFirstStep) {
    acc += static_cast<double>(getStride()) * exp(ene/(kbt_));
    const double mean_acc = acc/((double) getStep());
    getPntrToComponent("acc")->set(mean_acc);
  } else if (acceleration && isFirstStep && acc_restart_mean_ > 0.0) {
    acc = acc_restart_mean_ * static_cast<double>(getStep());
  }

//  getPntrToComponent("work")->set(work_);
  // set Forces
  for(unsigned i=0; i<ncv; ++i) {
    setOutputForce(i,-der[i]);
  }
}

void ExternalProtocol::update() {
  vector<double> cv(getNumberOfArguments());
  vector<double> thissigma;
  bool multivariate;

  // adding hills criteria (could be more complex though)
  bool nowAddAHill;
  if(getStep()%stride_==0 && !isFirstStep )nowAddAHill=true;
  else {
    nowAddAHill=false;
    isFirstStep=false;
  }

  for(unsigned i=0; i<cv.size(); ++i) cv[i] = getArgument(i);

  double vbias=getBiasAndDerivatives(cv)-reweight_factor_;

  // if you use adaptive, call the FlexibleBin
  if(adaptive_!=FlexibleBin::none) {
    flexbin->update(nowAddAHill);
    multivariate=true;
  } else {
    multivariate=false;
  }

//--------------------------------------------------------------------------------------------------
//read newHill from protocol file
  if(nowAddAHill) {
    const unsigned ncv=getNumberOfArguments();
    for(unsigned n=0; n<nhills_; n++) {
      vector<double> center(ncv);
      vector<double> sigma(ncv);
      double height=0;
      bool multivariate=false;
      int int_multivariate=0;
      if(comm.Get_rank()==0 && multi_sim_comm.Get_rank()==0) {
        std::vector<Value> tmpvalues;
        for(unsigned j=0; j<getNumberOfArguments(); ++j)
          tmpvalues.push_back( Value( this, getPntrToArgument(j)->getName(), false ) );
        if(scanOneHill(iprotocol.get(),tmpvalues,center,sigma,height,multivariate)) {
          // note that for gamma=1 we store directly -F
          int_multivariate=multivariate; //an int is needed for Bcast
          if(welltemp_ && biasf_>1.0)
            height*=(biasf_-1.0)/biasf_;
          if (inverted_protocol_)
            height*=(-1.);
        } else {
          log.printf("  END of protocol reached!!!\n");
          //many possibilities, for instance go back on your step in a cycle
        }
        iprotocol->reset(false);
      }
      if(comm.Get_rank()==0) {
        multi_sim_comm.Bcast(center,0);
        multi_sim_comm.Bcast(sigma,0);
        multi_sim_comm.Bcast(height,0);
        multi_sim_comm.Bcast(int_multivariate,0);
      }
      comm.Bcast(center,0);
      comm.Bcast(sigma,0);
      comm.Bcast(height,0);
      comm.Bcast(int_multivariate,0);
      multivariate=int_multivariate;
      if (height!=0) { //should be zero only if end of protocol was reached
        current_hills_[n]=Gaussian(center,sigma,height,multivariate);
        addGaussian(current_hills_[n]);
        // print on HILLS file
        writeGaussian(current_hills_[n],hillsOfile_);
      }
    }
//    getPntrToComponent("instwork")->set(inst_work_);
    inst_work_=0;
  }
//--------------------------------------------------------------------------------------------------


  // dump grid on file
  if(wgridstride_>0&&(getStep()%wgridstride_==0||getCPT())) {
    // in case old grids are stored, a sequence of grids should appear
    // this call results in a repetition of the header:
    if(storeOldGrids_) gridfile_.clearFields();
    // in case only latest grid is stored, file should be rewound
    // this will overwrite previously written grids
    else {
      int r = 0;
      if(walkers_mpi) {
        if(comm.Get_rank()==0) r=multi_sim_comm.Get_rank();
        comm.Bcast(r,0);
      }
      if(r==0) gridfile_.rewind();
    }
    BiasGrid_->writeToFile(gridfile_);
    // if a single grid is stored, it is necessary to flush it, otherwise
    // the file might stay empty forever (when a single grid is not large enough to
    // trigger flushing from the operating system).
    // on the other hand, if grids are stored one after the other this is
    // no necessary, and we leave the flushing control to the user as usual
    // (with FLUSH keyword)
    if(!storeOldGrids_) gridfile_.flush();
  }

  // Recalculate special bias quantities whenever the bias has been changed by the update.
  bool bias_has_changed = (nowAddAHill || (mw_n_ > 1 && getStep() % mw_rstride_ == 0));
  if (calc_rct_ && bias_has_changed && getStep()%(stride_*rct_ustride_)==0) computeReweightingFactor();
  if (calc_max_bias_ && bias_has_changed) {
    max_bias_ = BiasGrid_->getMaxValue();
    getPntrToComponent("maxbias")->set(max_bias_);
  }
  if (calc_transition_bias_ && bias_has_changed) {
    transition_bias_ = getTransitionBarrierBias();
    getPntrToComponent("transbias")->set(transition_bias_);
  }

  double vbias1=getBiasAndDerivatives(cv)-reweight_factor_;
  work_+=vbias1-vbias;
  getPntrToComponent("instwork")->set(vbias1-vbias);
  getPntrToComponent("work")->set(work_);
}

/// takes a pointer to the file and a template string with values v and gives back the next center, sigma and height
bool ExternalProtocol::scanOneHill(IFile *ifile,  vector<Value> &tmpvalues, vector<double> &center, vector<double>  &sigma, double &height, bool &multivariate)
{
  double dummy;
  multivariate=false;
  if(ifile->scanField("time",dummy)) {
    unsigned ncv; ncv=tmpvalues.size();
    for(unsigned i=0; i<ncv; ++i) {
      ifile->scanField( &tmpvalues[i] );
      if( tmpvalues[i].isPeriodic() && ! getPntrToArgument(i)->isPeriodic() ) {
        error("in hills file periodicity for variable " + tmpvalues[i].getName() + " does not match periodicity in input");
      } else if( tmpvalues[i].isPeriodic() ) {
        std::string imin, imax; tmpvalues[i].getDomain( imin, imax );
        std::string rmin, rmax; getPntrToArgument(i)->getDomain( rmin, rmax );
        if( imin!=rmin || imax!=rmax ) {
          error("in hills file periodicity for variable " + tmpvalues[i].getName() + " does not match periodicity in input");
        }
      }
      center[i]=tmpvalues[i].get();
    }
    // scan for kerneltype
    std::string ktype="gaussian";
    if( ifile->FieldExist("kerneltype") ) ifile->scanField("kerneltype",ktype);
    // scan for multivariate label: record the actual file position so to eventually rewind
    std::string sss;
    ifile->scanField("multivariate",sss);
    if(sss=="true") multivariate=true;
    else if(sss=="false") multivariate=false;
    else plumed_merror("cannot parse multivariate = "+ sss);
    if(multivariate) {
      sigma.resize(ncv*(ncv+1)/2);
      Matrix<double> upper(ncv,ncv);
      Matrix<double> lower(ncv,ncv);
      for(unsigned i=0; i<ncv; i++) {
        for(unsigned j=0; j<ncv-i; j++) {
          ifile->scanField("sigma_"+getPntrToArgument(j+i)->getName()+"_"+getPntrToArgument(j)->getName(),lower(j+i,j));
          upper(j,j+i)=lower(j+i,j);
        }
      }
      Matrix<double> mymult(ncv,ncv);
      Matrix<double> invmatrix(ncv,ncv);
      mult(lower,upper,mymult);
      // now invert and get the sigmas
      Invert(mymult,invmatrix);
      // put the sigmas in the usual order: upper diagonal (this time in normal form and not in band form)
      unsigned k=0;
      for(unsigned i=0; i<ncv; i++) {
        for(unsigned j=i; j<ncv; j++) {
          sigma[k]=invmatrix(i,j);
          k++;
        }
      }
    } else {
      for(unsigned i=0; i<ncv; ++i) {
        ifile->scanField("sigma_"+getPntrToArgument(i)->getName(),sigma[i]);
      }
    }

    ifile->scanField("height",height);
    ifile->scanField("biasf",dummy);
    if(ifile->FieldExist("clock")) ifile->scanField("clock",dummy);
    if(ifile->FieldExist("lower_int")) ifile->scanField("lower_int",dummy);
    if(ifile->FieldExist("upper_int")) ifile->scanField("upper_int",dummy);
    ifile->scanField();
    return true;
  } else {
    return false;
  }
}

void ExternalProtocol::computeReweightingFactor()
{
  if(biasf_==1.0) { // in this case we have no bias, so reweight factor is 0.0
    getPntrToComponent("rct")->set(0.0);
    return;
  }

  double Z_0=0; //proportional to the integral of exp(-beta*F)
  double Z_V=0; //proportional to the integral of exp(-beta*(F+V))
  double minusBetaF=biasf_/(biasf_-1.)/kbt_;
  double minusBetaFplusV=1./(biasf_-1.)/kbt_;
  if (biasf_==-1.0) { //non well-tempered case
    minusBetaF=1;
    minusBetaFplusV=0;
  }
  const double big_number=minusBetaF*BiasGrid_->getMaxValue(); //to avoid exp overflow

  const unsigned rank=comm.Get_rank();
  const unsigned stride=comm.Get_size();
  for (GridBase::index_t t=rank; t<BiasGrid_->getSize(); t+=stride) {
    const double val=BiasGrid_->getValue(t);
    Z_0+=std::exp(minusBetaF*val-big_number);
    Z_V+=std::exp(minusBetaFplusV*val-big_number);
  }
  if (stride>1) {
    comm.Sum(Z_0);
    comm.Sum(Z_V);
  }

  reweight_factor_=kbt_*std::log(Z_0/Z_V);
  getPntrToComponent("rct")->set(reweight_factor_);
}

double ExternalProtocol::getTransitionBarrierBias() {

  // If there is only one well of interest, return the bias at that well point.
  if (transitionwells_.size() == 1) {
    double tb_bias = getBiasAndDerivatives(transitionwells_[0], NULL);
    return tb_bias;

    // Otherwise, check for the least barrier bias between all pairs of wells.
    // Note that because the paths can be considered edges between the wells' nodes
    // to make a graph and the path barriers satisfy certain cycle inequalities, it
    // is sufficient to look at paths corresponding to a minimal spanning tree of the
    // overall graph rather than examining every edge in the graph.
    // For simplicity, I chose the star graph with center well 0 as the spanning tree.
    // It is most efficient to start the path searches from the wells that are
    // expected to be sampled last, so transitionwell_[0] should correspond to the
    // starting well. With this choice the searches will terminate in one step until
    // transitionwell_[1] is sampled.
  } else {
    double least_transition_bias;
    vector<double> sink = transitionwells_[0];
    vector<double> source = transitionwells_[1];
    least_transition_bias = BiasGrid_->findMaximalPathMinimum(source, sink);
    for (unsigned i = 2; i < transitionwells_.size(); i++) {
      if (least_transition_bias == 0.0) {
        break;
      }
      source = transitionwells_[i];
      double curr_transition_bias = BiasGrid_->findMaximalPathMinimum(source, sink);
      least_transition_bias = fmin(curr_transition_bias, least_transition_bias);
    }
    return least_transition_bias;
  }
}

}
}
