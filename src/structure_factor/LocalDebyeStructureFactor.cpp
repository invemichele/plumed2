/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  Development version of the Local Structure Factor collective variable, 
  in its Debye approximation.

  Author: Luigi Bonati - https://github.com/luigibonati

  Please read and cite: 
    "Silicon Liquid Structure and Crystal Nucleation from Ab Initio Deep Metadynamics"
    Bonati, Parrinello - PRL (2018)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "multicolvar/MultiColvarBase.h"
#include "multicolvar/AtomValuePack.h"
#include "tools/NeighborList.h"
#include "core/ActionRegister.h"
#include "tools/SwitchingFunction.h"
#if __has_include("Compton_consts.h")
#  include "Compton_consts.h" //source: http://lammps.sandia.gov/doc/compute_xrd.html
#endif

#include <string>
#include <cmath>

using namespace std;

namespace PLMD{
namespace multicolvar{

//+PLUMEDOC MCOLVAR LOCAL_DEBYE_STRUCTURE_FACTOR
/*
Local Debye Structure Factor of atom i is defined as:
\f[
  S_i(q) = 1+ \frac{1}{\sum^N_i f_i^2(q)} \sum^N_{j} f_i(q)f_j(q) \frac{sin(q r_{ij})}{q r_{ij}}w(r_{ij})
\f]

\par Examples
some usage examples
\plumedfile
sql: LOCAL_DEBYE_STRUCTURE_FACTOR SPECIES=1:216 ACTIVE_Q=1.96 CUTOFF=8
sql_t: LOCAL_DEBYE_STRUCTURE_FACTOR SPECIES=1:216 LAMBDA=1.5406 ACTIVE_2THETA=13.5
\endplumedfile

*/
//+ENDPLUMEDOC


class LocalStructureFactor : public MultiColvarBase {
private:
//  double nl_cut;
  double Q_;
  double cutoff_;
  std::string typeA,typeB;
  vector<int> atomType;
  unsigned numberOfAatoms, numberOfBatoms;
  double f_AA,f_BB,f_AB;
  bool mono=true;

public:
  static void registerKeywords( Keywords& keys );
  explicit LocalStructureFactor(const ActionOptions&);
// active methods:
  virtual double compute( const unsigned& tindex, AtomValuePack& myatoms ) const ;
/// Returns the number of coordinates of the field
  bool isPeriodic(){ return false; }
};

PLUMED_REGISTER_ACTION(LocalStructureFactor,"LOCAL_DEBYE_STRUCTURE_FACTOR")

void LocalStructureFactor::registerKeywords( Keywords& keys ){
  MultiColvarBase::registerKeywords( keys );
  keys.use("SPECIES"); keys.use("SPECIESA"); keys.use("SPECIESB");
  keys.add("atoms","GROUPA","Atoms of type A");
  keys.add("atoms","GROUPB","Atoms of type B");
  keys.add("optional","TYPEA","Chemical species A");
  keys.add("optional","TYPEB","Chemical species B");
  keys.add("optional","ACTIVE_Q","Q vector of the peak");
  keys.add("optional","ACTIVE_2THETA","manually set which frequencies will be considered, by setting the angle in degrees");
  keys.add("optional","LAMBDA","wavelength of incident radiation. Compulsory when using angles instead of frequencies");
  keys.add("compulsory","CUTOFF","Cutoff radius for calculation of S(Q)");

  // Use actionWithDistributionKeywords
  keys.use("MEAN"); keys.use("MORE_THAN"); keys.use("LESS_THAN"); keys.use("MAX");
  keys.use("MIN"); keys.use("BETWEEN"); keys.use("HISTOGRAM"); keys.use("MOMENTS");
  keys.use("ALT_MIN"); keys.use("LOWEST"); keys.use("HIGHEST");
}

LocalStructureFactor::LocalStructureFactor(const ActionOptions&ao):
Action(ao),
MultiColvarBase(ao)
{

  //parse lists
  vector<AtomNumber> ga_lista,gb_lista;
  parseAtomList("GROUPA",ga_lista);
  parseAtomList("GROUPB",gb_lista);
  numberOfAatoms=ga_lista.size();
  numberOfBatoms=gb_lista.size();

  if (numberOfAatoms>0 && numberOfBatoms>0) mono=false;

  if(!mono)
  {
  parse("TYPEA",typeA);
  parse("TYPEB",typeB);
  }

  parse("CUTOFF",cutoff_);
  log.printf(" -- PBC: distances will be calculated using the minimal image convention\n");
  log.printf("  distance CUTOFF radius: %g",cutoff_);

  //- get active q frequencies
    double lambda=-1;
    parse("LAMBDA",lambda);
    const auto from_2theta_to_q=[&lambda](const double _2theta)
      { plumed_massert(_2theta>0 && _2theta<=180,"2theta must be between 0 and 180 degrees");
        return 4*PLMD::pi/lambda*sin(_2theta*PLMD::pi/360); };
    double active_2theta=-1;
    Q_=-1;
    parse("ACTIVE_2THETA",active_2theta);
    parse("ACTIVE_Q",Q_);
    if (Q_>0.)
    {
      plumed_massert(lambda==-1,"when manually setting ACTIVE_Q, no LAMBDA is needed");
      plumed_massert(active_2theta<0,"either set ACTIVE_Q or ACTIVE_2THETA or none");
    }
    else if (active_2theta>0)
    {
      plumed_massert(lambda!=-1,"a LAMBDA is needed in order to set ACTIVE_2THETA");
        Q_=from_2theta_to_q(active_2theta);
      log.printf("  converting 2theta values to q=4*pi/lambda*sin(2theta*pi/360), using LAMBDA = %g\n",lambda);
    }

  // Set the link cell cutoff
  setLinkCellCutoff( cutoff_ );

  // And setup the ActionWithVessel
  std::vector<AtomNumber> all_atoms; setupMultiColvarBase( all_atoms );

  //retrieve atom types
  if(!mono){
  if (getNumberOfAtoms()!=(ga_lista.size() + gb_lista.size() )) error("Number of atoms in SPECIES is different from the sum of the number of atoms in GROUPA and GROUPB. ");
  atomType.reserve(getNumberOfAtoms());
  for(unsigned i=0;i<getNumberOfAtoms();++i){
     atomType[i]=0;
     for(unsigned j=0;j<ga_lista.size();++j){
        if (getAbsoluteIndex(i)==ga_lista[j]) atomType[i]=1;
     }
     for(unsigned j=0;j<gb_lista.size();++j){
        if (getAbsoluteIndex(i)==gb_lista[j]) atomType[i]=2;
     }
     //log.printf("Reached atom %d index %d type %d \n", i, getAbsoluteIndex(i), atomType[i]);
     if (atomType[i]==0) error("At least one atom in SPECIES doesn't have a counterpart in GROUPA or GROUPB. ");
     //log.printf("index %d, atomType %d \n", getAbsoluteIndex(i), atomType[i]);
    }

  // compute form factors
  double compton_A=0, compton_B=0;
  unsigned a_type=XRDmaxType, b_type=XRDmaxType;
  // retrieve values for type A
  for (unsigned x=0; x<XRDmaxType; x++)
  {
    if (strcasecmp(typeA.c_str(),XRDtypeList[x])==0)
    {
      a_type=x;
      break;
    }
  }
  // retrieve values for type B
  for (unsigned x=0; x<XRDmaxType; x++)
  {
    if (strcasecmp(typeB.c_str(),XRDtypeList[x])==0)
    {
      b_type=x;
      break;
    }
  }
  plumed_massert(a_type<XRDmaxType,"no match found for ATOM_TYPE="+typeA);
  plumed_massert(b_type<XRDmaxType,"no match found for ATOM_TYPE="+typeB);

  const double argument=pow(Q_/(4*PLMD::pi),2);
  for (unsigned C=0; C<8; C+=2){
    compton_A+=ASFXRD[a_type][C]*exp(-1*ASFXRD[a_type][C+1]*argument);
    compton_B+=ASFXRD[b_type][C]*exp(-1*ASFXRD[b_type][C+1]*argument);
  }
  compton_A+=ASFXRD[a_type][8];
  compton_B+=ASFXRD[b_type][8];

  f_AA=compton_A*compton_A;
  f_BB=compton_B*compton_B;
  f_AB=compton_A*compton_B;

  double norm=numberOfAatoms*f_AA+numberOfBatoms*f_BB;

  f_AA/=norm;
  f_BB/=norm;
  f_AB/=norm;

  }

  // all done
  checkRead();
}

double LocalStructureFactor::compute( const unsigned& tindex, AtomValuePack& myatoms ) const {

   // Define output quantities
   double DebyeS=0;
   double N=getNumberOfAtoms();

   vector<Vector> deriv(getNumberOfAtoms());
   Tensor virial;

   for(unsigned j=1;j<myatoms.getNumberOfAtoms();++j){
      Vector& vR_ij=myatoms.getPosition(j); 		//relative position of atom j (with respect to i)
      const double R_ij=vR_ij.modulo();

      const double QR=Q_*R_ij;
      double cosQR,sinQR;

      if (R_ij>=cutoff_)
        continue;

      double form_factor=1./N; // it is defined as f_i * f_j / normalization

    if(!mono)
    {
      if (atomType[myatoms.getIndex(0)]==1) { //i-atom is type A
        if (atomType[myatoms.getIndex(j)]==1) form_factor=f_AA;
        else form_factor=f_AB;
      }
      else {//i-atom is type B
        if (atomType[myatoms.getIndex(j)]==1) form_factor=f_AB;
        else form_factor=f_BB;
      }
    }
      //window correction function
      const double window_arg=PLMD::pi*R_ij/cutoff_;
      double window=sin(window_arg)/window_arg;
      double d_window=(window_arg*cos(window_arg)-sin(window_arg))/window_arg/R_ij;

			// -- COMPUTE --
      sinQR=sin(QR);
      cosQR=cos(QR);
      DebyeS+=form_factor*window*sinQR/QR;

			// -- (2) DERIVATE --
      Vector vDeriv_ij=N*form_factor*vR_ij*((window*(QR*cosQR-sinQR)/R_ij+d_window*sinQR)/QR/R_ij);

			deriv[0] -= vDeriv_ij;
			deriv[j] += vDeriv_ij;

      addAtomDerivatives(1,j,deriv[j],myatoms);

			// -- (3) VIRIAL --
			 myatoms.addBoxDerivatives( 1, -Tensor(vR_ij,vDeriv_ij) );

      }


   addAtomDerivatives(1,0,deriv[0],myatoms);

//   std::cout << DebyeS << endl;
   DebyeS=1+N*DebyeS;

   return DebyeS;
}

}
}
