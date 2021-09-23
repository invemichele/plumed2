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
#ifndef __PLUMED_colvar_LandauFourierComp_h
#define __PLUMED_colvar_LandauFourierComp_h
#include "Colvar.h"

//+PLUMEDOC COLVAR LANDAU_FOURIER_COMP
/*
  The idea is that LandauFourierComp and VariationalLandau can work only if used together,
  so this header is needed.
*/
//+ENDPLUMEDOC

namespace PLMD {

//compares plumed vectors looking at their module. Needed to sort fwv_, can't be a class method
static inline bool Landau_VectorsCompare (Vector a, Vector b)
{ return a.modulo2()<b.modulo2(); }

namespace colvar {

class LandauFourierComp : public Colvar {

private:
  bool first_run_;
  bool add_rho_in_zero_;//to monitor the density in one point (0,0,0)
  bool parallel_;
  bool no_virial_;

  unsigned int NumAtom_;
  double Vol_;
  double sigma_; //sigma for gaussian model
  std::vector<double> sigma_weight_; //used if sigma!=0
  std::vector<Vector> fwv_; //fourier wave vector
  std::vector<double> Re_rho_;
  std::vector<double> Im_rho_;

  Value* valueRhoInZero;
  std::vector<Value*> valueReRho;
  std::vector<Value*> valueImRho;

  void generate_int_fwv(int);

public:
  LandauFourierComp(const ActionOptions&);
  virtual void calculate();
  static void registerKeywords(Keywords& );

  double getVol() const { return Vol_; }; //returns -1 if volume is not yet initialized
  unsigned int getNumAtom() const { return NumAtom_; };
  unsigned int getFwvSize() const { return fwv_.size(); };
  const double* getRe_adr() const { return &Re_rho_[0]; };
  const double* getIm_adr() const { return &Im_rho_[0]; };
};

}
}
#endif
