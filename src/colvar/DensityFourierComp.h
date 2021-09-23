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
#ifndef __PLUMED_colvar_DensityFourierComp_h
#define __PLUMED_colvar_DensityFourierComp_h
#include "Colvar.h"

//+PLUMEDOC COLVAR LANDAU_FOURIER_COMP
/*
  The idea is that DensityFourierComp and DensityFouriesVar can work only if used together,
  so this header is needed.
*/
//+ENDPLUMEDOC

namespace PLMD {
namespace colvar {

class DensityFourierComp : public Colvar {

private:
  bool first_run_;
  bool no_virial_;
  bool parallel_;
  bool old_;

  unsigned NumAtom_;
  unsigned n_max_;
  double box_edge_;
  double k_const_;
  double sigma2_; //sigma for gaussian model
  std::vector<Vector> fwv_; //fourier wave vectors
  std::vector<double> Re_rho_;
  std::vector<double> Im_rho_;

  std::vector<Value*> valueReRho;
  std::vector<Value*> valueImRho;

  void generate_int_fwv(int);

public:
  DensityFourierComp(const ActionOptions&);
  virtual void calculate();
  static void registerKeywords(Keywords& );

  unsigned getNumAtom() const { return NumAtom_; };
  double getBoxEdge() const { return box_edge_; }; //returns -1 if box_edge_ is not yet initialized
  unsigned getFwvSize() const { return fwv_.size(); };
//  const std::vector<Vector>* getFwv_p() const { return &fwv_; }; //fwv_[0].modulo2()==(2*pi/L)^2
  const double* getRe_adr() const { return &Re_rho_[0]; };
  const double* getIm_adr() const { return &Im_rho_[0]; };
};

}
}
#endif
