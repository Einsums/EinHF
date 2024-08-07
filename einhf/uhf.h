/*
 * @BEGIN LICENSE
 *
 * einhf by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2023 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with Psi4; if not, write to the Free Software Foundation, Inc., 51
 * Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#pragma once

#include "einsums.hpp"
#include <deque>
#include <vector>

#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

class EinsumsUHF : public Wavefunction {
public:
  /// The constuctor
  EinsumsUHF(SharedWavefunction ref_wfn,
             const std::shared_ptr<SuperFunctional> &functional,
             Options &options);
  /// Copy constructor.
  EinsumsUHF(const EinsumsUHF &ref_wfn);
  /// Another constructor.
  EinsumsUHF(const EinsumsUHF &ref_wfn, Options &options);
  /// The destuctor
  ~EinsumsUHF();
  /// Computes the SCF energy, and returns it
  virtual double compute_energy();

  void
  compute_diis_coefs(const std::deque<einsums::BlockTensor<double, 2>> &errors,
                     std::vector<double> *out) const;

  void
  compute_diis_fock(const std::vector<double> &coefs,
                    const std::deque<einsums::BlockTensor<double, 2>> &focks,
                    einsums::BlockTensor<double, 2> *out) const;

  virtual void print_header();

  const einsums::BlockTensor<double, 2> &getH() const { return H_; }
  const einsums::BlockTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockTensor<double, 2> &getX() const { return X_; }
  const einsums::BlockTensor<double, 2> &getF() const { return Fa_; }
  const einsums::BlockTensor<double, 2> &getFt() const { return Fta_; }
  const einsums::BlockTensor<double, 2> &getC() const { return Ca_; }
  const einsums::BlockTensor<double, 2> &getCocc() const { return Cocca_; }
  const einsums::BlockTensor<double, 2> &getD() const { return Da_; }
  const einsums::Tensor<double, 1> &getEvals() const { return evalsa_; }

  einsums::BlockTensor<double, 2> &getH() { return H_; }
  einsums::BlockTensor<double, 2> &getS() { return S_; }
  einsums::BlockTensor<double, 2> &getX() { return X_; }
  einsums::BlockTensor<double, 2> &getF() { return Fa_; }
  einsums::BlockTensor<double, 2> &getFt() { return Fta_; }
  einsums::BlockTensor<double, 2> &getC() { return Ca_; }
  einsums::BlockTensor<double, 2> &getCocc() { return Cocca_; }
  einsums::BlockTensor<double, 2> &getD() { return Da_; }
  einsums::Tensor<double, 1> &getEvals() { return evalsa_; }

  const einsums::BlockTensor<double, 2> &getFa() const { return Fa_; }
  const einsums::BlockTensor<double, 2> &getFta() const { return Fta_; }
  const einsums::BlockTensor<double, 2> &getCa() const { return Ca_; }
  const einsums::BlockTensor<double, 2> &getCocca() const { return Cocca_; }
  const einsums::BlockTensor<double, 2> &getDa() const { return Da_; }
  const einsums::Tensor<double, 1> &getEvalsa() const { return evalsa_; }

  einsums::BlockTensor<double, 2> &getFa() { return Fa_; }
  einsums::BlockTensor<double, 2> &getFta() { return Fta_; }
  einsums::BlockTensor<double, 2> &getCa() { return Ca_; }
  einsums::BlockTensor<double, 2> &getCocca() { return Cocca_; }
  einsums::BlockTensor<double, 2> &getDa() { return Da_; }
  einsums::Tensor<double, 1> &getEvalsa() { return evalsa_; }

  const einsums::BlockTensor<double, 2> &getFb() const { return Fb_; }
  const einsums::BlockTensor<double, 2> &getFtb() const { return Ftb_; }
  const einsums::BlockTensor<double, 2> &getCb() const { return Cb_; }
  const einsums::BlockTensor<double, 2> &getCoccb() const { return Coccb_; }
  const einsums::BlockTensor<double, 2> &getDb() const { return Db_; }
  const einsums::Tensor<double, 1> &getEvalsb() const { return evalsb_; }

  einsums::BlockTensor<double, 2> &getFb() { return Fb_; }
  einsums::BlockTensor<double, 2> &getFtb() { return Ftb_; }
  einsums::BlockTensor<double, 2> &getCb() { return Cb_; }
  einsums::BlockTensor<double, 2> &getCoccb() { return Coccb_; }
  einsums::BlockTensor<double, 2> &getDb() { return Db_; }
  einsums::Tensor<double, 1> &getEvalsb() { return evalsb_; }

protected:
  /// The amount of information to print to the output file
  int print_;

  /// The occupation per irrep.
  std::vector<int> aocc_per_irrep_;
  std::vector<int> bocc_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_;

  /// The number of symmetrized spin orbitals
  int nso_;
  /// The number of alpha and beta occupied oribitals.
  int naocc_, nbocc_;
  /// The maximum number of iterations
  int maxiter_;
  /// The number of DIIS iterations to hold.
  int diis_max_iters_;
  /// The nuclear repulsion energy
  double e_nuc_;
  /// The convergence criterion for the density
  double d_convergence_;
  /// The convergence criterion for the energy
  double e_convergence_;
  /// The one electron integrals
  einsums::BlockTensor<double, 2> H_;
  /// The overlap matrix
  einsums::BlockTensor<double, 2> S_;
  /// The inverse square root of the overlap matrix
  einsums::BlockTensor<double, 2> X_;
  /// The Fock Matrix
  einsums::BlockTensor<double, 2> Fa_, Fb_;
  /// The two-electron non-exchange contributions.
  einsums::BlockTensor<double, 2> JKwKa_, JKwKb_;
  /// The transformed Fock matrix
  einsums::BlockTensor<double, 2> Fta_, Ftb_;
  /// The MO coefficients
  einsums::BlockTensor<double, 2> Ca_, Cb_;
  /// The occupied MO coefficients
  einsums::BlockTensor<double, 2> Cocca_, Coccb_;
  /// The density matrix
  einsums::BlockTensor<double, 2> Da_, Db_;
  /// The orbital energies.
  einsums::Tensor<double, 1> evalsa_, evalsb_;
  /// The ubiquitous JK object
  std::shared_ptr<JK> jk_;
  /// The functional.
  std::shared_ptr<SuperFunctional> func_;
  /// The functional exchange integrator.
  std::shared_ptr<VBase> v_;
  /// Computes the electronic part of the SCF energy, and returns it
  double compute_electronic_energy(const einsums::BlockTensor<double, 2> &JKwK,
                                   const einsums::BlockTensor<double, 2> &D);
  /// Sets up the integrals object
  void init_integrals();
  /// Updates the occupied MO coefficients
  void update_Cocc(const einsums::Tensor<double, 1> &alpha_energies,
                   const einsums::Tensor<double, 1> &beta_energies);
};
} // namespace einhf
} // namespace psi