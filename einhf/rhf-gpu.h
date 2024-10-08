/*
 * @BEGIN LICENSE
 *
 * einhf_gpu by Psi4 Developer, a plugin to:
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

#include <deque>
#include <vector>

#include "psi4/libfock/v.h"
#include "psi4/libfunctional/superfunctional.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/psi4-dec.h"

#include "einsums.hpp"

namespace psi {
// Forward declare several variables
class Options;
class JK;

namespace einhf {

class GPUEinsumsRHF : public Wavefunction {
public:
  /// The constuctor
  GPUEinsumsRHF(SharedWavefunction ref_wfn,
                const std::shared_ptr<SuperFunctional> &functional,
                Options &options);
  /// The destuctor
  ~GPUEinsumsRHF();
  /// Computes the SCF energy, and returns it
  double compute_energy();

  void compute_diis_coefs(
      const std::deque<einsums::BlockDeviceTensor<double, 2>> &errors,
      std::vector<double> *out) const;

  void compute_diis_fock(
      const std::vector<double> &coefs,
      const std::deque<einsums::BlockDeviceTensor<double, 2>> &focks,
      einsums::BlockDeviceTensor<double, 2> *out) const;

  void print_header();

  const einsums::BlockDeviceTensor<double, 2> &getH() const { return H_; }
  const einsums::BlockDeviceTensor<double, 2> &getS() const { return S_; }
  const einsums::BlockDeviceTensor<double, 2> &getX() const { return X_; }
  const einsums::BlockDeviceTensor<double, 2> &getF() const { return F_; }
  const einsums::BlockDeviceTensor<double, 2> &getFt() const { return Ft_; }
  const einsums::BlockDeviceTensor<double, 2> &getC() const { return C_; }
  const einsums::BlockDeviceTensor<double, 2> &getCocc() const { return Cocc_; }
  const einsums::BlockDeviceTensor<double, 2> &getD() const { return D_; }

  const einsums::DeviceTensor<double, 1> &getEvals() const { return evals_; }

  const std::vector<int> &getIrrepSizes() const { return irrep_sizes_; }
  const std::vector<int> &getOccPerIrrep() const { return occ_per_irrep_; }

  einsums::BlockDeviceTensor<double, 2> &getH() { return H_; }
  einsums::BlockDeviceTensor<double, 2> &getS() { return S_; }
  einsums::BlockDeviceTensor<double, 2> &getX() { return X_; }
  einsums::BlockDeviceTensor<double, 2> &getF() { return F_; }
  einsums::BlockDeviceTensor<double, 2> &getFt() { return Ft_; }
  einsums::BlockDeviceTensor<double, 2> &getC() { return C_; }
  einsums::BlockDeviceTensor<double, 2> &getCocc() { return Cocc_; }
  einsums::BlockDeviceTensor<double, 2> &getD() { return D_; }

  einsums::DeviceTensor<double, 1> &getEvals() { return evals_; }

  int getNDocc() const { return ndocc_; }

protected:
  /// The amount of information to print to the output file
  int print_;
  /// The number of doubly occupied orbitals
  int ndocc_;

  /// The occupation per irrep.
  std::vector<int> occ_per_irrep_;
  /// The sizes of each irrep.
  std::vector<int> irrep_sizes_;

  /// The number of symmetrized spin orbitals
  int nso_;
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
  einsums::BlockDeviceTensor<double, 2> H_;
  /// The overlap matrix
  einsums::BlockDeviceTensor<double, 2> S_;
  /// The inverse square root of the overlap matrix
  einsums::BlockDeviceTensor<double, 2> X_;
  /// The Fock Matrix
  einsums::BlockDeviceTensor<double, 2> F_;
  /// The Fock Matrix
  einsums::BlockDeviceTensor<double, 2> JKwK_;
  /// The transformed Fock matrix
  einsums::BlockDeviceTensor<double, 2> Ft_;
  /// The MO coefficients
  einsums::BlockDeviceTensor<double, 2> C_;
  /// The occupied MO coefficients
  einsums::BlockDeviceTensor<double, 2> Cocc_;
  /// The density matrix
  einsums::BlockDeviceTensor<double, 2> D_;
  /// The ubiquitous JK object
  std::shared_ptr<JK> jk_;
  /// The functional.
  std::shared_ptr<SuperFunctional> func_;
  /// The functional exchange integrator.
  std::shared_ptr<VBase> v_;
  /// The orbital energies.
  einsums::DeviceTensor<double, 1> evals_;
  /// Computes the electronic part of the SCF energy, and returns it
  double compute_electronic_energy();
  /// Sets up the integrals object
  void init_integrals();
  /// Updates the occupied MO coefficients
  void update_Cocc(const einsums::DeviceTensor<double, 1> &energies);
};

} // namespace einhf
} // namespace psi
