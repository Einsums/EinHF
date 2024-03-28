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

#include "scf.h"

#include "einsums.hpp"

#include "psi4/libfock/jk.h"
#include "psi4/libmints/basisset.h"
#include "psi4/libmints/integral.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/molecule.h"
#include "psi4/libmints/sobasis.h"
#include "psi4/libmints/vector.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/libpsi4util/process.h"
#include <LinearAlgebra.hpp>
#include <_Common.hpp>
#include <_Index.hpp>

namespace psi {
namespace einhf {

SCF::SCF(SharedWavefunction ref_wfn, Options &options) : Wavefunction(options) {

  // Shallow copy useful objects from the passed in wavefunction
  shallow_copy(ref_wfn);

  nirrep_ = sobasisset_->nirrep();
  nso_ = basisset_->nbf();

  print_ = options_.get_int("PRINT");
  maxiter_ = options_.get_int("SCF_MAXITER");
  e_convergence_ = options_.get_double("E_CONVERGENCE");
  d_convergence_ = options_.get_double("D_CONVERGENCE");

  init_integrals();

  // Set Wavefunction matrices
  X_.set_name("S^1/2");
  F_.set_name("Fock Matrix");
  Ft_.set_name("Transformed Fock Matrix");
  C_.set_name("MO Coefficients");
  Cocc_.set_name("Occupied MO Coefficients");
  D_.set_name("Density Matrix");

  for (int i = 0; i < this->nirrep_; i++) {
    X_.push_block(einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                             irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    F_.push_block(einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                             irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    this->Ft_.push_block(einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), irrep_sizes_[i], irrep_sizes_[i]));
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    C_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    Cocc_.push_block(block);
  }

  for (int i = 0; i < this->nirrep_; i++) {
    auto block = einsums::Tensor<double, 2>(molecule_->irrep_labels().at(i),
                                            irrep_sizes_[i], irrep_sizes_[i]);

    D_.push_block(block);
  }
}

SCF::~SCF() {}

void SCF::init_integrals() {
  // The basisset object contains all of the basis information and is formed in
  // the new_wavefunction call The integral factory oversees the creation of
  // integral objects
  auto integral = std::make_shared<IntegralFactory>(basisset_, basisset_,
                                                    basisset_, basisset_);

  // Determine the number of electrons in the system
  // The molecule object is built into all wavefunctions
  int charge = molecule_->molecular_charge();
  int nelec = 0;
  for (int i = 0; i < molecule_->natom(); ++i) {
    nelec += (int)molecule_->Z(i);
  }
  nelec -= charge;
  if (nelec % 2) {
    throw PSIEXCEPTION("This is only an RHF code, but you gave it an odd "
                       "number of electrons.  Try again!");
  }
  ndocc_ = nelec / 2;

  outfile->Printf("    There are %d doubly occupied orbitals\n", ndocc_);
  molecule_->print();
  if (print_ > 1) {
    basisset_->print_detail();
    options_.print();
  }

  // nso_ = basisset_->nso();

  // Nuclear repulsion without a field
  e_nuc_ = molecule_->nuclear_repulsion_energy({0, 0, 0});
  outfile->Printf("\n    Nuclear repulsion energy: %16.8f\n\n", e_nuc_);

  // Make a MintsHelper object to help
  auto mints = std::make_shared<MintsHelper>(basisset_);

  // These don't need to be declared, because they belong to the class
  auto S_mat = mints->so_overlap();

  // Core Hamiltonian is Kinetic + Potential
  auto H_mat = mints->so_kinetic();

  this->nirrep_ = (int)S_mat->nirrep();

  if (this->nirrep_ == 1) {
    outfile->Printf("Hi");
  }

  occ_per_irrep_.resize(nirrep_);

  H_mat->add(mints->so_potential());

  H_.set_name("Hamiltonian");
  S_.set_name("Overlap");

  for (int i = 0; i < S_mat->nirrep(); i++) {
    irrep_sizes_.push_back(S_mat->rowdim(i));
    auto S_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), S_mat->rowdim(i), S_mat->coldim(i));

    for (int j = 0; j < S_mat->rowdim(i); j++) {
      for (int k = 0; k < S_mat->coldim(i); k++) {
        S_block(j, k) = S_mat->get(i, j, k);
      }
    }

    S_.push_block(S_block);
  }

  for (int i = 0; i < S_mat->nirrep(); i++) {
    auto H_block = einsums::Tensor<double, 2>(
        molecule_->irrep_labels().at(i), H_mat->rowdim(i), H_mat->coldim(i));

    for (int j = 0; j < H_mat->rowdim(i); j++) {
      for (int k = 0; k < H_mat->coldim(i); k++) {
        H_block(j, k) = H_mat->get(i, j, k);
      }
    }

    H_.push_block(H_block);
  }

  if (print_ > 3) {
    fprintln(*outfile->stream(), S_);
    fprintln(*outfile->stream(), H_);
    outfile->stream()->flush();
  }

  outfile->Printf("    Forming JK object\n\n");
  // Construct a JK object that compute J and K SCF matrices very efficiently
  jk_ = JK::build_JK(basisset_, std::shared_ptr<BasisSet>(), options_);

  // This is a very heavy compute object, lets give it 80% of our total memory
  jk_->set_memory(Process::environment.get_memory() * 0.8);
  jk_->initialize();
  jk_->print_header();
}

double SCF::compute_electronic_energy() {
  // Compute the electronic energy: (H + F)_pq * D_pq -> energy

  einsums::Tensor<double, 0> e_tens;
  auto temp = einsums::BlockTensor<double, 2>("temp", H_.vector_dims());

  temp = H_;
  temp += F_;

  einsums::tensor_algebra::einsum(
      0.0, einsums::tensor_algebra::Indices{}, &e_tens, 1.0,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      temp);

  return (double)e_tens;
}

void SCF::update_Cocc(const einsums::Tensor<double, 1> &energies) {
  // Update occupation.

  for (int i = 0; i < nirrep_; i++) {
    occ_per_irrep_[i] = 0;
  }

  for (int i = 0; i < ndocc_; i++) {
    double curr_min = INFINITY;
    int irrep_occ = -1;

    for (int j = 0; j < nirrep_; j++) {
      if (occ_per_irrep_[j] >= irrep_sizes_[j]) {
        continue;
      }

      double energy = energies(S_.block_range(j)[0] + occ_per_irrep_[j]);

      if (energy < curr_min) {
        curr_min = energy;
        irrep_occ = j;
      }
    }

    occ_per_irrep_[irrep_occ]++;
  }

  Cocc_.zero();

#pragma omp taskloop
  for (int i = 0; i < nirrep_; i++) {
#pragma omp parallel for
    for (int j = 0; j < Cocc_[i].dim(0); j++) {
#pragma omp parallel for
      for (int k = 0; k < occ_per_irrep_[i]; k++) {
        Cocc_[i](j, k) = C_[i](j, k);
      }
    }
  }
#pragma omp taskgroup
  ;
}

double SCF::compute_energy() {

  // Allocate a few temporary matrices
  auto Temp1 =
      einsums::BlockTensor<double, 2>("Temporary Array 1", irrep_sizes_);
  auto Temp2 =
      einsums::BlockTensor<double, 2>("Temporary Array 2", irrep_sizes_);
  auto FDS = einsums::BlockTensor<double, 2>("FDS", irrep_sizes_);
  auto SDF = einsums::BlockTensor<double, 2>("SDF", irrep_sizes_);
  auto Evecs = einsums::BlockTensor<double, 2>("Eigenvectors", irrep_sizes_);
  auto Evals = einsums::Tensor<double, 1>("Eigenvalues", nso_);

  // Form the X_ matrix (S^-1/2)
  X_ = einsums::linear_algebra::pow(S_, -0.5);

  F_ = H_;

  einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, &Temp1);
  einsums::linear_algebra::gemm<true, false>(1.0, X_, Temp1, 0.0, &Ft_);

  for (int i = 0; i < Ft_.num_blocks(); i++) {
    outfile->Printf("Range %d: %d - %d\n", i, Ft_.block_range(i)[0],
                    Ft_.block_range(i)[1]);
  }

  Evecs = Ft_;

  for (int i = 0; i < Evecs.num_blocks(); i++) {
    outfile->Printf("Range %d: %d - %d\n", i, Evecs.block_range(i)[0],
                    Evecs.block_range(i)[1]);
  }

  fprintln(*outfile->stream(), Evecs);
  outfile->stream()->flush();

  Evals.zero();
  einsums::linear_algebra::syev(&Evecs, &Evals);

  einsums::linear_algebra::gemm<false, true>(1.0, X_, Evecs, 0.0, &C_);

  update_Cocc(Evals);

  einsums::tensor_algebra::einsum(
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::j},
      &D_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                       einsums::tensor_algebra::index::m},
      Cocc_,
      einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                       einsums::tensor_algebra::index::m},
      Cocc_);

  if (print_ > 1) {
    outfile->Printf(
        "MO Coefficients and density from Core Hamiltonian guess:\n");
    fprintln(*outfile->stream(), X_);
    fprintln(*outfile->stream(), C_);
    fprintln(*outfile->stream(), D_);
    fprintln(*outfile->stream(), Evals);
    fprintln(*outfile->stream(), Cocc_);
  }

  int iter = 1;
  bool converged = false;
  double e_old;
  double e_new = e_nuc_ + compute_electronic_energy();

  outfile->Printf("    Energy from core Hamiltonian guess: %20.16f\n\n", e_new);

  outfile->Printf(
      "    *=======================================================*\n");
  outfile->Printf(
      "    * Iter       Energy            delta E    ||gradient||  *\n");
  outfile->Printf(
      "    *-------------------------------------------------------*\n");

  while (!converged && iter < maxiter_) {
    e_old = e_new;

    // Add the core Hamiltonian term to the Fock operator
    F_ = H_;

    // The JK object handles all of the two electron integrals
    // To enhance efficiency it does use the density, but the orbitals
    // themselves D_uv = C_ui C_vj J_uv = I_uvrs D_rs K_uv = I_urvs D_rs

    // Here we clear the old Cocc and push_back our new one
    std::vector<SharedMatrix> &Cl = jk_->C_left();
    Cl.clear();

    SharedMatrix CTemp = std::make_shared<Matrix>(nirrep_, irrep_sizes_.data(), irrep_sizes_.data());

    for (int i = 0; i < nirrep_; i++) {
      if(irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < occ_per_irrep_[i]; k++) {
          (*CTemp.get())(i, j, k) = Cocc_[i](j, k);
        }
      }
    }

    Cl.push_back(CTemp);
    jk_->compute();

    // Obtain the new J and K matrices
    const std::vector<SharedMatrix> &J_mat = jk_->J();
    const std::vector<SharedMatrix> &K_mat = jk_->K();

    // Proceede as normal
    auto J = einsums::BlockTensor<double, 2>("J matrix", irrep_sizes_);
    auto K = einsums::BlockTensor<double, 2>("K matrix", irrep_sizes_);

    for (int i = 0; i < nirrep_; i++) {
      if(irrep_sizes_[i] == 0) {
        continue;
      }
      for (int j = 0; j < irrep_sizes_[i]; j++) {
        for (int k = 0; k < irrep_sizes_[i]; k++) {
          J[i](j, k) = 2 * J_mat[0]->get(i, j, k);
          K[i](j, k) = K_mat[0]->get(i, j, k);
        }
      }
    }

    J *= 2.0;

    F_ += J;
    F_ -= K;

    // Compute the energy
    e_new = e_nuc_ + compute_electronic_energy();
    double dE = e_new - e_old;

    // Compute the orbital gradient, FDS-SDF
    einsums::linear_algebra::gemm<false, false>(1.0, D_, S_, 0.0, &Temp1);
    einsums::linear_algebra::gemm<false, false>(1.0, F_, Temp1, 0.0, &FDS);
    einsums::linear_algebra::gemm<false, false>(1.0, D_, F_, 0.0, &Temp1);
    einsums::linear_algebra::gemm<false, false>(1.0, S_, Temp1, 0.0, &SDF);

    Temp1 = FDS;
    Temp1 -= SDF;

    // Density RMS
    einsums::Tensor<double, 0> dRMS_tens;

    einsums::tensor_algebra::einsum(
        0.0, einsums::tensor_algebra::Indices{}, &dRMS_tens,
        1.0 / (nso_ * nso_),
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        Temp1,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        Temp1);

    double dRMS = std::sqrt(dRMS_tens);

    // Optional printing
    if (print_ > 1) {
      fprintln(*outfile->stream(), Ft_);
      fprintln(*outfile->stream(), Evecs);
      fprintln(*outfile->stream(), Evals);
      fprintln(*outfile->stream(), C_);
      fprintln(*outfile->stream(), D_);
      fprintln(*outfile->stream(), FDS);
      fprintln(*outfile->stream(), SDF);
      Temp1.set_name("Orbital Gradient");
      fprintln(*outfile->stream(), Temp1);
    }

    converged = (fabs(dE) < e_convergence_) && (dRMS < d_convergence_);

    outfile->Printf("    * %3d %20.14f    %9.2e    %9.2e    *\n", iter, e_new,
                    dE, dRMS);

    einsums::linear_algebra::gemm<false, false>(1.0, F_, X_, 0.0, &Temp1);
    einsums::linear_algebra::gemm<true, false>(1.0, X_, Temp1, 0.0, &Ft_);

    Evecs = Ft_;
    einsums::linear_algebra::syev(&Evecs, &Evals);

    einsums::linear_algebra::gemm<false, true>(1.0, X_, Evecs, 0.0, &C_);

    update_Cocc(Evals);

    einsums::tensor_algebra::einsum(
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::j},
        &D_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::i,
                                         einsums::tensor_algebra::index::m},
        Cocc_,
        einsums::tensor_algebra::Indices{einsums::tensor_algebra::index::j,
                                         einsums::tensor_algebra::index::m},
        Cocc_);
    iter++;
  }
  outfile->Printf(
      "    *=======================================================*\n");

  if (!converged)
    throw PSIEXCEPTION("The SCF iterations did not converge.");

  Evals.set_name("Orbital Energies");
  outfile->Printf("Occupied:\n");
  fprintln(*outfile->stream(), Evals(einsums::Range{0, ndocc_}));
  outfile->Printf("Unoccupied:\n");
  fprintln(*outfile->stream(), Evals(einsums::Range{ndocc_, nso_}));
  energy_ = e_new;

  return e_new;
}
} // namespace einhf
} // namespace psi
