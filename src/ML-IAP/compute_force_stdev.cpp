// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_force_stdev.h"
#include "compute.h"
#include "atom.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair_mliap.h"
using namespace LAMMPS_NS;


ComputeForceAtomStdev::ComputeForceAtomStdev(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), force_stdev(nullptr), data(nullptr)
{
  peratom_flag = 1;
  size_peratom_cols = 3;
  nmax = 0;
  //Check for args here

}

/* ---------------------------------------------------------------------- */

ComputeForceAtomStdev::~ComputeForceAtomStdev()
{
  memory->destroy(force_stdev);

  //data is managed by PairMLIAP so we should only set it to nullptr
  data = nullptr;
}

/* ---------------------------------------------------------------------- */

void ComputeForceAtomStdev::init()
{
  // Check if a pair style has been defined
  if (force->pair == nullptr)
    error->all(FLERR,"Compute force_stdev requires a pair style be defined");

  // Check if it is safe downcast to PairMLIAP pair style
  PairMLIAP* castedPair = dynamic_cast<PairMLIAP *>(force->pair);
  if (castedPair == nullptr)
    error->all(FLERR, "Compute force_stdev requires pair mliap to be active");  

  //Get the pointer
  data = castedPair->data;
}

/* ---------------------------------------------------------------------- */

void ComputeForceAtomStdev::compute_peratom()
{

  //Resize or create eatom_stdev array if needed
  if (force_stdev == nullptr) {
    nmax = atom->nmax;
    memory->create(force_stdev, nmax, 3, "compute_force_stdev:force_stdev");
    array_atom = force_stdev;
  } else if (atom->nmax > nmax) {
    nmax = atom->nmax;
    memory->grow(force_stdev, nmax, 3, "compute_force_stdev:force_stdev");
    array_atom = force_stdev;
  }

  //Check if eatom_stdev exists in data
  if (data->uqflag == 0 || data->force_stdev == nullptr) {
    error->all(FLERR,"Compute force_stdev requires pair mliap to have uncertainty quantification on");
  }

  //Copy the values in data to eatom_stdev
  for (int i = 0; i < atom->nlocal; i++) {
    force_stdev[i][0] = data->force_stdev[i][0];
    force_stdev[i][1] = data->force_stdev[i][1];
    force_stdev[i][2] = data->force_stdev[i][2];
  }
}

/* ---------------------------------------------------------------------- */

double ComputeForceAtomStdev::memory_usage()
{
  double bytes = nmax * 3 * sizeof(double); //force_stdev

  return bytes;
}
