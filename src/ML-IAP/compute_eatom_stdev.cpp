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

#include "compute_eatom_stdev.h"
#include "compute.h"
#include "atom.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "pair_mliap.h"
using namespace LAMMPS_NS;


ComputeEatomStdev::ComputeEatomStdev(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), eatom_stdev(nullptr), data(nullptr)
{
  peratom_flag = 1;
  size_peratom_cols = 0;
  nmax = 0;
  //Check for args here

}

/* ---------------------------------------------------------------------- */

ComputeEatomStdev::~ComputeEatomStdev()
{
  memory->destroy(eatom_stdev);

  //data is managed by PairMLIAP so we should only set it to nullptr
  data = nullptr;
}

/* ---------------------------------------------------------------------- */

void ComputeEatomStdev::init()
{
  // Check if a pair style has been defined
  if (force->pair == nullptr)
    error->all(FLERR,"Compute eatom_stdev requires a pair style be defined");

  // Check if it is safe downcast to PairMLIAP pair style
  PairMLIAP* castedPair = dynamic_cast<PairMLIAP *>(force->pair);
  if (castedPair == nullptr)
    error->all(FLERR, "Compute eatom_stdev requires pair mliap to be active");  

  //Get the pointer
  data = castedPair->data;
}

/* ---------------------------------------------------------------------- */

void ComputeEatomStdev::compute_peratom()
{

  //Resize or create eatom_stdev array if needed
  if (eatom_stdev == nullptr || atom->nmax > nmax) {
    if (eatom_stdev) {
      memory->destroy(eatom_stdev);
    }
    nmax = atom->nmax;
    memory->create(eatom_stdev, nmax, "compute_eatom_stdev:eatom_stdev");
    vector_atom = eatom_stdev;
  }

  //Check if eatom_stdev exists in data
  if (data->uqflag == 0 || data->eatoms_stdev == nullptr) {
    error->all(FLERR,"Compute eatom_stdev requires pair mliap to have uncertainty quantification on");
  }

  //Copy the values in data to eatom_stdev
  for (int i = 0; i < atom->nlocal; i++) {
    eatom_stdev[i] = data->eatoms_stdev[i];
  }
}

/* ---------------------------------------------------------------------- */

double ComputeEatomStdev::memory_usage()
{
  double bytes = nmax * sizeof(double); //eatom_stdev

  return bytes;
}
