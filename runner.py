import os
import sys
import json
import argparse
import time


import uproot
from coffea.util import load, save
from coffea import processor
from test_processor import NanoProcessor



if __name__ == "__main__":

    # Execute
    sample_dict={
        ## specify data(no correction)
        "data":["root://xrootd-cms.infn.it///store/data/Run2017B/MuonEG/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/280000/E0BC21C5-3494-F64E-B419-4449D3F1FDB7.root"],
        # "mc":["root://xrootd-cms.infn.it///store/mc/RunIISummer20UL17NanoAODv9/GluGluHToWWTo2L2Nu_M-125_TuneCP5_13TeV-powheg-jhugen727-pythia8/NANOAODSIM/106X_mc2017_realistic_v9-v2/40000/702ED98F-6108-9945-BAD3-4F4DF1682C4E.root"]
    }
      
    output = processor.run_uproot_job(
            sample_dict,
            treename="Events",
            processor_instance=NanoProcessor(),
            executor=processor.iterative_executor,
            executor_args={
                #"skipbadfiles": args.skipbadfiles,
                "schema": processor.NanoAODSchema,
                #"workers": args.workers,
            },
            chunksize=10000,
            #maxchunks=args.max,
        )
   
    save(output, args.output)

    print(output)
