import pickle, os, sys, numpy as np
from coffea import hist, processor
import awkward as ak
import hist as Hist
from coffea.analysis_tools import Weights
from functools import partial

import gc
import os,psutil
import gzip
import cloudpickle
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
def update(events, collections):
    """Return a shallow copy of events array with some collections swapped out"""
    out = events
    for name, value in collections.items():
        out = ak.with_field(out, value, name)
    return out

def add_jec_variables(jets, event_rho):
    jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
    jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
    if hasattr(jets, "genJetIdxG"):
        jets["pt_gen"] = ak.values_astype(
            ak.fill_none(jets.matched_gen.pt, 0), np.float32
        )
    jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
    return jets
def load_jmefactory(campaign, filename):
    with gzip.open(filename) as fin:
        jmestuff = cloudpickle.load(fin)
    jet_factory = jmestuff["jet_factory"]
    met_factory = jmestuff["met_factory"]
    return jet_factory,met_factory



class NanoProcessor(processor.ProcessorABC):
    # Define histograms
    def __init__(
        self#, year="2017", campaign="UL17", BDTversion="dymore", export_array=False, systematics= True,isData=True
    ):
        print("nothing : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")        
        # self._year = year
        # self.systematics = systematics    
        # self._campaign = campaign
        self._jet_factory,self._met_factory = load_jmefactory(
                "UL17", "mc_compile_jec.pkl.gz" 
            )
               
        
        self.make_output = lambda:{
                "cutflow": processor.defaultdict_accumulator(
                    partial(processor.defaultdict_accumulator, int)
                ),
                'sumw': processor.defaultdict_accumulator(float),
        }
    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        isRealData = not hasattr(events, "genWeight")
        if isRealData:

            jets =events.Jet
            met = events.MET
        else:
            jetfac_name = "mc"
            jets = self._jet_factory[jetfac_name].build(
            add_jec_variables(events.Jet, events.fixedGridRhoFastjetAll), lazy_cache=events.caches[0]
        )
            met = self._met_factory.build(events.MET, jets, {})
        
        shifts = [
            ({"Jet": jets, "MET": met}, None),
        ]
        
        if not isRealData:
        
            shifts += [
                (
                    {
                        "Jet": jets.JES_jes.up,
                        "MET": met.JES_jes.up,
                    },
                    "JESUp",
                ),
                (
                    {
                        "Jet": jets.JES_jes.down,
                        "MET": met.JES_jes.down,
                    },
                    "JESDown",
                ),
                (
                    {
                        "Jet": jets,
                        "MET": met.MET_UnclusteredEnergy.up,
                    },
                    "UESUp",
                ),
                (
                    {
                        "Jet": jets,
                        "MET": met.MET_UnclusteredEnergy.down,
                    },
                    "UESDown",
                ),
                (
                    {
                        "Jet": jets.JER.up,
                        "MET": met.JER.up,
                        
                    },
                    "JERUp",
                ),
                (
                    {
                        "Jet": jets.JER.down,
                        "MET": met.JER.down,
                        
                    },
                    "JERDown",
                ),
            ]
        

        
        print("load : ",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")
        return processor.accumulate(
            self.process_shift(update(events, collections), name)
            for collections, name in shifts
        )

    def process_shift(self, events,shift_name):
        print(shift_name)

        output = self.make_output()
        dataset = events.metadata["dataset"]
        isRealData = not hasattr(events, "genWeight")
        selection = processor.PackedSelection()
        if isRealData:
            output["sumw"][dataset] += len(events)
        else:
            output["sumw"][dataset] += ak.sum(events.genWeight / abs(events.genWeight))
        
        ## Muon cuts
        # muon twiki: https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideMuonIdRun2
        event_mu = events.Muon
        musel = (
            (event_mu.pt > 13)
            & (abs(event_mu.eta) < 2.4)
            & (event_mu.mvaId >= 3)
            & (event_mu.pfRelIso04_all < 0.15)
            & (abs(event_mu.dxy) < 0.05)
            & (abs(event_mu.dz) < 0.1)
        )
        event_mu["lep_flav"] = 13 * event_mu.charge
        event_mu = event_mu[ak.argsort(event_mu.pt, axis=1, ascending=False)]
        event_mu = event_mu[musel]
        event_mu = ak.pad_none(event_mu, 2, axis=1)
        nmu = ak.sum(musel, axis=1)
        amu = events.Muon[
            (events.Muon.pt > 10)
            & (abs(events.Muon.eta) < 2.4)
            & (events.Muon.pfRelIso04_all < 0.25)
            & (events.Muon.mvaId >= 1)
        ]
        namu = ak.count(amu.pt, axis=1)
        # ## Electron cuts
        # # electron twiki: https://twiki.cern.ch/twiki/bin/viewauth/CMS/CutBasedElectronIdentificationRun2
        event_e = events.Electron
        event_e["lep_flav"] = 11 * event_e.charge
        elesel = (
            (event_e.pt > 13)
            & (abs(event_e.eta) < 2.5)
            & (event_e.mvaFall17V2Iso_WP90 == 1)
            & (abs(event_e.dxy) < 0.05)
            & (abs(event_e.dz) < 0.1)
        )
        event_e = event_e[elesel]
        event_e = event_e[ak.argsort(event_e.pt, axis=1, ascending=False)]
        event_e = ak.pad_none(event_e, 2, axis=1)
        nele = ak.sum(elesel, axis=1)
        aele = events.Electron[
            (events.Electron.pt > 12)
            & (abs(events.Electron.eta) < 2.5)
            & (events.Electron.mvaFall17V2Iso_WPL == 1)
        ]
        naele = ak.count(aele.pt, axis=1)

        selection.add("lepsel", ak.to_numpy((nele + nmu >= 2)))


        good_leptons = ak.with_name(
            ak.concatenate([event_e, event_mu], axis=1),
            "PtEtaPhiMCandidate",
        )
        del event_e, event_mu
        good_leptons = good_leptons[
            ak.argsort(good_leptons.pt, axis=1, ascending=False)
        ]
        
        leppair = ak.combinations(
            good_leptons,
            n=2,
            replacement=False,
            axis=-1,
            fields=["lep1", "lep2"],
        )
        
        del good_leptons
        ll_cand = ak.zip(
            {
                "lep1": leppair.lep1,
                "lep2": leppair.lep2,
                "pt": (leppair.lep1 + leppair.lep2).pt,
                "eta": (leppair.lep1 + leppair.lep2).eta,
                "phi": (leppair.lep1 + leppair.lep2).phi,
                "mass": (leppair.lep1 + leppair.lep2).mass,
            },
            with_name="PtEtaPhiMLorentzVector",
        )
        del leppair
        ll_cand = ak.pad_none(ll_cand, 1, axis=1)
        ll_cand = ak.packed(ll_cand)
        # other = event_e.nearest(events.Jet)
        print("before nearest:",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")      
        events_jet = events.Jet
        events_jet =ak.pad_none(events_jet,1,axis=1)
        ### Place to increase 
        topjet1cut=ll_cand[:,0].lep1.nearest(events.Jet)
        topjet2cut=ll_cand[:,0].lep2.nearest(events.Jet)

        other = events_jet[:,0].nearest(events.Jet)
        
        print("after nearest:",psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, "MB")        
        
        return {dataset:output}

    def postprocess(self, accumulator):
        return accumulator