import glob
import sys
import os
import fnmatch
import numpy as np
#from __future__ import print_function
#from future import standard_library
# from tkinter import *
# from tkinter.filedialog import askopenfilename
# from nipype.interfaces.ants import Registration, ApplyTransforms
from nipype.interfaces import fsl
import nibabel as nib

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

class tractography:
    def __init__(
        self, 
        subject, 
        dti_source, 
        mask_source, 
        bval_source, 
        bvec_source, 
        seed_source, 
        datasink
    ):

        self.subject = subject
        self.bval_source = bval_source
        self.bvec_source = bvec_source
        self.dti_source = dti_source
        self.mask_source = mask_source
        self.seed_source = seed_source
        self.datasink = datasink

        ##############################################################################
        # declare nodes
        ##############################################################################
        self.bedp = pe.Node(interface=fsl.BEDPOSTX5(), name="bedpostx")
        self.bedp.inputs.n_fibres = 2 # Maximum number of fibres to fit in each voxel
        self.bedp.inputs.non_linear = True

        self.pbx = pe.Node(interface=fsl.ProbTrackX(), name='ProbTrackX')
        self.pbx.inputs.mode = 'seedmask'
        self.pbx.inputs.n_samples = 500
        self.pbx.inputs.opd = True
        self.pbx.inputs.os2t = True

        # ======================= SPLIT LINE =========================================

        ##############################################################################
        # declare workflow, and connect nodes
        ##############################################################################
        self.workflow = pe.Workflow(name='tractography_workflow')
        self.workflow.connect([
            (self.bval_source, self.bedp, [('outfiles', 'bvals')]),
            (self.bvec_source, self.bedp, [('outfiles', 'bvecs')]),
            (self.dti_source, self.bedp, [('outfiles', 'dwi')]),
            (self.mask_source, self.bedp, [('outfiles', 'mask')]),
            (self.mask_source, self.pbx, [('outfiles', 'mask')]),
            (self.bedp, self.pbx, [
                ('merged_fsamples', 'fsamples'),
                ('merged_phsamples', 'phsamples'),
                ('merged_thsamples', 'thsamples') 
            ]),
            (self.seed_source, self.pbx, [('outfiles', 'seed')]),
            (self.pbx, self.datasink, [
                ('fdt_paths', 'streamline_density_map'),
                ('way_total', 'way_total')
            ])
        ])
    
    # Execute the node(s)
    def forward(self):
        self.workflow.run()
