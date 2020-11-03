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
from nipype.interfaces import ants, fsl
import nibabel as nib

import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

class ants_registr_FA:
    def __init__(self, subject, fa_source, datasink):
        self.subject = subject
        self.fa_source = fa_source
        self.datasink = datasink

        self.bet = pe.Node(interface=fsl.BET(), name="bet_fa")
        self.bet.inputs.mask = True
        self.bet.inputs.output_type = "NIFTI_GZ"
        self.bet.inputs.frac = 0.3
        ##############################################################################
        # declare node(s) for Registration
        ##############################################################################
        self.reg = pe.Node(interface=ants.Registration(), name="dti_fa_registration")
        self.reg.inputs.fixed_image = '/mnt/Program/python/dti/imgs/FMRIB58_FA_1mm.nii'
        # self.reg.inputs.transforms = ['Translation', 'Rigid', 'Affine']
        self.reg.inputs.transforms = ['Affine','SyN']
        # self.reg.inputs.transforms = ['Translation','Affine','SyN']
        self.reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1,), (0.2, 3.0, 0.0)]
        self.reg.inputs.number_of_iterations = ([[10000, 111110, 11110]] * 3 +
                                                [[100, 50, 30]])
        self.reg.inputs.dimension = 3
        self.reg.inputs.write_composite_transform = True
        self.reg.inputs.collapse_output_transforms = False
        self.reg.inputs.metric = ['Mattes'] * 3 + [['Mattes', 'CC']]
        self.reg.inputs.metric_weight = [1] * 3 + [[0.5, 0.5]]
        self.reg.inputs.radius_or_number_of_bins = [32] * 3 + [[32, 4]]
        self.reg.inputs.sampling_strategy = ['Regular'] * 3 + [[None, None]]
        self.reg.inputs.sampling_percentage = [0.3] * 3 + [[None, None]]
        self.reg.inputs.convergence_threshold = [1.e-8] * 3 + [-0.01]
        self.reg.inputs.convergence_window_size = [20] * 3 + [5]
        self.reg.inputs.smoothing_sigmas = [[4, 2, 1]] * 3 + [[1, 0.5, 0]]
        self.reg.inputs.sigma_units = ['vox'] * 4
        self.reg.inputs.shrink_factors = [[6, 4, 2]] + [[3, 2, 1]] * 2 + [[4, 2, 1]]
        self.reg.inputs.use_estimate_learning_rate_once = [True] * 4
        self.reg.inputs.use_histogram_matching = [False] * 3 + [True]
        self.reg.inputs.initial_moving_transform_com = True
        # self.reg.inputs.verbose = True
        self.reg.inputs.num_threads  = 4
        self.reg.inputs.output_warped_image = True      
        # self.reg.inputs.output_transform_prefix = 'wASyN_' + sub_names[x] + "_transform_"

        ##############################################################################
        # declare workflow, and connect nodes
        ##############################################################################
        self.workflow = pe.Workflow(name='preprocess_workflow')
        self.workflow.connect([
            (self.fa_source, self.bet, [('outfiles', 'in_file')]),
            # (self.bet, self.reg, [('out_file', 'moving_image')]),
            # (self.reg, self.datasink, [('warped_image', 'wASyN')]),
            (self.bet, self.datasink, [('out_file', 'bet_img'), ('mask_file', 'bet_brain_mask')])
        ])
    
    # Execute the node(s)
    def forward(self):
        self.workflow.run()

# STOP
# -----------------------------------------------------------------------------
## Apply transform parameters to  other images in same space
# from nipype.interfaces.ants import ApplyTransforms
#
# raw_img = glob.glob('DWI_raw/*.nii')
# nii_trace = glob.glob('DWI_wASyN/*.nii')
# mat_trace = glob.glob('DWI_wASyN/*.mat')
# comp_trace = glob.glob('DWI_wASyN/*Composite.h5')
#
# nii_FA = glob.glob('FA/*.nii')
#
# for ind_trans in range(0, len(nii_trace)):
#
#     ind_trans = 0
#     tr = ApplyTransforms()
#     tr.inputs.reference_image = template
#     # tr.inputs.reference_image = raw_img[ind_trans]
#     # tr.inputs.reference_image = nii_trace[ind_trans]
#     tr.inputs.input_image = nii_FA[ind_trans]
#     assert(nii_trace[ind_trans].split('wASyN_')[1][0:4] ==
#         nii_FA[ind_trans].split('/')[1][0:4])
#
#     tr.inputs.transforms = [mat_trace[ind_trans], comp_trace[ind_trans]]
#     tr.inputs.invert_transform_flags = [False, False]
#     tr.inputs.interpolation = 'Linear' # Chose what fits better to you here
#     # tr.inputs.interpolation = 'NearestNeighbor' # Chose what fits better to you here
#     tr.inputs.output_image= 'transforms/' + \
#         nii_trace[ind_trans].split('wASyN_')[1][0:4] + '_transformed_FA.nii'
#     tr.run()
