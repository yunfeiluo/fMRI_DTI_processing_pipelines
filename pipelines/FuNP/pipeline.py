import os

# warped software packages
import nipype.interfaces.io as nio
import nipype.interfaces.spm as spm
from nipype.interfaces import afni as afni
from nipype.interfaces import fsl

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function

class pipeline:
    def __init__(self, experiment_dir, output_dir, working_dir, func_source, struct_source, datasink):
        self.experiment_dir = experiment_dir
        self.output_dir = output_dir
        self.working_dir = working_dir

        # specify input and output nodes
        self.func_source = func_source
        self.struct_source = struct_source
        self.datasink = datasink

        # specify nodes
        # structual process
        self.refit_struct = pe.Node(interface=afni.Refit(), name='de_oblique_struct')
        self.refit_struct.inputs.deoblique = True

        self.resample_struct = pe.Node(interface=afni.Resample(), name='re_orientation_struct')
        self.resample_struct.inputs.orientation= 'RPI'
        self.resample_struct.inputs.outputtype = "NIFTI"

        self.bet_struct = pe.Node(interface=fsl.BET(), name='non_brain_removal_BET_struct')
        self.bet_struct.inputs.output_type = "NIFTI"

        # functional process
        self.refit_func = pe.Node(interface=afni.Refit(), name='de_oblique_func')
        self.refit_func.inputs.deoblique = True

        self.resample_func = pe.Node(interface=afni.Resample(), name='re_orientation_func')
        self.resample_func.inputs.orientation= 'RPI'
        self.resample_func.inputs.outputtype = "NIFTI"

        self.slice_timer = pe.Node(interface=fsl.SliceTimer(), name='time_slice_correction')

        self.mcflirt = pe.Node(interface=fsl.MCFLIRT(), name='motion_correction')
        self.mcflirt.inputs.output_type = "NIFTI"
        self.mcflirt.inputs.mean_vol = True

        self.fslsplit = pe.Node(interface=fsl.Split(), name='fslsplit')
        self.fslsplit.inputs.dimension = 't'
        self.fslsplit.inputs.output_type = "NIFTI"

        self.fslmerge = pe.Node(interface=fsl.Merge(), name='fslmerge')
        self.fslmerge.inputs.dimension = 't'
        self.fslmerge.inputs.output_type = "NIFTI"

        self.bet_mean = pe.Node(interface=fsl.BET(), name='non_brain_removal_BET_mean')
        self.bet_mean.inputs.output_type = "NIFTI"
        
        # helper function(s)
        def bet_each(in_files):
            '''
            @param in_files: list of image files
            @return out_files: list of image files after applied fsl.BET on it
            '''
            from nipype.interfaces import fsl
            import nipype.pipeline.engine as pe

            out_files = list()
            step_no = 0
            for file_ in in_files:
                bet = pe.Node(interface=fsl.BET(), name='BET_for_step_{}'.format(step_no))
                bet.inputs.in_file = file_
                bet.inputs.out_file = file_[:len(file_) - 4] + '_bet.nii'
                bet.inputs.output_type = "NIFTI"

                bet.run()
                out_files.append(bet.inputs.out_file)

                step_no += 1
            return out_files
        # bet_func return a list of NIFITI files
        self.bet_func = pe.Node(interface=Function(input_names=['in_files'], output_names=['out_files'], function=bet_each), name='non_brain_removal_BET_func')

        self.coregister = pe.Node(interface=spm.Coregister(), name="coregister")
        self.coregister.inputs.jobtype = 'estimate'

        self.segment = pe.Node(interface=spm.Segment(), name="segment")

        self.normalize = pe.Node(interface=spm.Normalize(), name="normalize")
        self.normalize.inputs.jobtype = "write"

        # self.fourier = pe.Node(interface=afni.Fourier(), name='temporal_filtering')
        # self.fourier.inputs.highpass = 0.01
        # self.fourier.inputs.lowpass = 0.1

        self.smooth = pe.Node(interface=spm.Smooth(), name="smooth")
        self.smooth.inputs.fwhm = [8, 8, 8]

        # specify workflow instance
        self.workflow = pe.Workflow(name='FuNP_workflow')

        # connect nodes
        self.workflow.connect([
            (self.struct_source, self.refit_struct, [('outfiles', 'in_file')]),
            (self.refit_struct, self.resample_struct, [('out_file', 'in_file')]),
            (self.resample_struct, self.bet_struct, [('out_file', 'in_file')]),
            #(self.func_source, self.refit_func, [('outfiles', 'in_file')]),
            #(self.refit_func, self.resample_func, [('out_file', 'in_file')]),
            #(self.resample_func, self.slice_timer, [('out_file', 'in_file')]),
            (self.func_source, self.slice_timer, [('outfiles', 'in_file')]),
            (self.slice_timer, self.mcflirt, [('slice_time_corrected_file', 'in_file')]),
            (self.mcflirt, self.bet_mean, [('mean_img', 'in_file')]),
            (self.mcflirt, self.fslsplit, [('out_file', 'in_file')]),
            (self.fslsplit, self.bet_func, [('out_files', 'in_files')]),
            (self.bet_func, self.fslmerge, [('out_files', 'in_files')]),# intersect
            (self.bet_struct, self.coregister, [('out_file', 'source')]),
            (self.bet_mean, self.coregister, [('out_file', 'target')]),
            (self.coregister, self.segment, [('coregistered_source', 'data')]),
            (self.segment, self.normalize, [('transformation_mat', 'parameter_file')]), 
            (self.fslmerge, self.normalize, [('merged_file', 'apply_to_files')]),
            (self.normalize, self.smooth, [('normalized_files', 'in_files')]),
            #(self.mcflirt, self.datasink, [('out_file', 'motion_corrected')]),
            (self.coregister, self.datasink, [('coregistered_source', 'registered_file')]),
            (self.normalize, self.datasink, [('normalized_files', 'before_smooth')]),
            (self.smooth, self.datasink, [('smoothed_files', 'final_out')])
        ])

    # Execute the node(s)
    def forward(self, subject):
        self.workflow.run()
    
