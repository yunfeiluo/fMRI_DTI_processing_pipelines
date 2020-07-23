import os

# warped software packages
import nipype.interfaces.io as nio
import nipype.interfaces.spm as spm
import nipype.pipeline.engine as pe

class pipeline:
    def __init__(self, experiment_dir, output_dir, working_dir, func_source, struct_source, datasink):
        self.experiment_dir = experiment_dir
        self.output_dir = output_dir
        self.working_dir = working_dir

        # specify input and output nodes
        self.func_source = func_source
        self.struct_source = struct_source
        self.datasink = datasink

        # specify workflow instance
        self.workflow = pe.Workflow(name='workflow')

        # specify nodes
        self.realign = pe.Node(interface=spm.Realign(), name='realign')

        self.coregister = pe.Node(interface=spm.Coregister(), name="coregister")
        self.coregister.inputs.jobtype = 'estimate'
        
        self.segment = pe.Node(interface=spm.Segment(), name="segment")

        self.normalize_func = pe.Node(interface=spm.Normalize(), name="normalize_func")
        self.normalize_func.inputs.jobtype = "write"

        self.normalize_struc = pe.Node(interface=spm.Normalize(), name="normalize_struc")
        self.normalize_struc.inputs.jobtype = "write"

        self.smooth = pe.Node(interface=spm.Smooth(), name="smooth")        

        # connect the nodes to complete the workflow
        self.workflow.connect([
            (self.func_source, self.realign, [('outfiles', 'in_files')]),
            (self.struct_source, self.coregister, [('outfiles', 'source')]),
            (self.realign, self.coregister, [('mean_image', 'target')]),
            (self.coregister, self.segment, [('coregistered_source', 'data')]),
            (self.segment, self.normalize_func, [('transformation_mat', 'parameter_file')]),
            (self.realign, self.normalize_func, [('realigned_files', 'apply_to_files')]),
            (self.normalize_func, self.smooth, [('normalized_files', 'in_files')]),
            #(self.realign, self.datasink, [('realigned_files', 'realign')]),
            #(self.realign, self.datasink, [('mean_image', 'mean')]),
            (self.normalize_func, self.datasink, [('normalized_files', 'norm')]),
            (self.smooth, self.datasink, [('smoothed_files', 'smooth')])
        ])

    # Execute the node(s)
    def forward(self, subject):
        self.workflow.run()
