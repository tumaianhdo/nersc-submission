#!/usr/bin/env python3
import os
import sys
import logging
from pathlib import Path
from argparse import ArgumentParser

logging.basicConfig(level=logging.DEBUG)

# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

class DiamondWorkflow():
    wf = None
    sc = None
    tc = None
    rc = None
    props = None

    dagfile = None
    wf_name = None
    wf_dir = None
    
    # --- Init ---------------------------------------------------------------------
    def __init__(self, temperatures, dagfile="workflow.yml"):
        self.dagfile = dagfile
        self.wf_name = "namd_wf"
        self.temperatures = temperatures
        self.wf_dir = Path(__file__).parent.resolve()

    
    # --- Write files in directory -------------------------------------------------
    def write(self):
        self.props.write()
        self.sc.write()
        self.rc.write()
        self.tc.write()
        self.wf.write()


    # --- Configuration (Pegasus Properties) ---------------------------------------
    def create_pegasus_properties(self):
        self.props = Properties()

        #self.props["pegasus.transfer.threads"] = "1"
        #self.props["pegasus.transfer.worker.package"] = "false"
        #self.props["pegasus.stagein.remote.clusters"] = "1"
        #self.props["pegasus.transfer.*.remote.sites"] = "*"

        #self.props["pegasus.condor.arguments.quote"] = "false"
        
        return


    # --- Site Catalog -------------------------------------------------------------
    def create_sites_catalog(self):
        self.sc = SiteCatalog()

        shared_scratch_dir = os.path.join(self.wf_dir, "scratch")
        local_storage_dir = os.path.join(self.wf_dir, "output")

        local = Site("local")\
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, shared_scratch_dir)
                            .add_file_servers(FileServer("file://" + shared_scratch_dir, Operation.ALL)),
                        Directory(Directory.LOCAL_STORAGE, local_storage_dir)
                            .add_file_servers(FileServer("file://" + local_storage_dir, Operation.ALL))
                    )

        cori = Site("cori")\
                    .add_grids(
                        Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${NERSC_USER}@cori.nersc.gov", job_type=SupportedJobs.COMPUTE),
                        Grid(grid_type=Grid.BATCH, scheduler_type=Scheduler.SLURM, contact="${NERSC_USER}@cori.nersc.gov", job_type=SupportedJobs.AUXILLARY)
                    )\
                    .add_directories(
                        Directory(Directory.SHARED_SCRATCH, "/global/cscratch1/sd/${NERSC_USER}/pegasus/scratch")
                            .add_file_servers(FileServer("file:///global/cscratch1/sd/${NERSC_USER}/pegasus/scratch", Operation.ALL)),
                        Directory(Directory.SHARED_STORAGE, "/global/cscratch1/sd/${NERSC_USER}/pegasus/storage")
                            .add_file_servers(FileServer("file:///global/cscratch1/sd/${NERSC_USER}/pegasus/storage", Operation.ALL))
                    )\
                    .add_pegasus_profile(
                        style="ssh",
                        data_configuration="sharedfs",
                        change_dir="true",
                        project="${NERSC_PROJECT}",
                        runtime=300
                    )\
                    .add_env(key="PEGASUS_HOME", value="${NERSC_PEGASUS_HOME}")
        
        self.sc.add_sites(local, cori)
                        


    # --- Transformation Catalog (Executables and Containers) ----------------------
    def create_transformation_catalog(self):
        self.tc = TransformationCatalog()

        # Add pegasus functions to be run on the xfer queue
        pegasus_transfer = Transformation("transfer", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
                                .add_pegasus_profile(
                                    queue="@escori",
                                    runtime="300",
                                    glite_arguments="--qos xfer --licenses=SCRATCH"
                                )

        pegasus_dirmanager = Transformation("dirmanager", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
                                .add_pegasus_profile(
                                    queue="@escori",
                                    runtime="300",
                                    glite_arguments="--qos xfer --licenses=SCRATCH"
                                )

        pegasus_cleanup = Transformation("cleanup", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
                                .add_pegasus_profile(
                                    queue="@escori",
                                    runtime="300",
                                    glite_arguments="--qos xfer --licenses=SCRATCH"
                                )
        
        system_chmod = Transformation("chmod", namespace="system", site="cori", pfn="/usr/bin/chmod", is_stageable=False)\
                                .add_pegasus_profile(
                                    queue="@escori",
                                    runtime="60",
                                    glite_arguments="--qos xfer --licenses=SCRATCH"
                                )

        # Add the namd executable
        namd = Transformation("namd", site="cori", pfn="https://raw.githubusercontent.com/pegasus-isi/pegasus-nersc-bosco/main/data/workflows/sns-namd-example/bin/namd_wrapper.sh", is_stageable=True)\
                    .add_pegasus_profile(
                        cores="32",
                        runtime="1200",
                        exitcode_success_msg="End of program",
                        glite_arguments="--qos debug --constraint=haswell --licenses=SCRATCH"
                    )\
                    .add_env(key="PEGASUS_CORES", value="32")
        
        self.tc.add_transformations(pegasus_transfer, pegasus_dirmanager, pegasus_cleanup, system_chmod, namd)


    # --- Replica Catalog ----------------------------------------------------------
    def create_replica_catalog(self):
        self.rc = ReplicaCatalog()

        github_location = "https://raw.githubusercontent.com/pegasus-isi/pegasus-nersc-bosco/main/data/workflows/sns-namd-example/input"
        self.rc.add_replica("GitHub", "Q42.psf", os.path.join(github_location, "Q42.psf"))
        self.rc.add_replica("GitHub", "crd.md18_vmd_autopsf.pdb", os.path.join(github_location, "crd.md18_vmd_autopsf.pdb"))
        self.rc.add_replica("GitHub", "par_all27_prot_lipid.inp", os.path.join(github_location, "par_all27_prot_lipid.inp"))
        self.rc.add_replica("GitHub", "init.xsc", os.path.join(github_location, "init.xsc"))
        for temperature in self.temperatures:
            self.rc.add_replica("GitHub", ("equilibrate_%s.conf" % temperature), os.path.join(github_location, ("equilibrate_%s.conf" % temperature)))
            self.rc.add_replica("GitHub", ("production_%s.conf" % temperature), os.path.join(github_location, ("production_%s.conf" % temperature)))

    
    # --- Create Workflow ----------------------------------------------------------
    def create_workflow(self):
        self.wf = Workflow(self.wf_name, infer_dependencies=True)
        
        structure = File("Q42.psf")
        coordinates = File("crd.md18_vmd_autopsf.pdb")
        parameters = File("par_all27_prot_lipid.inp")
        extended_system = File("init.xsc")
        
        for temperature in self.temperatures:
            eq_conf = File("equilibrate_%s.conf" % temperature)
            eq_coord = File("equilibrate_%s.restart.coor" % temperature)
            eq_xsc = File("equilibrate_%s.restart.xsc" % temperature)
            eq_vel = File("equilibrate_%s.restart.vel" % temperature)

            # Add a equilibrate job
            #equilibrate_job = Job("namd", node_label="namd_eq_%s" % temperature)\
            equilibrate_job = Job("namd")\
                            .add_args(eq_conf)\
                            .add_inputs(eq_conf, structure, coordinates, parameters, extended_system)\
                            .add_outputs(eq_coord, eq_xsc, eq_vel, stage_out=False, register_replica=False)

            prod_conf = File("production_%s.conf" % temperature)
            prod_dcd = File("production_%s.dcd" % temperature)
            
            # Add a production job
            #production_job = Job("namd", node_label="namd_prod_%s" % temperature)\
            production_job = Job("namd")\
                            .add_args(prod_conf)\
                            .add_inputs(prod_conf, structure, coordinates, parameters, eq_coord, eq_xsc, eq_vel)\
                            .add_outputs(prod_dcd, stage_out=True, register_replica=True)

            self.wf.add_jobs(equilibrate_job, production_job)


if __name__ == '__main__':
    parser = ArgumentParser(description="Pegasus Diamond Workflow")

    parser.add_argument("-t", "--temperatures", metavar="INT", type=int, nargs="+", default=[200, 250], help="List of temperatures")
    parser.add_argument("-o", "--output", metavar="STR", type=str, default="workflow.yml", help="Output file (default: workflow.yml)")

    args = parser.parse_args()
    
    workflow = DiamondWorkflow(args.temperatures, args.output)
    
    print("Creating execution sites...")
    workflow.create_sites_catalog()

    print("Creating workflow properties...")
    workflow.create_pegasus_properties()
    
    print("Creating transformation catalog...")
    workflow.create_transformation_catalog()

    print("Creating replica catalog...")
    workflow.create_replica_catalog()

    print("Creating diamond workflow dag...")
    workflow.create_workflow()

    workflow.write()
