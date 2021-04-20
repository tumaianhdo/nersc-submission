#!/usr/bin/python3
import os
import sys
import logging
logging.basicConfig(level=logging.DEBUG)
from pathlib import Path
from argparse import ArgumentParser
# --- Import Pegasus API -----------------------------------------------------------
from Pegasus.api import *

if __name__ == '__main__':
	parser = ArgumentParser(description="Simple Workflow")
	props = Properties()

	sc = SiteCatalog()
	wf_dir = Path(__file__).parent.resolve()

	shared_scratch_dir = os.path.join(wf_dir, "scratch")
	local_storage_dir = os.path.join(wf_dir, "output")

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
	
	sc.add_sites(local, cori)

	tc = TransformationCatalog()
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
	wrapper = Transformation("wrapper", site="cori", pfn="https://raw.githubusercontent.com/tumaianhdo/nersc-submission/main/data/workflows/simple-gpu/bin/wrapper.sh", is_stageable=True)\
					.add_pegasus_profile(
						# cores="2",
						# ppn=""
						runtime="1200",
						queue="@escori",
						# exitcode_success_msg="End of program",
						glite_arguments="--constraint=gpu --gpus=2 --ntasks=2 --cpus-per-task=2 --gpus-per-task=1 --hint=nomultithread"
					)\
					.add_env(key="PEGASUS_NUM_WORKERS", value="2")
	tc.add_transformations(pegasus_transfer, pegasus_dirmanager, pegasus_cleanup, system_chmod, wrapper)

	rc = ReplicaCatalog()
	github_location = "https://raw.githubusercontent.com/tumaianhdo/nersc-submission/main/data/workflows/simple-gpu/input"
	rc.add_replica("GitHub", "test.py", os.path.join(github_location, "test.py"))

	wf = Workflow("simple-gpu", infer_dependencies=True)
	test_py = File("test.py")
	out_file = File("output")
	wrapper_job = Job("wrapper")\
						.add_inputs(test_py)\
						.add_outputs(out_file, stage_out=True, register_replica=True)
	wf.add_jobs(wrapper_job)

	props.write()
	sc.write()
	rc.write()
	tc.write()
	wf.write()



