#!/usr/bin/env python3

import glob 
import os
import numpy as np
from Pegasus.api import *
from pathlib import Path
import logging
import pickle
import time
import argparse
# import torch
import random


logging.basicConfig(level=logging.DEBUG)
props = Properties()
props["pegasus.mode"] = "development"
# props["pegasus.gridstart.invoke.length"] = "100"
props["pegasus.transfer.links"] = "true"
# props["pegasus.integrity.checking"] = "none"
props["pegasus.transfer.threads"] = "16"
#props["dagman.retry"] = "3"
#props["pegasus.transfer.arguments"] = "-m 1"
props.write()


# 10 percent of data (this will need to be replade
# by actual number to run with full dataset)
# MAX_IMG_0 = 84
# MAX_IMG_1 = 80
# MAX_IMG_2 = 8
# MAX_IMG_3 = 39
# MAX_IMG_4 = 78

# full dataset
MAX_IMG_0 = 8436
MAX_IMG_1 = 8069
MAX_IMG_2 = 579
MAX_IMG_3 = 3903
MAX_IMG_4 = 7806



def create_pkl(name):
	pkl_filename = name
	file = open(pkl_filename, 'ab')
	pickle.dump("", file, pickle.HIGHEST_PROTOCOL)
	return pkl_filename



def split_preprocess_jobs(preprocess_images_job, input_images, postfix):
	
	resized_images = []
	
	for i in range(len(input_images)):
		curr = i % len(preprocess_images_job)
		preprocess_images_job[curr].add_inputs(input_images[i])
		out_file = File(str(input_images[i]).split(".")[0] + postfix + ".jpg")
		preprocess_images_job[curr].add_outputs(out_file)
		resized_images.append(out_file)
		
	return resized_images


def add_augmented_images(class_str, num, start_num):
	augmented_files = []
	for i in range(num):
		augmented_files.append(File("train_" + class_str + "_" + str(start_num) + "_proc.jpg"))
		start_num +=1
	return augmented_files


def create_files_hpo(input_files):
	files = []
	for file in input_files:
		name = File(file.split("/")[-1].split(".")[0] + "_proc.jpg")
		files.append(name)
	return files


def add_prefix(file_paths, prefix):
	new_paths = []
	for fpath in file_paths:
		new_paths.append(prefix + "_" + fpath)
	return new_paths


def split_data_filenames(file_paths,seed):
	random.seed(seed)
	random.shuffle(file_paths)
	train, val, test = np.split(file_paths, [int(len(file_paths)*0.8), int(len(file_paths)*0.9)])
	return train, val, test



def get_files(all_images_paths,rc):
	input_images  = []
	for image_path in all_images_paths:
		image_file = image_path.split("/")[-1]
		input_images.append(File(image_file))
		rc.add_replica("local", image_file,  os.path.join(os.getcwd(), image_path))
	return input_images


def create_output_file_names(class_str, img_per_class):
	output_files = []
	for j in range(img_per_class):
		output_files.append("class_{}_{}.jpg".format(class_str,j))
	return output_files


def run_workflow(DATA_PATH):

	# SITE CATALOG
	#---------------------------------------------------------------------------------------------------------
	sc = SiteCatalog()
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
					runtime="300"
					# grid_start = "NoGridStart"
				)\
				.add_env(key="PEGASUS_HOME", value="${NERSC_PEGASUS_HOME}")
	
	sc.add_sites(cori)
	sc.write()

	### ADD INPUT FILES TO REPILCA CATALOG
	#-------------------------------------------------------------------------------------------------------
	rc = ReplicaCatalog()
	github_location = "https://raw.githubusercontent.com/tumaianhdo/nersc-submission/main/data/workflows/galaxy"
	remote_data_dir = "/global/cscratch1/sd/${NERSC_USER}/application/nersc-submission/data/workflows/galaxy"
	full_galaxy_images = glob.glob(DATA_PATH + "*.jpg")
	input_images  = []
	for image_path in full_galaxy_images:
		image_file = image_path.split("/")[-1]
		input_images.append(File(image_file))
		# rc.add_replica("GitHub", image_file,  os.path.join(github_location, image_path))
		rc.add_replica("cori", image_file,  os.path.join(remote_data_dir, image_path))
		# rc.add_replica("local", image_file,  os.path.join(os.getcwd(), image_path))
	# input_images = get_files(full_galaxy_images,rc)

	metadata_file      = 'training_solutions_rev1.csv'
	rc.add_replica("GitHub", metadata_file,  os.path.join(github_location, metadata_file))
	# rc.add_replica("local", metadata_file,  os.path.join(os.getcwd(), metadata_file))

	dataset_class_0 = create_output_file_names(0,MAX_IMG_0)
	dataset_class_1 = create_output_file_names(1,MAX_IMG_1)
	dataset_class_2 = create_output_file_names(2,MAX_IMG_2)
	dataset_class_3 = create_output_file_names(3,MAX_IMG_3)
	dataset_class_4 = create_output_file_names(4,MAX_IMG_4)

	dataset_class = dataset_class_0 + dataset_class_1 + dataset_class_2 + dataset_class_3 + dataset_class_4
	dataset_class.sort()

	train, val, test = split_data_filenames(dataset_class,SEED)

	pf_train = add_prefix(train, "train")
	pf_val   = add_prefix(val, "val")
	pf_test  = add_prefix(test, "test")

	output_images = pf_train + pf_val + pf_test
	output_files = [File(i) for i in output_images]  

	# ADDITIONAL PYTHON SCRIPS NEEDED BY TUNE_MODEL
	#-------------------------------------------------------------------------------------------------------
	data_loader_fn = 'data_loader.py'
	data_loader_file = File(data_loader_fn)
	rc.add_replica("GitHub", data_loader_fn, os.path.join(github_location, "bin", data_loader_fn))
	# rc.add_replica("local", data_loader_fn, os.path.join(os.getcwd(), "bin/" + data_loader_fn ))

	model_selection_fn = 'model_selection.py'
	model_selection_file = File(model_selection_fn)
	rc.add_replica("GitHub", model_selection_fn, os.path.join(github_location, "bin", model_selection_fn))
	# rc.add_replica("local", model_selction_fn, os.path.join(os.getcwd(),"bin/" + model_selction_fn ))

	# FILES FOR vgg16_hpo.py VGG 16
	#--------------------------------------------------------------------------------------------------------
	# vgg16_pkl = create_pkl("hpo_galaxy_vgg16.pkl")
	vgg16_pkl = 'hpo_galaxy_vgg16.pkl'
	vgg16_pkl_file = File(vgg16_pkl)
	rc.add_replica("GitHub", vgg16_pkl, os.path.join(github_location, vgg16_pkl))   
	# rc.add_replica("local", vgg16_pkl, os.path.join(os.getcwd(), vgg16_pkl))    

	# FILES FOR train_model.py 
	#--------------------------------------------------------------------------------------------------------
	# checkpoint_vgg16_pkl = create_pkl("checkpoint_vgg16.pkl")
	checkpoint_vgg16_pkl = 'checkpoint_vgg16.pkl'
	checkpoint_vgg16_pkl_file = File(checkpoint_vgg16_pkl)
	rc.add_replica("GitHub", checkpoint_vgg16_pkl, os.path.join(github_location, checkpoint_vgg16_pkl))
	# rc.add_replica("local",checkpoint_vgg16_pkl, os.path.join(os.getcwd(), checkpoint_vgg16_pkl))

	create_dataset_fn = 'create_dataset.py'
	create_dataset_file = File(create_dataset_fn)
	rc.add_replica("GitHub", create_dataset_fn, os.path.join(github_location, "bin", create_dataset_fn))

	preprocess_images_fn = 'preprocess_resize.py'
	preprocess_images_file = File(preprocess_images_fn)
	rc.add_replica("GitHub", preprocess_images_fn, os.path.join(github_location, "bin", preprocess_images_fn))

	augment_images_fn = 'preprocess_augment.py'
	augment_images_file = File(augment_images_fn)
	rc.add_replica("GitHub", augment_images_fn, os.path.join(github_location, "bin", augment_images_fn))

	vgg16_hpo_fn = 'vgg16_hpo.py'
	vgg16_hpo_file = File(vgg16_hpo_fn)
	rc.add_replica("GitHub", vgg16_hpo_fn, os.path.join(github_location, "bin", vgg16_hpo_fn))

	
	vgg16_train_fn = 'train_model_vgg16.py'
	vgg16_train_file = File(vgg16_train_fn)
	rc.add_replica("GitHub", vgg16_train_fn, os.path.join(github_location, "bin", vgg16_train_fn))

	vgg16_eval_fn = 'eval_model_vgg16.py'
	vgg16_eval_file = File(vgg16_eval_fn)
	rc.add_replica("GitHub", vgg16_eval_fn, os.path.join(github_location, "bin", vgg16_eval_fn))

	rc.write()

	# TRANSFORMATION CATALOG
	#---------------------------------------------------------------------------------------------------------
	tc = TransformationCatalog()

	# Pegasus transfer prerequisites
	pegasus_transfer = Transformation("transfer", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
							.add_pegasus_profile(
								queue="@escori",
								runtime="7200",
								glite_arguments="--qos=xfer --licenses=SCRATCH"
							)\
							.add_profiles(Namespace.PEGASUS, key="transfer.threads", value="16")\
							.add_env(key="PEGASUS_TRANSFER_THREADS", value="16")	

	pegasus_dirmanager = Transformation("dirmanager", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
							.add_pegasus_profile(
								queue="@escori",
								runtime="7200",
								glite_arguments="--qos=xfer --licenses=SCRATCH"
							)
	pegasus_cleanup = Transformation("cleanup", namespace="pegasus", site="cori", pfn="$PEGASUS_HOME/bin/pegasus-transfer", is_stageable=False)\
							.add_pegasus_profile(
								queue="@escori",
								runtime="7200",
								glite_arguments="--qos=xfer --licenses=SCRATCH"
							)
	system_chmod = Transformation("chmod", namespace="system", site="cori", pfn="/usr/bin/chmod", is_stageable=False)\
							.add_pegasus_profile(
								queue="@escori",
								runtime="120",
								glite_arguments="--qos=xfer --licenses=SCRATCH"
							)

	wrapper = Transformation("wrapper", site="cori", pfn="https://raw.githubusercontent.com/tumaianhdo/nersc-submission/main/data/workflows/galaxy/bin/wrapper.sh", is_stageable=True)\
							.add_pegasus_profile(
								cores="1",
								runtime="1800",
								# exitcode_success_msg="End of program",
								glite_arguments="--qos=debug --constraint=haswell --licenses=SCRATCH",
								grid_start="NoGridStart"
							)\
							.add_env(key="USER_HOME", value="${NERSC_USER_HOME}")
							# .add_profiles(Namespace.PEGASUS, key="transfer.threads", value="8")\
							# .add_env(key="PEGASUS_TRANSFER_THREADS", value="8")	

	gpu_wrapper = Transformation("gpu_wrapper", site="cori", pfn="https://raw.githubusercontent.com/tumaianhdo/nersc-submission/main/data/workflows/galaxy/bin/gpu_wrapper.sh", is_stageable=True)\
							.add_pegasus_profile(
								# cores="2",
								# ppn="",
								# runtime="14400",
								queue="@escori"
								# exitcode_success_msg="End of program",
								# glite_arguments="--constraint=gpu --gpus=1 --ntasks=1 --cpus-per-task=1 --gpus-per-task=1 --hint=nomultithread"
							)\
							.add_env(key="USER_HOME", value="${NERSC_USER_HOME}")
							# .add_profiles(Namespace.PEGASUS, key="transfer.threads", value="8")\
							# .add_env(key="PEGASUS_TRANSFER_THREADS", value="8")	
							# .add_env(key="NUM_GPUS", value="1")\
							# .add_env(key="CORES_PER_GPU", value="1")\
							

	# Data Aqusition: Create Dataset
	# create_dataset = Transformation("create_dataset",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/create_dataset.py"), 
	# 								is_stageable= True)  

	# # Data preprocessing part 1: image resize
	# preprocess_images = Transformation("preprocess_images",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/preprocess_resize.py"), 
	# 								is_stageable= True)

	# # Data preprocessing part 2: image augmentation
	# augment_images = Transformation("augment_images",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/preprocess_augment.py"), 
	# 								is_stageable= True)

	# # HPO: main script
	# vgg16_hpo      = Transformation("vgg16_hpo",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/vgg16_hpo.py"), 
	# 								is_stageable= True)

	# # Train Model
	# train_model     = Transformation("train_model",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/train_model_vgg16.py"), 
	# 								is_stageable= True)

	# # Eval Model
	# eval_model     = Transformation("eval_model",site="local",
	# 								pfn = str(Path(".").parent.resolve() / "bin/eval_model_vgg16.py"), 
	# 								is_stageable= True)

	tc.add_transformations(pegasus_transfer, pegasus_dirmanager, pegasus_cleanup, system_chmod, wrapper, gpu_wrapper)
	# tc.add_transformations(
	# 	create_dataset,
	# 	preprocess_images,
	# 	augment_images,
	# 	vgg16_hpo,
	# 	train_model,
	# 	eval_model
	# 	)
	tc.write()

	## CREATE WORKFLOW
	#---------------------------------------------------------------------------------------------------------
	wf = Workflow('Galaxy-Classification-Workflow')

	job_create_dataset = Job(wrapper)\
						.add_args("create_dataset.py -seed {} --max_img {}".format(SEED, 10000))\
						.add_inputs(*input_images, File(metadata_file), create_dataset_file)\
						.add_outputs(*output_files)
						# .add_profiles(Namespace.PEGASUS, key="gridstart.invoke.length", value="100")

	job_preprocess_images = [Job(wrapper).add_args(preprocess_images_file).add_inputs(preprocess_images_file) for i in range(NUM_PREPROCESSORS)]
	resized_images = split_preprocess_jobs(job_preprocess_images, output_files, "_proc")

	train_class_2         = "train_class_2"
	train_files_class_2   = [i for i in output_images if train_class_2 in i]
	input_aug_class_2     = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_2 ]
	output_aug_class_2    = add_augmented_images("class_2", NUM_CLASS_2, 4000)
	
	train_class_3         = "train_class_3"
	train_files_class_3   = [i for i in output_images if train_class_3 in i]
	input_aug_class_3     = [ File(file.split("/")[-1].split(".")[0] + "_proc.jpg") for file in train_files_class_3 ]
	output_aug_class_3    = add_augmented_images("class_3", NUM_CLASS_3, 4000)
	
	job_augment_class_2 = Job(wrapper)\
						.add_args("preprocess_augment.py --class_str class_2 --num {}".format(NUM_CLASS_2))\
						.add_inputs(*input_aug_class_2, augment_images_file)\
						.add_outputs(*output_aug_class_2)

	job_augment_class_3 = Job(wrapper)\
						.add_args("preprocess_augment.py --class_str class_3 --num {}".format(NUM_CLASS_3))\
						.add_inputs(*input_aug_class_3, augment_images_file)\
						.add_outputs(*output_aug_class_3)

	# Job HPO
	train_class = 'train_class_'
	train_class_files = [i for i in output_images if train_class in i]
	val_class = 'val_class_'
	val_class_files = [i for i in output_images if val_class in i]
	test_class = 'test_class_'
	test_class_files = [i for i in output_images if test_class in i]

	input_hpo_train = create_files_hpo(train_class_files)
	input_hpo_val   = create_files_hpo(val_class_files)
	input_hpo_test  = create_files_hpo(test_class_files)

	best_params_file = File("best_vgg16_hpo_params.txt")
	hpo_log_fn = "hpo.log"
	hpo_log_file = File(hpo_log_fn)
	
	gpu_slurm_flags = "--constraint=gpu --gpus="+str(NUM_WORKERS)+" --ntasks="+str(NUM_WORKERS)+" --cpus-per-task="+str(CORES_PER_GPU)+" --gpus-per-task=1 --hint=nomultithread"
	trials_per_worker = TRIALS // NUM_WORKERS
	job_vgg16_hpo = Job(gpu_wrapper)\
						.add_args("vgg16_hpo.py --trials {} --epochs {} --batch_size {}".format(trials_per_worker, EPOCHS, BATCH_SIZE))\
						.add_inputs(*output_aug_class_3, *output_aug_class_2,*input_hpo_train, *input_hpo_val, data_loader_file, model_selection_file, vgg16_hpo_file)\
						.add_checkpoint(vgg16_pkl_file, stage_out=True)\
						.add_outputs(best_params_file)\
						.add_profiles(Namespace.PEGASUS, key="glite.arguments", value=gpu_slurm_flags)\
						.add_profiles(Namespace.ENV, key="NUM_GPUS", value=str(NUM_WORKERS))\
						.add_profiles(Namespace.ENV, key="CORES_PER_GPU", value=str(CORES_PER_GPU))\
						.add_profiles(Namespace.PEGASUS, key="runtime", value="21600")
						# .add_profiles(Namespace.PEGASUS, key="maxwalltime", value="360")\
						# .add_profiles(Namespace.ENV, key="EXTRA_ARGS", value="--output=" + hpo_log_fn)


	# # Job train model
	gpu_slurm_flags = "--constraint=gpu --gpus=1 --ntasks=1 --cpus-per-task="+str(CORES_PER_GPU)+" --gpus-per-task=1 --hint=nomultithread"
	job_train_model = Job(gpu_wrapper)\
						.add_args("{} --epochs {} --batch_size {}".format(vgg16_train_fn, EPOCHS, BATCH_SIZE))\
						.add_inputs(*output_aug_class_3, *output_aug_class_2, best_params_file,\
							*input_hpo_train, *input_hpo_val, *input_hpo_test,\
							data_loader_file, model_selection_file, vgg16_train_file)\
						.add_checkpoint(checkpoint_vgg16_pkl_file , stage_out=True)\
						.add_outputs(File("final_vgg16_model.pth"), File("loss_vgg16.png"))\
						.add_profiles(Namespace.PEGASUS, key="glite.arguments", value=gpu_slurm_flags)\
						.add_profiles(Namespace.ENV, key="NUM_GPUS", value="1")\
						.add_profiles(Namespace.ENV, key="CORES_PER_GPU", value=str(CORES_PER_GPU))
						# .add_profiles(Namespace.PEGASUS, key="runtime", value="7200")\
						# .add_profiles(Namespace.PEGASUS, key="maxwalltime", value="120")\
						


	# # Job eval
	gpu_slurm_flags = "--constraint=gpu --gpus=1 --ntasks=1 --cpus-per-task="+str(CORES_PER_GPU)+" --gpus-per-task=1 --hint=nomultithread"
	job_eval_model = Job(gpu_wrapper)\
						.add_args(vgg16_eval_fn)\
						.add_inputs(*input_hpo_test,data_loader_file,best_params_file,\
									model_selection_file,File("final_vgg16_model.pth"),\
									vgg16_eval_file)\
						.add_outputs(File("final_confusion_matrix_norm.png"),File("exp_results.csv"))\
						.add_profiles(Namespace.PEGASUS, key="glite.arguments", value=gpu_slurm_flags)\
						.add_profiles(Namespace.ENV, key="NUM_GPUS", value="1")\
						.add_profiles(Namespace.ENV, key="CORES_PER_GPU", value=str(CORES_PER_GPU))
						# .add_profiles(Namespace.PEGASUS, key="runtime", value="7200")\
						# .add_profiles(Namespace.PEGASUS, key="maxwalltime", value="120")\
						


	## ADD JOBS TO THE WORKFLOW
	wf.add_jobs(job_create_dataset, *job_preprocess_images, job_augment_class_2 ,job_augment_class_3, job_vgg16_hpo, job_train_model, job_eval_model)
	# wf.add_jobs(job_create_dataset, *job_preprocess_images, job_augment_class_2 ,job_augment_class_3, job_vgg16_hpo, job_train_model)
	# wf.add_dependency(job_create_dataset, children=job_preprocess_images)
	# wf.add_jobs(job_create_dataset,
	# 			*job_preprocess_images,job_augment_class_2 ,job_augment_class_3, job_vgg16_hpo,\
	# 			job_train_model,job_eval_model)  


	# EXECUTE THE WORKFLOW
	#-------------------------------------------------------------------------------------
	try:
		wf.plan(submit=False, sites=["cori"], output_sites=["cori"], dir="submit", cleanup="inplace")
		# wf.wait()
		# wf.statistics()
	except PegasusClientError as e:
		print(e.output)   

	# graph_filename = "galaxy-wf.dot"
	# wf.graph(include_files=True, no_simplify=True, label="xform-id", output = graph_filename)


def main():
	
	start = time.time()
	
	global ARGS
	global BATCH_SIZE
	global SEED
	global DATA_PATH
	global EPOCHS
	global TRIALS
	global NUM_WORKERS
	global NUM_CLASS_2
	global NUM_CLASS_3
	global MAXTIMEWALL
	global CORES_PER_GPU
	global NUM_PREPROCESSORS

	
	parser = argparse.ArgumentParser(description="Galaxy Classification")   
	parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
	parser.add_argument('--seed', type=int, default=10, help='select seed number for reproducibility')
	# parser.add_argument('--data_path', type=str, default='galaxy_data/',help='path to dataset ')
	# parser.add_argument('--num_class_2', type=int, default= 3, help = "number of augmented class 2 files")
	# parser.add_argument('--num_class_3', type=int, default= 4, help = "number of augmented class 3 files")
	parser.add_argument('--data_path', type=str, default='full_galaxy_data/',help='path to dataset ')
	parser.add_argument('--num_class_2', type=int, default= 7000, help = "number of augmented class 2 files")
	parser.add_argument('--num_class_3', type=int, default= 4000, help = "number of augmented class 3 files")
	parser.add_argument('--epochs', type=int,default=10, help = "number of training epochs")  
	parser.add_argument('--trials', type=int,default=1, help = "number of trials") 
	parser.add_argument('--num_workers', type=int, default= 1, help = "number of workers")
	parser.add_argument('--num_preprocessors', type=int, default= 1, help = "number of preprocessors")
	parser.add_argument('--maxwalltime', type=int, default= 120, help = "maxwalltime")
	parser.add_argument('--cores_per_gpu', type=int, default= 5, help = "Number of physical cores per GPU")

	


	ARGS        = parser.parse_args()
	BATCH_SIZE  = ARGS.batch_size
	SEED        = ARGS.seed
	DATA_PATH   = ARGS.data_path
	EPOCHS      = ARGS.epochs
	TRIALS      = ARGS.trials
	NUM_WORKERS = ARGS.num_workers
	NUM_PREPROCESSORS = ARGS.num_preprocessors
	NUM_CLASS_2 = ARGS.num_class_2
	NUM_CLASS_3 = ARGS.num_class_3
	MAXTIMEWALL = ARGS.maxwalltime
	CORES_PER_GPU = ARGS.cores_per_gpu


	# torch.manual_seed(SEED)
	np.random.seed(SEED)

	run_workflow(DATA_PATH)
	
	exec_time = time.time() - start

	print('Execution time in seconds: ' + str(exec_time))
	

	return

if __name__ == "__main__":
	
	main()