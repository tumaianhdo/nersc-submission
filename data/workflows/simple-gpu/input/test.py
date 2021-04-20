import os
import torch
from mpi4py import MPI

if __name__ == "__main__":
	worker_id = MPI.COMM_WORLD.Get_rank()
	print(f"Worker = {worker_id}: torch.cuda.is_available() = {torch.cuda.is_available()}")
	print(f"Worker = {worker_id}: torch.cuda.current_device() = {torch.cuda.current_device()}")
	print(f"Worker = {worker_id}: torch.cuda.device(0) = {torch.cuda.device(0)}")
	print(f"Worker = {worker_id}: torch.cuda.device_count() = {torch.cuda.device_count()}")
	print(f"Worker = {worker_id}: torch.cuda.get_device_name(0) = {torch.cuda.get_device_name(0)}")
	print(f'Worker = {worker_id}: torch.device("cuda") = {torch.device("cuda")}')
	print(f"Worker = {worker_id}: CPUs = {os.sched_getaffinity(0)}")