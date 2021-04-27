import os
import torch
import cv2

if __name__ == "__main__":
	print(f"Hello World!!!")
	print(f"CPUs = {os.sched_getaffinity(0)}")