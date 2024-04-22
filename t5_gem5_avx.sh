#!/bin/bash

# $1 how many parameters need to be updated
# $2 file name for memory trace
# $3 file for statistic data 

# for T5-large 737M
# for GPT-2 122M
# for albert 223M
# for bert 334M



nohup ./build/X86/gem5.opt \
	--debug-flag=MemoryAccess \
	-d $3  \
	./configs/example/se.py \
       	--cmd="/home/cc/avx_cpp_single_thread/cpu_comp" \
	--options="/home/cc/avx_cpp_single_thread/pytorch_model.bin_c /home/cc/avx_cpp_single_thread/gradients.bin_c 48 737000000" \
	--cpu-type=DerivO3CPU \
	--caches \
	--mem-type DDR4_2400_4x16 \
	--num-cpus 48 \
	--mem-channels 8 \
	--mem-ranks 16 \
	--mem-size="16340MB" > results/$2 2>&1 &
