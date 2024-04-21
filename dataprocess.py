
import torch 
import os

import numpy as np

import pickle
# load all the data
import multiprocessing
from multiprocessing import Pool
import struct
import sys

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def float_to_hex(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def count_1(n):
    base=0x000000ff
    res=list()
    for i in range(4):
        tmp=n&base
        if tmp > 0:
            res.append(1)
        else:
            res.append(0)
        base<<=8
    return res

def process_ckp(start, end, filename):
    if start == 1 :
        start += 1
    pm_metrics=list()
    # print()
    path="/home/cc/scripts/output1/checkpoint-"+ str(start -1) +"/"
    # grad=path+"/gradients.bin"
    param=path+filename
    # grad_pre=torch.load(grad)

    param_pre=torch.load(param,map_location=torch.device('cpu'))
    # param_pre=torch.load(param)
    for i in range(start,end):
        print("process "+str(i))
        path="/home/cc/scripts//output1/checkpoint-"+str(i)+"/"
        # grad=path+"/gradients.bin"
        param=path+filename
        # grad=torch.load(grad)
        param=torch.load(param,map_location=torch.device('cpu'))
        # param=torch.load(param)
        keys=param.keys()
        
        number=0
        chd_num=0
        # v_chd=list()
        g_v_chd_var=list()
        dirty_bytes =[0,0,0,0]
        for key in keys:
            # print(key)
            # v_cur=param[key].cpu().numpy().flatten().tolist()
            # v_pre=param_pre[key].cpu().numpy().flatten().tolist()
            # import pdb; pdb.set_trace()
            v_cur=param[key]
            v_pre=param_pre[key]

            chd_num+=torch.numel(v_cur)
            v_tmp=v_cur-v_pre

            v_tmp_abs=torch.abs(v_tmp)
            v_chd_var = torch.abs((v_pre-v_cur)/v_pre)
            # v_chd+=v_tmp_abs.cpu().numpy().flatten().tolist()
            v_chd_var=v_chd_var.cpu().numpy()
            v_chd_var[~np.isfinite(v_chd_var)]=0
            v_chd_var[np.isnan(v_chd_var)]=0
            
            g_v_chd_var+=v_chd_var.flatten().tolist()
            # val_median=torch.median(v_tmp)
            number+=torch.count_nonzero(v_tmp)

            v_cur=param[key].cpu().numpy().flatten().tolist()
            v_pre=param_pre[key].cpu().numpy().flatten().tolist()
            # print("dirty bytes")
            for j in range(len(v_cur)):
                # print(v_cur[j])
                # print(v_cur[j].tolist())
                v_diff = float_to_hex(v_cur[j]) ^ float_to_hex(v_pre[j])
                res = count_1(v_diff)
                dirty_bytes[0] += res[0]
                dirty_bytes[1] += res[1]
                dirty_bytes[2] += res[2]
                dirty_bytes[3] += res[3]
        
        print(len(g_v_chd_var))
        metrics_chd_var=np.percentile(g_v_chd_var, [0,25,50,75,90,95,99,100])
        # print(metrics_chd_var)
        pm_metrics.append(metrics_chd_var)
        print("testst")
        print("step: {:d} {:.16f} {:.16f} {:.16f} {:.16f} {:.16f} {:.16f} {:.16f} {:.16} {:.16}".format(i,metrics_chd_var[0],metrics_chd_var[1],metrics_chd_var[2],metrics_chd_var[3],metrics_chd_var[4],metrics_chd_var[5], metrics_chd_var[6], metrics_chd_var[7], number/chd_num))
        # del param_pre
        param_pre=param
        
        # how many parameters are changed
        pm_metrics.append(number/chd_num)
        # dirty bytes distribution
        pm_metrics.append(dirty_bytes)
        
    filepath="/home/cc/scripts/results/"
    file=open(filepath+filename+str(start)+str(end)+".bin", "wb")
    pickle.dump(pm_metrics,file)
    file.close()
    print("step close file")
    

if __name__ == "__main__":
    # process_ckp(1, 48, "pytorch_model.bin")
    print(sys.argv)

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    with Pool(24) as pool:
        param=[]
        for i in range(start, end, 2):
            if i+3>=end:
                param.append((i, end, "pytorch_model.bin"))
            else:
                param.append((i, i+2, "pytorch_model.bin"))
        for i in range(start, end, 2):
            if i+3>=end:
                param.append((i, end, "gradients.bin"))
            else: 
                param.append((i, i+2, "gradients.bin"))
        
        print(param)
        start_p = [param[i][0] for i in range(len(param))]
        end_p = [param[i][1] for i in range(len(param))]
        filename_p = [param[i][2] for i in range(len(param))]
        pool.starmap(process_ckp, [*zip(start_p, end_p, filename_p)])



