'''
This document is used to reconstruct the situation of the simulated lung. 
The location of the measured voltage needs to be informed. 
The code will first perform the first two stages of reconstruction and then pass the reconstruction 
results to the third stage of reconstruction. The final output of the code contains the real situation, 
the first two stages reconstruction graph, and the third stage processing graph.

'''

import argparse
import os
import torch
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
sys.path.append(r"/home/guided-diffusion-main")
from guided_diffusion import dist_util, logger
from guided_diffusion import my_datasets
import fancon_unet_mode_Dilation
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def normalization_data(data):
    data_max = data.max()
    data_min = data.min()
    gap = data_max - data_min
    temp_data = data - data_min
    temp_data = temp_data / gap
    return temp_data   


def main():

    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()


    # Construct the measured voltage data
    measure_data = pd.read_csv(args.measure_data + '//' + str(args.num) + ".csv")
    measure_data = measure_data.to_numpy()
    measure_data = measure_data.T
    end_data = measure_data[1] - measure_data[0]
    end_data = normalization_data(end_data)
    end_data = np.reshape(end_data,(8,5))
    end_data = my_datasets.Fusion_splicing(end_data,image_size = 64)
    end_data = torch.tensor(end_data).to(dist_util.dev())
    end_data = torch.unsqueeze(end_data,0)
    end_data = torch.unsqueeze(end_data,0)
    end_data = end_data.to(torch.float32)

    model_net = fancon_unet_mode_Dilation.UNet(n_channels=1, n_classes=1)
    model_net.load_state_dict(torch.load(args.pre_model_path))  
    model_net.to(dist_util.dev())
    model_net.eval()

    
    logger.log("creating Diffusion_model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.diffusion_model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating Pre_result...")
    with torch.no_grad():  
        outputs = model_net(end_data)  


    distribution_data = pd.read_csv(args.distribution_path + '//' + str(args.num) + ".csv")
    distribution_data = distribution_data.to_numpy()
    temp3 = np.zeros((64,64))
    temp3[:1,26:38]=1
    temp3[1:64,:] = distribution_data
    distribution_data = torch.tensor(temp3)
    distribution_data = torch.unsqueeze(distribution_data,0)
    distribution_data = distribution_data.numpy()

    
    logger.log("creating End_result...")
    model_kwargs = {}
    if args.class_cond:
        classes = torch.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )   
    sample = sample_fn(
        model,
        (1, 1, args.image_size, args.image_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
        pre_reconstruction = outputs,
    )

    # The first two stages reconstruct the image
    temp = my_datasets.Create_Gaussian_Template(64,32)
    mask = temp.Creat_Circle_Mask()
    mask = torch.tensor(mask)
    mask = torch.unsqueeze(mask,0)
    mask = torch.unsqueeze(mask,0)
    output = outputs.cpu()
    pre_image = output * mask

    # Processing results of the masked diffusion model
    sample = torch.squeeze(sample,1) 
    end_image = sample.cpu()
    end_image = end_image * mask

    plt.subplot(131)
    plt.imshow(distribution_data[0])
    plt.axis("off")
    plt.subplot(132)
    plt.imshow(pre_image[0][0])
    plt.axis("off")
    plt.subplot(133)
    plt.imshow(end_image[0][0])
    plt.axis("off")

    plt.show()


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=10000,
        batch_size=1,
        use_ddim=False,
        diffusion_model_path="/home/guided-diffusion-main/diffusion_model.pt",
        pre_model_path="/home/guided-diffusion-main/model.pt",
        measure_data = "/home/guided-diffusion-main/data/lung/measure",
        distribution_path = "/home/guided-diffusion-main/lung/test/distribution",
        flag_c = 1,
        num = 666,  
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
    


if __name__=="__main__":

    main()
