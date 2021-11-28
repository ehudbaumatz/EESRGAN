import argparse
import os
import torch

import data_loader.data_loaders as module_data
import model.ESRGAN_EESN_Model as ESRGAN_EESN
from parse_config import ConfigParser
from utils import tensor2img, save_img
from tqdm.auto import tqdm

'''
python test.py -c config_GAN.json
'''



def main(config):

    # data_loader = module_data.COWCGANFrcnnDataLoader('datasets/DetectionPatches_256x256/Potsdam_ISPRS/HR/x4/valid_img/',
    # 'datasets/DetectionPatches_256x256/Potsdam_ISPRS/LR/x4/valid_img/', 1, training=False)
    # tester = COWCGANFrcnnTrainer(config=config, data_loader=data_loader)
    # tester.test()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # config = ConfigParser(json.loads(path.read_text()))
    model = ESRGAN_EESN.ESRGAN_EESN_Model(config, device)
    model.load()

    # this is not a bug - ht e SR and LQ have the same paths
    loader = module_data.COWCGANDataLoader(config["data_loader"]["args"]["data_dir_LQ"],
                                           config["data_loader"]["args"]["data_dir_LQ"],
                                                     batch_size=1, training=False, num_workers=0)


    for data, _ in tqdm(loader):
        #print(val_data)
        img_name = data['LQ_path'][0] #  os.path.splitext(os.path.basename(data['LQ_path'][0]))[0]
        save_to = img_name[:-3] + 'SR.png'
        # print(save_to)

        model.feed_data(data)
        model.test()
        #
        visuals = model.get_current_visuals()
        # sr_img = tensor2img(visuals['SR'])  # uint8
        final_SR = tensor2img(visuals['final_SR']) # uint8
        save_img(final_SR, save_to)




if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
