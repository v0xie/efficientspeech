

import yaml
#import os

#import numpy as np
import torch

from datamodule import LJSpeechDataModule

from utils.tools import get_args, timed
from model import EfficientFSModule

os.environ['MASTER_ADDR'] = 'localhost'

def print_args(args):
    opt_log =  '--------------- Options ---------------\n'
    opt = vars(args)
    for k, v in opt.items():
        opt_log += f'{str(k)}: {str(v)}\n'
    opt_log += '---------------------------------------\n'
    print(opt_log)
    return opt_log


#def convert_to_torchscipt(args, pl_module, preprocess_config):
#    if not os.path.exists(args.checkpoints):
#        os.makedirs(args.checkpoints, exist_ok=True)
#    phoneme2mel_ckpt = os.path.join(args.checkpoints, args.phoneme2mel_jit)
#    hifigan_ckpt = os.path.join(args.checkpoints, args.hifigan_jit)
    
#    phoneme2mel, hifigan = load_module(args, pl_module, preprocess_config)

    #print("Saving JIT script ... ", hifigan_ckpt)
    #script = torch.jit.script(hifigan) #hifigan.to_torchscript()
    #torch.jit.save(script, hifigan_ckpt)

#    print("Saving JIT script ... ", phoneme2mel_ckpt)
#    script = torch.jit.script(phoneme2mel) #phoneme2mel.to_torchscript()
#    torch.jit.save(script, phoneme2mel_ckpt)



if __name__ == "__main__":
    args = get_args()

    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    
    args.num_workers *= args.devices 

    datamodule = LJSpeechDataModule(preprocess_config=preprocess_config,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers)

    pl_module = EfficientFSModule(preprocess_config=preprocess_config, lr=args.lr,
                                  warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs,
                                  depth=args.depth, n_blocks=args.n_blocks, block_depth=args.block_depth,
                                  reduction=args.reduction, head=args.head,
                                  embed_dim=args.embed_dim, kernel_size=args.kernel_size,
                                  decoder_kernel_size=args.decoder_kernel_size,
                                  expansion=args.expansion, wav_path=args.out_folder,
                                  hifigan_checkpoint=args.hifigan_checkpoint,
                                  infer_device=args.infer_device, dropout=args.dropout,
                                  verbose=args.verbose)
    #opt_pl_module = torch.compile(pl_module, dynamic=True, mode='reduce-overhead')

    if args.verbose:
        print_args(args)

    # You are using a CUDA device ('NVIDIA GeForce RTX 3080') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    torch.set_float32_matmul_precision('medium')
    tb_logger = TensorBoardLogger(save_dir=args.log_dir)
        
    trainer = Trainer(accelerator=args.accelerator, 
                      devices=args.devices,
                      precision=args.precision,
                      #strategy="ddp",
                      strategy = DDPStrategy(find_unused_parameters=False),
                      check_val_every_n_epoch=10,
                      max_epochs=args.max_epochs,
                      log_every_n_steps=3,
                      logger=tb_logger)

    trainer.fit(pl_module, datamodule=datamodule, ckpt_path=args.resume_from_checkpoint)
