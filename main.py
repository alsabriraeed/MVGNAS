"""Entry point."""
import argparse       
import time     
import torch              
import searchAlgorithm.trainer as trainer     
import utils.tensor_utils as utils
from ERExtraction.utilsRE import *
   

def build_args():     
    parser = argparse.ArgumentParser(description='MVGNAS')
    parser1 = parser
    register_default_args(parser)
    register_default_args1(parser1)
    
    args = parser.parse_args()
    args1 = parser1.parse_args()
    args1.split_nb = int(args1.split_nb)
    configs = prepare_configs(args1.config_name, args1.dataset, args1.split_nb)

    return args, configs

def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training MVGNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)  
    # controller
    parser.add_argument('--layers_of_child_model', type=int, default=1)       
    parser.add_argument('--num_of_cell', type=int, default=1)                 
    parser.add_argument('--num_of_model', type=int, default=1)                                        
    parser.add_argument('--shared_initial_step', type=int, default=0)                 
    parser.add_argument('--batch_size', type=int, default=64)                          
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')        
    parser.add_argument('--format', type=str, default='two')                
    parser.add_argument('--max_epoch', type=int, default=5)                          
    parser.add_argument("--direct", type=bool, default=False, required=False,
                        help="calling the model directly")
    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)          
    parser.add_argument('--controller_max_step', type=int, default=30, 
                        help='step for controller parameters')                  
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=30) 
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)

    parser.add_argument("--epochs", type=int, default=50,        
                        help="number of training epochs")               
    parser.add_argument("--retrain_epochs", type=int, default=50,    
                        help="number of training epochs")          
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")     
    parser.add_argument("--in-drop", type=float, default=0.6,          
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005, 
                        help="learning rate") #0.005      
    parser.add_argument("--param_file", type=str, default="ADE_test.pkl",
                        help="learning rate")    
    parser.add_argument("--optim_file", type=str, default="opt_ADE_test.pkl",
                        help="optimizer save path")     
    parser.add_argument('--weight_decay', type=float, default=5e-4)       
    parser.add_argument('--max_param', type=float, default=5E6)                
    parser.add_argument('--supervised', type=bool, default=True)        
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")    

def register_default_args1(parser1):
    parser1.add_argument('-c', '--config_name', default='with_view_encoding')                           
    parser1.add_argument('-d', '--dataset', default=ADE, choices=DATASETS)                     
    # parser1.add_argument('-d', '--dataset', default=BIORELEX, choices=DATASETS)                                
    parser1.add_argument('-s', '--split_nb', default=0)  # Only affect ADE dataset                                            

def main(args, configs):  # pylint:disable=redefined-outer-name                  

    if args.cuda and not torch.cuda.is_available():  
        args.cuda = False        
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args,configs)

    if args.mode == 'train':
        print(args)
        trnr.train()
    elif args.mode == 'derive':
        trnr.derive()
    else:
        raise Exception(f"[!] Mode not found: {args.mode}")


if __name__ == "__main__":
    args,configs = build_args()
    main(args,configs)
