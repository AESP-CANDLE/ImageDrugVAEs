

class Config():
    def __init__(self):
        self.config_im_im = {

        }

        self.config_im_smi = {
            'log_images_edit_dist' : None, # put a folder here and it will output images with smiles
            'save_model_checkpoints' : 'best', #'best' saves only best model and optimizer states, None doesn't do anything, 'all' saves per epoch
            'log_interval' :  4, # batch interval to print to screen
            'print_example_smiles' : False, #will output examples of smiles
            'use_checkpoint' : None,
            'load_checkpoint' : None,

            'batch_size' : 350,
            'num_epochs' : 1000,
            'early_stopping' : None,  #import and use EarlyStopping module from utils
            'distrubted'     : False,  #run models on separate GPUs and shuffle data, use EXPORT CUDA DEVICES to specify three devices.
            'grad_clip' : 5,
            'alpha_c' : 1.0,  # regularization parameter for 'doubly stochastic attention', as in the paper

            ##Encoder
            'use_pretrained_encoder' : None, # Put file of a torch.save ENTIRE MODEL (not the state dict)
            'train_encoder' : True,
            'encoder_dim': 512,
            # # # Optimizer
            'enc_optimizer': 'adam',  # pass optimizer here
            'enc_optimizer_kwags': {'lr': 0.001},  # pass other options here
            'enc_lr_scheduling'  : None, # ['constant', 'reduce_on_plateau', 'annealing']
            'enc_lr_scheduling_kwargs' : None, # pass kwargs here

            #Decoder
            'emb_dim' : 128,  # dimension of word embeddings
            'attention_dim' : 256,  # dimension of attention linear layers
            'decoder_dim' : 256,  # dimension of decoder RNN
            'dropout' : 0.15,

            # # # Optimizer
            'dec_optimizer' : 'adam', # pass optimizer here
            'dec_optimizer_kwargs' : {'lr' : 0.001}, #pass other options here
            'dec_lr_scheduling': None,  # ['constant', 'reduce_on_plateau', 'annealing']
            'dec_lr_scheduling_kwargs': None,  # pass kwargs here
        }

        self.config_global = {
            'cuda_devices' : None, #use this or do it for your terminal
            'use_comet' : True,
            'comet_project_name': "pytorch",
            'cuda': True,
            'seed': 42,  # set torch manuel seed
            'data_loader_kwargs': {'num_workers': 64, 'pin_memory': True},  # passes to dataloader
            'train_data': 'file.csv',
            'val_data': 'file.csv',
            'vocab_file' : 'vocab.pkl', #use make vocab utility on your dataset
            'vocab_max_len' : 70,
            'start_char' : '!',
            'end_char' :  '?'
        }

    def get_im_smi_config(self):
        return {**self.config_global, **self.config_im_smi}

    def get_im_im_config(self):
        return {**self.config_global, **self.config_im_im}

    def get_global_config(self):
        return self.config_global