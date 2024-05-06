import os
import time
import random
import argparse

import torch
from torch.utils.data import DataLoader

from loss_function import GE2ELoss
from data_load import SpeechDataset
from parse_config import config
from model_network import SpeechEmbedder

torch.autograd.set_detect_anomaly(True)

def train(model_path=None):

    if config.task == "tdsv":
        train_config = config.train.TD_SV_train
    else:
        train_config = config.train.TI_SV_train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load train dataset and dataloader
    train_set = SpeechDataset()
    train_loader = DataLoader(train_set, batch_size=train_config.N, shuffle=False, drop_last=True)

    # construct midel
    speech_embedder = SpeechEmbedder().to(device)
    ge2e_loss = GE2ELoss().to(device)

    # whether restore previous saved model
    if train_config.restore:
        model_load = torch.load(model_path)
        speech_embedder.load_state_dict(model_load['speech_embedder'])
        ge2e_loss.load_state_dict(model_load['ge2e_loss'])

    # set optimizer. For different component, set different gradient scale
    optimizer = torch.optim.SGD([
        {'params': speech_embedder.LSTM_stack.parameters(), 'lr': train_config.lr*0.5},
        {'params': speech_embedder.projection.parameters(), 'lr': train_config.lr},
        {'params': ge2e_loss.parameters(), 'lr': train_config.lr*0.01}
    ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.5)

    # create file directory for saving training results.
    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(train_config.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(train_config.final_model_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_config.optim_model_path), exist_ok=True)


    # start train
    speech_embedder.train()
    loss_log = []
    total_loss_log = []
    iteration = 0

    optim_loss = float("inf")
    for e in range(train_config.epochs):
        # Because dataloader drop last batch, so we shuffle all data in case some data will never be used
        train_set.shuffle()
        total_loss = 0
        for batch_id, batch in enumerate(train_loader):

            iteration += 1

            batch = batch.to(device)
            N, M, frames, nmels = batch.shape
            batch = batch.reshape(N*M, frames, nmels)

            # shuffle the training batch
            perm = random.sample(range(0, N*M), N*M)
            unperm = list(perm)
            for i, j in enumerate(perm):
                unperm[j] = i
            batch_shuffle = batch[perm]

            if config.task == "tisv":
                batch_shuffle = batch_shuffle.float()
            
            embeddings = speech_embedder(batch_shuffle)
            embeddings = embeddings[unperm] # un-shuffle batch for similarity
            embeddings = torch.reshape(embeddings, (N, M, embeddings.size(1)))

            #gradient accumulates
            optimizer.zero_grad()
            #get loss, call backward, step optimizer
            
            loss = ge2e_loss(embeddings)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(speech_embedder.parameters(), 3.0) # avoid gradient exploding
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 3.0)
            optimizer.step()

            total_loss += loss
            loss_log.append(loss.item())
            total_loss_log.append(total_loss.item() / (batch_id + 1))

            # print training loss every interval batches
            if (batch_id + 1) % train_config.log_intervals == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(
                                                                                        time.ctime(),
                                                                                        e + 1,
                                                                                        batch_id + 1,
                                                                                        len(train_set) // train_config.N,
                                                                                        iteration, 
                                                                                        loss,
                                                                                        total_loss/(batch_id + 1))
            
                print(mesg)
                with open(train_config.log_file, 'a') as f:
                    f.write(mesg)

        # after finishing one epoch
        scheduler.step()
        
        if optim_loss > (total_loss.item()/(batch_id+1)):
            optim_loss = (total_loss.item()/(batch_id+1))
        	
            print("Save model with optim loss {} in {}".format(optim_loss, train_config.optim_model_path))
        	
            torch.save({'speech_embedder': speech_embedder.state_dict(), 'ge2e_loss': ge2e_loss.state_dict()}, train_config.optim_model_path)

        # save checkpoint every interval epochs
        if (e+1) % train_config.checkpoint_interval == 0:
            speech_embedder.eval().cpu()
            ge2e_loss.eval().cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(e+1) + "_batch_id_" + str(batch_id+1) + ".pth"
            ckpt_model_path = os.path.join(train_config.checkpoint_dir, ckpt_model_filename)
            torch.save({'speech_embedder': speech_embedder.state_dict(), 'ge2e_loss': ge2e_loss.state_dict()}, ckpt_model_path)
            speech_embedder.to(device).train()
            ge2e_loss.to(device).train()

    # save final model
    speech_embedder.eval().cpu()
    ge2e_loss.eval().cpu()
    torch.save({'speech_embedder': speech_embedder.state_dict(), 'ge2e_loss': ge2e_loss.state_dict()}, train_config.final_model_path)
    # save loss log
    torch.save({'loss':loss_log, 'total_loss':total_loss_log}, train_config.loss_log_file)
    print("\nDone, trained model saved at", train_config.loss_log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Restore model from this path.")
    args = parser.parse_args()

    train(args.model_path)