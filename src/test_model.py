import os
import torch
import argparse

from torch.utils.data import DataLoader

from parse_config import config
from data_load import SpeechDataset
from model_network import SpeechEmbedder
from utils import get_similarity_eva, get_EER

def test(model_path):

    if config.task == "tdsv":
        test_config = config.test.TD_SV_test
    else:
        test_config = config.test.TI_SV_test

    if model_path == None:
        model_path = test_config.final_model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(os.path.dirname(test_config.EER_log_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_config.log_file), exist_ok=True)

    # load test dataset and dataloader
    test_set = SpeechDataset()
    test_loader = DataLoader(test_set, batch_size=test_config.N, shuffle=False, drop_last=True)

    # load the model
    speech_embedder = SpeechEmbedder().to(device) # construct model
    speech_embedder.load_state_dict(torch.load(model_path)['speech_embedder']) # restore model
    speech_embedder.eval()
    
    print("successfully load model")

    avg_EER = 0
    batch_avg_EER_log = []
    avg_EER_log = []
    for e in range(test_config.epochs):
        # Because dataloader drop last batch, so we shuffle all data in case some data will never be used
        test_set.shuffle()
        batch_avg_EER = 0
        for batch_id, batch in enumerate(test_loader):

            batch = batch.to(device)
            N, M, frames, nmels = batch.shape
            enrollment_batch, evaluation_batch = torch.split(batch, M//2, dim=1)
            
            enrollment_batch = enrollment_batch.reshape(N * M//2, frames, nmels).float()
            evaluation_batch = evaluation_batch.reshape(N * (M-M//2), frames, nmels).float()
            
            enrollment_embedding = speech_embedder(enrollment_batch)
            evaluation_embedding = speech_embedder(evaluation_batch)
            
            enrollment_embedding = enrollment_embedding.reshape(N, M//2, -1)
            evaluation_embedding = evaluation_embedding.reshape(N, M-M//2, -1)

            similarity = get_similarity_eva(enrollment_embedding, evaluation_embedding)

            EER, FRR, FAR, thresh =  get_EER(similarity)
            batch_avg_EER += EER
            mesg = "\nepoch %d batch_id %d: EER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (e+1, batch_id+1, EER, thresh, FAR, FRR)
            print(mesg)
            if (batch_id+1) % test_config.log_intervals == 0:
                with open(test_config.log_file, 'a') as f:
                        f.write(mesg)

            batch_avg_EER = batch_avg_EER / (batch_id + 1)
            batch_avg_EER_log.append(batch_avg_EER)
            avg_EER += batch_avg_EER
            avg_EER_log.append(avg_EER/(e+1))

    avg_EER = avg_EER / test_config.epochs
    avg_EER_log.append(avg_EER)
    mesg = "\n EER across {0} epochs: {1:.4f}".format(test_config.epochs, avg_EER)
    print(mesg)
    with open(test_config.log_file, 'w') as f:
        f.write(mesg)

    EER_log = {"batch_avg_EER_log": batch_avg_EER_log, "avg_EER_log": avg_EER_log}
    torch.save(EER_log, test_config.EER_log_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Restore model from this path.")
    args = parser.parse_args()

    test(args.model_path)

