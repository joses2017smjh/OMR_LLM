import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torcheval.metrics.functional import bleu_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from tqdm import tqdm
import json

from data.MathQA import MathQA
from models.DecoderTransformer import DecoderTransformer


def argmax_sample(src_seq, max_steps):
    # start generated sequence with <SOS>
    curr_seq = torch.ones((1,), dtype=int).to(device)

    # run through generation
    for step in range(max_steps):

        out = decoder(torch.unsqueeze(src_seq, dim=0), torch.unsqueeze(curr_seq, dim=0))
        out = torch.squeeze(out, dim=0)
        pred = torch.unsqueeze(torch.argmax(out[-1]), dim=0)

        curr_seq = torch.cat([curr_seq, pred], dim=0)
        if pred == 2:
            break
    
    return curr_seq


def smart_sample(src_seq, max_steps):

    operation_range = (3, 55)
    #number_range = (56, 282)
    number_reference_range = (83, 182)
    output_reference_range = (183, 282)

    # start generated sequence with <SOS>
    curr_seq = torch.ones((1,), dtype=int).to(device)
    
    operands = 0  #track how many operands we expect

    operation_count = 0  #track how many operations we've completed
    numbers = (src_seq == 4).sum().item() # how many <NUM> tokens in src_seq
    
    for step in range(max_steps):
        out = decoder(torch.unsqueeze(src_seq, dim=0), torch.unsqueeze(curr_seq, dim=0))
        out = torch.squeeze(out, dim=0)
        logits = out[-1]

        

        # Apply mask depending on whether we're expecting operands
        mask = torch.zeros_like(logits, dtype=torch.bool)

        mask[:] = True

        if operands > 0: #Want Numbers

            num_start, num_end = number_reference_range
            out_start, out_end = output_reference_range
            
            mask[num_start : num_start+numbers+1] = False
            mask[out_start : out_start+operation_count+1] = False


        else: #Want operation
            op_start, op_end = operation_range
            mask[op_start : op_end+1] = False
            mask[2] = False #allow EOS token
            operation_count += 1

        logits = logits.masked_fill(mask, float('-inf'))


        # Sample the best allowed token
        pred = torch.argmax(logits, dim=-1, keepdim=True)

        curr_seq = torch.cat([curr_seq, pred], dim=0)
        if pred == 2:  # <EOS>
            break

        # Update state
        pred_token = trg_vocab.idx2word[pred.item()]

        if operands > 0:
            operands -= 1
        else:
            operands = operations_dict[pred_token]

    return curr_seq


if __name__ == '__main__':
    # Open and read the JSON file
    with open('data/Operations.json', 'r') as file:
        operations_dict = json.load(file)

    # load MathQA train dataset
    train_set = MathQA(split='train')

    # get source and target vocabs
    src_vocab = train_set.src_vocab
    trg_vocab = train_set.trg_vocab

    # load MathQA test dataset
    test_set = MathQA(split='test', src_vocab=src_vocab)

    # load MathQA validation dataset
    validation_set = MathQA(split='validation', src_vocab=src_vocab)

    # get source and target vocab lengths
    src_vocab_len = len(src_vocab)
    trg_vocab_len = len(trg_vocab)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

    # get checkpoint of best model
    chkpt_path = "./trained_models/decoder_transformer_2025_06_11_14_06_e29"
    chkpt = torch.load(chkpt_path, weights_only=False, map_location=torch.device(device))

    # set model configuration
    model_config = chkpt['model_config']

    # create decoder for generation
    decoder = DecoderTransformer(
        src_vocab_size=src_vocab_len,
        trg_vocab_size=trg_vocab_len,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads']
    )

    # load weights
    decoder.load_state_dict(chkpt['model_state_dict'])
    decoder.to(device)

    # get number of parameters
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print("Model has: " + str(trainable_params) + " trainable parameters")

    # set maximum number of operations to predict
    max_steps = 100

    # randomly sample problems from test dataset
    num_samples = 100
    sample_idx = torch.randperm(len(test_set))[:num_samples]

    # keep track of accuracy
    correct = 0

    # run through samples
    pbar = tqdm(total=num_samples, desc="Test Problems", unit="example")
    for idx in sample_idx:

        # get test source and target sequences
        src_seq, trg_seq = test_set[idx]
        src_seq = src_seq.to(device)
        trg_seq = trg_seq.to(device)

        # pred_seq = argmax_sample(src_seq, max_steps)
        pred_seq = smart_sample(src_seq, max_steps)
        
        if torch.equal(trg_seq, pred_seq):
            correct += 1
        
        pbar.update(1)

        # print("Example " + str(idx.item()))
        # print("Word problem:")
        # word_problem = " ".join(src_vocab.idx2text(src_seq.to('cpu').numpy()))
        # print(word_problem)

        # print("True linear formula")
        # tru_lin_form = " ".join(trg_vocab.idx2text(trg_seq.to('cpu').numpy()))
        # print(tru_lin_form)

        # print("Predicted linear formula")
        # pred_lin_form = " ".join(trg_vocab.idx2text(curr_seq.to('cpu').numpy()))
        # print(pred_lin_form)
        
        # print("")
    
    print("Accuracy: " + str(correct/num_samples))
    pbar.close()