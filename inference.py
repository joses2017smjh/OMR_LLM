import torch
from torcheval.metrics.functional import bleu_score

from tqdm import tqdm
import json

from data.MathQA import MathQA
from models.DecoderTransformer import DecoderTransformer


def smart_sample(decoder, trg_vocab, op_dict, device, src_seq, max_ops):

    decoder.eval()

    # keep track of sections of vocab
    operation_range = (2, 56)
    const_range = (56, 90)
    num_ref_range = (90, 190)
    out_ref_range = (190, 290)

    # start generated sequence with <SOS>
    curr_seq = torch.ones((1,), dtype=int).to(device)

    # how many <NUM> tokens in src_seq
    total_num_refs = (src_seq == 4).sum().item()
    
    for op in range(max_ops):
        
        # run through decoder
        out = decoder(torch.unsqueeze(src_seq, dim=0), torch.unsqueeze(curr_seq, dim=0))
        out = torch.squeeze(out, dim=0)[-1]

        # create mask to predict over operations only
        op_mask = torch.zeros_like(out, dtype=torch.bool)
        op_mask[:] = True

        # allow all operations
        op_start, op_end = operation_range
        op_mask[op_start : op_end] = False

        # apply mask
        logits = out.masked_fill(op_mask, float('-inf'))

        # make prediction and add to sequence
        pred = torch.unsqueeze(torch.argmax(logits), dim=0)
        curr_seq = torch.cat([curr_seq, pred], dim=0)

        # terminate on <EOS>
        if pred == 2:
            break
        
        # get operation itself
        pred_op = trg_vocab.idx2word[pred.item()]

        # iterate over number of operation arguments
        for i in range(op_dict[pred_op]):

            # run through decoder
            out = decoder(torch.unsqueeze(src_seq, dim=0), torch.unsqueeze(curr_seq, dim=0))
            out = torch.squeeze(out, dim=0)[-1]

            # create mask to predict over specific entities only
            ent_mask = torch.zeros_like(out, dtype=torch.bool)
            ent_mask[:] = True

            # allow all constants
            const_start, const_end = const_range
            ent_mask[const_start : const_end] = False

            # allow appropriate number references
            num_ref_start, _ = num_ref_range
            ent_mask[num_ref_start : num_ref_start + total_num_refs] = False

            # allow appropriate output references
            out_ref_start, _ = out_ref_range
            ent_mask[out_ref_start : out_ref_start + op] = False

            # apply mask
            logits = out.masked_fill(ent_mask, float('-inf'))

            # make prediction and add to sequence
            pred = torch.unsqueeze(torch.argmax(logits), dim=0)
            curr_seq = torch.cat([curr_seq, pred], dim=0)

    return curr_seq


if __name__ == '__main__':
    
    # open and read JSON file
    with open('data/operations.json', 'r') as file:
        op_dict = json.load(file)

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
    chkpt_path = "./trained_models/decoder_transformer_2025_06_11_18_06_e29"
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

    # randomly sample problems from test dataset
    num_samples = 100
    sample_idx = torch.randperm(len(test_set))[:num_samples]

    # keep track of accuracy
    correct_argmax = 0
    correct_smart = 0
    total_bleu_argmax = 0.0
    total_bleu_smart  = 0.0

    # ask user for input
    user_input = input("Please enter your math word problem: ")

    src_seq = torch.tensor(src_vocab.text2idx(user_input))
    src_seq = src_seq.to(device)

    print(src_vocab.idx2text(src_seq.to('cpu').numpy()))

    # autoregressive sampling (argmax vs smart)
    pred_seq = smart_sample(decoder=decoder, trg_vocab=trg_vocab, op_dict=op_dict, device=device, src_seq=src_seq, max_ops=100)

    pred_str = " ".join(trg_vocab.idx2text(pred_seq.to('cpu').numpy()))
    print(pred_str)