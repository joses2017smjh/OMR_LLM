import torch
from torcheval.metrics.functional import bleu_score

from tqdm import tqdm
import json

from data.MathQA import MathQA
from models.DecoderTransformer import DecoderTransformer
from graph_edit import compute


def argmax_sample(decoder, trg_vocab, device, src_seq, max_steps):

    decoder.eval()

    # start generated sequence with <SOS>
    curr_seq = torch.ones((1,), dtype=int).to(device)

    # run through generation
    for step in range(max_steps):

        # run through decoder
        out = decoder(torch.unsqueeze(src_seq, dim=0), torch.unsqueeze(curr_seq, dim=0))
        out = torch.squeeze(out, dim=0)[-1]

        # make prediction and add to sequence
        pred = torch.unsqueeze(torch.argmax(out), dim=0)
        curr_seq = torch.cat([curr_seq, pred], dim=0)

        # terminate on <EOS>
        if pred == 2:
            break
    
    return curr_seq


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


def compute_bleu(trg_vocab, pred_seq, trg_seq, n_gram=2):

    # clean up predicted and target sequences (strip <SOS>, <EOS>)
    pred_seq = pred_seq[1:-1]
    trg_seq = trg_seq[1:-1]

    # compose sequences into strings
    pred_words = " ".join(trg_vocab.idx2text(pred_seq.to('cpu').numpy()))
    trg_words = " ".join(trg_vocab.idx2text(trg_seq.to('cpu').numpy()))

    # get lengths
    pred_len = len(pred_words)
    trg_len = len(trg_words)

    # revert to unigram if sequences shorter than given n
    n_gram = n_gram if (pred_len >= n_gram and trg_len >= n_gram) else 1

    # compute BLEU score
    score = bleu_score([pred_words], [trg_words], n_gram=n_gram)
    return score.item()


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

    # keep track of BLEU score
    total_bleu_argmax = 0.0
    total_bleu_smart  = 0.0

    # keep track of GED
    ged_smart = 0
    ged_argmax = 0

    # run through samples
    pbar = tqdm(total=num_samples, desc="Test Problems", unit="example")
    for idx in sample_idx:

        # get test source and target sequences
        src_seq, trg_seq = test_set[idx]
        src_seq = src_seq.to(device)
        trg_seq = trg_seq.to(device)

        # autoregressive sampling (argmax vs smart)
        pred_seq_argmax = argmax_sample(decoder=decoder, trg_vocab=trg_vocab, device=device, src_seq=src_seq, max_steps=100)
        pred_seq_smart = smart_sample(decoder=decoder, trg_vocab=trg_vocab, op_dict=op_dict, device=device, src_seq=src_seq, max_ops=100)

        # accuracy bookkeeping
        if torch.equal(trg_seq, pred_seq_argmax):
            correct_argmax += 1
        if torch.equal(trg_seq, pred_seq_smart):
            correct_smart += 1

        # BLEU score bookkeeping
        total_bleu_argmax += compute_bleu(trg_vocab, pred_seq_argmax, trg_seq)
        total_bleu_smart += compute_bleu(trg_vocab, pred_seq_smart, trg_seq)

        # pred_str_argmax = trg_vocab.idx2text(pred_seq_argmax[1:-1].to('cpu').numpy())
        # pred_str_smart = trg_vocab.idx2text(pred_seq_smart[1:-1].to('cpu').numpy())
        # trg_str = trg_vocab.idx2text(trg_seq[1:-1].to('cpu').numpy())

        # ged_smart += compute(trg_str, pred_str_smart)
        # ged_argmax += compute(trg_str, pred_str_argmax)
            
        pbar.update(1)
    

    print("argmax accuracy:\t" + str(correct_argmax/num_samples))
    print("smart accuracy:\t" + str(correct_smart/num_samples))

    print("argmax average BLEU:\t" + str(total_bleu_argmax/num_samples))
    print("smart average BLEU:\t" + str(total_bleu_smart/num_samples))

    # print("argmax average GED: " + str(ged_argmax/num_samples))
    # print("smart average GED: " + str(ged_smart/num_samples))


    pbar.close()
