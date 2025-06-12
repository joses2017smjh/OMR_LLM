import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from datasets import load_dataset, concatenate_datasets
import spacy

# heavily inspired by AI 539 Homework 4
# math problem vocabulary
class ProblemVocab:
    def __init__(self, corpus, tokenizer):
        self.tokenizer = tokenizer
        self.word2idx, self.idx2word = self.build_vocab(corpus)

    def __len__(self):
        return len(self.word2idx)

    def is_number(self, string):
        try:
            float(string)
            return True
        except ValueError:
            return False

    def text2idx(self, text):
        tokens = [str(x).strip().lower() if not self.is_number(str(x).replace(',', '')) else '<NUM>' for x in self.tokenizer(text)]
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['<UNK>'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else '<UNK>' for i in idxs]

    def build_vocab(self,corpus):

        cntr = Counter()
        for datapoint in corpus:
            cntr.update( [str(x).strip().lower() for x in self.tokenizer(datapoint)] )

        tokens = [t for t,c in cntr.items() if c >= 30]

        # remove numbers as tokens, genearlize to <NUM>
        for token in tokens:
            if self.is_number(token.replace(',', '')):
                tokens.remove(token)

        word2idx = {t:i+5 for i,t in enumerate(tokens)}
        idx2word = {i+5:t for i,t in enumerate(tokens)}

        # padding token
        word2idx['<PAD>'] = 0
        idx2word[0] = '<PAD>'

        # start of sequence token
        word2idx['<SOS>'] = 1
        idx2word[1] = '<SOS>'

        # end of sequence token
        word2idx['<EOS>'] = 2
        idx2word[2] = '<EOS>'

        # unknown token
        word2idx['<UNK>'] = 3
        idx2word[3] = '<UNK>'

        # number token
        word2idx['<NUM>'] = 4
        idx2word[4] = '<NUM>'

        return word2idx, idx2word

# linear formula vocabulary
class LinearFormulaVocab:
    def __init__(self):
       self.word2idx, self.idx2word = self.build_vocab()
      
    def __len__(self):
       return len(self.word2idx)
    
    def text2idx(self, text):
        return [self.word2idx[t] for t in text]

    def idx2text(self, idxs):
        return [self.idx2word[i] for i in idxs]

    def build_vocab(self):
        word2idx = {}
        idx2word = {}
       
        # padding token
        word2idx['<PAD>'] = 0
        idx2word[0] = '<PAD>'

        # start of sequence token
        word2idx['<SOS>'] = 1
        idx2word[1] = '<SOS>'

        # end of sequence token
        word2idx['<EOS>'] = 2
        idx2word[2] = '<EOS>'

        with open('./data/operation_list.txt') as f:
            operator_list = [l.strip() for l in f if l.strip()]
        with open('./data/constant_list.txt') as f:
            constant_list = [l.strip() for l in f if l.strip()]

        idx = 3
        
        # add valid operators
        for op in operator_list:
            word2idx[op] = idx
            idx2word[idx] = op
            idx += 1

        # add valid constants
        for const in constant_list:
            word2idx[const] = idx
            idx2word[idx] = const
            idx += 1

        # add number references
        for i in range(100):
            word2idx["n"+str(i)] = idx
            idx2word[idx] = "n"+str(i)
            idx += 1
           
        # add output references
        for i in range(100):
            word2idx["#"+str(i)] = idx
            idx2word[idx] = "#"+str(i)
            idx += 1
        
        return word2idx, idx2word


# MathQA dataset loader
class MathQA(Dataset):
    
    def __init__(self, src_vocab=None, split='test'):
      
        # get MathQA split dataset
        print("Loading MathQA " + split + " dataset")
        self.dataset = load_dataset(
            'math_qa',
            'all',
            split=split,
            trust_remote_code=True
        )

        # separate between problems and linear formulas
        temp_problem_data = [x["Problem"] for x in self.dataset]
        temp_linear_formula_data = [x["linear_formula"] for x in self.dataset]

        self.problem_data = []
        self.linear_formula_data = []

        # build a source vocab if needed
        if src_vocab == None:
            print("Building source vocab")
            self.src_vocab = ProblemVocab(temp_problem_data, spacy.load('en_core_web_sm').tokenizer)
        else:
            self.src_vocab = src_vocab

        # create a known target vocab
        self.trg_vocab = LinearFormulaVocab()

        # reformat strings to token tensors
        for i in range(len(temp_problem_data)):
            p = temp_problem_data[i]
            
            x = self.src_vocab.text2idx(p)
            x.insert(0, self.src_vocab.word2idx['<SOS>'])
            x.append(self.src_vocab.word2idx['<EOS>'])

            self.problem_data.append(x)
        
        # reformat strings to token tensors
        for i in range(len(temp_linear_formula_data)):
            lf = temp_linear_formula_data[i]
            lf = lf.replace('.', '_')
            lf = lf.replace(',', ' ').replace('(', ' ').replace(')', ' ').replace('|', ' ').replace('\0', ' ')
            lf = lf.split()

            y = self.trg_vocab.text2idx(lf)
            y.insert(0, self.src_vocab.word2idx['<SOS>'])
            y.append(self.src_vocab.word2idx['<EOS>'])

            self.linear_formula_data.append(y)

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        x = self.problem_data[idx]
        y = self.linear_formula_data[idx]

        return torch.tensor(x), torch.tensor(y)


    @staticmethod
    def pad_collate(batch):

        xx = [item[0] for item in batch]
        yy = [item[1] for item in batch]
        
        x_lens = torch.LongTensor([len(x)-1 for x in xx])
        y_lens = torch.LongTensor([len(y)-1 for y in yy])

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad, yy_pad, x_lens, y_lens