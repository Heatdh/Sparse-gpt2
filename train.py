import os
import sys
import torch
import random
import argparse
import numpy as np
from GPT2.model import (GPT2LMHeadModel)
from GPT2.utils import load_weight
from GPT2.config import GPT2Config
from GPT2.sample import sample_sequence
from GPT2.encoder import get_encoder


from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer




if __name__ == '__main__':
    encoder = get_encoder()
    config = GPT2Config()
    model = GPT2LMHeadModel(config)

    # Prepare data
    train_iter = WikiText2(split='train')
    tokenizer = get_tokenizer('basic_english')
    data = []
    for line in train_iter:
        data += tokenizer(line)
    data = ' '.join(data)  # Join the tokens into a single string
    data = torch.tensor([encoder.encode(data)])

    print(data.size())  
    print("Encoding done")
    # Train
    model.train()
    model.apply(lambda x: x.half())
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1):
        for i in range(0, data.size(1) - 1, config.n_ctx):
            optimizer.zero_grad()
            loss = model(data[:, i:i+config.n_ctx], labels=data[:, i+1:i+1+config.n_ctx])
            loss.backward()
            optimizer.step()
            print(loss.item())
    model.eval()
    model.apply(lambda x: x.float())
    torch.save(model.state_dict(), 'model.pth')

    
