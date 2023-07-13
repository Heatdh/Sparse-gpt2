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
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 4
    num_batches = data.size(1) // batch_size
    num_epochs = 10
    seq_len = 128  # Define the sequence length

    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch in range(0, data.size(1) - seq_len, seq_len):
            inputs = data[:, batch:batch+seq_len]
            targets = data[:, batch+1:batch+seq_len+1]
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            outputs = outputs[0]  # Access the tensor within the tuple
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss}")

        



    
