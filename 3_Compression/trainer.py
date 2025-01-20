import numpy as np
import torch
from tqdm import tqdm


def train(model, loss_function, optimizer, scheduler, train_dataloader, test_dataloader, num_epochs, device):
    loss_logs = {
        'train' : [],
        'test' : []
    }
    model.to(device)
    
    pb = tqdm(range(num_epochs))
    for epoch in pb:
        model.train()
        train_losses = []
        for x in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            y = model(x)
            loss = loss_function(y, x)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.detach().cpu().numpy())
        scheduler.step()
        
        model.eval()
        test_losses = []
        with torch.no_grad():
            for x in test_dataloader:
                x = x.to(device)
                y = model(x)
                loss = loss_function(y, x)
                
                test_losses.append(loss.detach().cpu().numpy())
                
        loss_logs['train'].append(np.mean(train_losses))
        loss_logs['test'].append(np.mean(test_losses))
        
        pb.set_postfix({'train' : loss_logs['train'][-1], 'test' : loss_logs['test'][-1]})
        
    return loss_logs