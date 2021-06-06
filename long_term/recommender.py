import pandas as pd
import numpy as np
import sys
import math
import torch
from torch.autograd import Variable

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
	# create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
	# create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
    	# matrix multiplication
        return (self.user_factors(user)*self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)


if __name__ == "__main__":
    train = sys.argv[1]
    test = sys.argv[2]
    
    print("Load Data ... ", end="", flush=True)
    df_ratings = pd.read_csv("./data-2/"+train, sep='\s+', names=['user','movie','rating','timestamp'])
    df_ratings.drop('timestamp', axis=1, inplace=True)
    df_test = pd.read_csv("./data-2/"+test, sep='\s+', names=['user','movie','rating','timestamp'])
    
    n_users = max(df_ratings.user.unique())
    n_items = max(df_ratings.movie.unique())
    
    model = MatrixFactorization(n_users, n_items, n_factors=20)
    loss_fn = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-6)

    for user, item in zip(users, items):
        # get user, item and rating data
        rating = Variable(torch.FloatTensor([ratings[user, item]]))
        user = Variable(torch.LongTensor([int(user)]))
        item = Variable(torch.LongTensor([int(item)]))

        # predict
        prediction = model(user, item)
        loss = loss_fn(prediction, rating)

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()