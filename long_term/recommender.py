import pandas as pd
import numpy as np
import sys
import math
import torch
from torch.autograd import Variable

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
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
    print("Done")

    print("Matrix Factorization ... ", end="", flush=True)
    n_users = max(df_ratings.user.unique())
    n_movies = max(df_ratings.movie.unique())
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MatrixFactorization(n_users, n_movies, n_factors=10)
    model.to(device)
    loss_fn = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    ratings = np.zeros((n_users+1, n_movies+1)) 
    for row in df_ratings.itertuples(index=False):
        user_id, movie_id, _ = row
        ratings[user_id, movie_id] = row[2]
    
    train_loss = AverageMeter()
    users = df_ratings['user']
    movies = df_ratings['movie']
    for idx, (user, movie) in enumerate(zip(users, movies)):
        optimizer.zero_grad()

        rating = Variable(torch.FloatTensor([ratings[user, movie]])).to(device)
        user = Variable(torch.LongTensor([int(user)])).to(device)
        movie = Variable(torch.LongTensor([int(movie)])).to(device)

        prediction = model(user, movie)
        loss = loss_fn(prediction, rating)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), 1)
        if (idx+1) % 1000 == 0:
            print("Index %d | Train Loss %.4f" % (idx+1, train_loss.avg))

    print("Done")

    print("Test ... ", end="", flush=True)
    result_df = df_test.drop(['timestamp'], axis=1).copy()
    result_df.astype({'user':'int','movie':'int','rating':'float'})
    for idx, row in result_df.iterrows():
        user = Variable(torch.LongTensor([int(row['user'])]))
        movie = Variable(torch.LongTensor([int(row['movie'])]))
        if user > n_users or movie > n_movies:
            result_df.loc[idx, 'rating'] = sum(rating[int(row['user'])]) / n_movies
        else :
            result_df.loc[idx,'rating'] = float(model(user, movie))
     
    result_df.to_csv('./test/'+train[:2]+".base_prediction.txt", sep='\t', index=False, header=False)
    print("Done")