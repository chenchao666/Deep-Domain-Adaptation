import random
import tensorflow as tf
from functools import partial



def shuffle(Data, Label):
    ind = range(Data.shape[0])
    random.shuffle(ind)
    Data = Data[ind, :, :, :]
    Label = Label[ind, :]
    return Data, Label


def KMMD(Xs,Xt):
    sigmas=[1e-3,1e-2,0.1,1,5,10,20,25,30,35,100,1000]
    guassian_kernel=partial(kernel,sigmas=tf.constant(sigmas))
    cost = tf.reduce_mean(guassian_kernel(Xs, Xs))
    cost += tf.reduce_mean(guassian_kernel(Xt, Xt))
    cost -= 2 * tf.reduce_mean(guassian_kernel(Xs, Xt))
    cost = tf.where(cost > 0, cost, 0)
    return cost

def kernel(X, Y, sigmas):
    beta = 1.0/(2.0 * (tf.expand_dims(sigmas,1)))
    dist = Cal_pairwise_dist(X,Y)
    s = tf.matmul(beta, tf.reshape(dist,(1,-1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))



def Cal_pairwise_dist(X,Y):
    norm = lambda x: tf.reduce_sum(tf.square(x), 1)
    dist= tf.transpose(norm(tf.expand_dims(X, 2)-tf.transpose(Y)))
    return dist




def ExpDistrubution(len):
    TargetOuput=random.expovariate(scale=1, size=(1,len))
    return TargetOuput


def WeibullDistribution(len):
    TargetOutput=random.weibullvariate(1,len)
    return TargetOutput



























