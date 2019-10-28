from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import time
import dill
import json
import copy

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#################################################################################
def gradtovec(grad):
    vec = np.array([])
    le = len(grad)
    for i in range(0, le):
        a = grad[i]
        b = a.numpy()

        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            b = b.reshape(da * db)
        # else:
        #    da = int(a.shape[0])
        #    b.reshape(da)
        vec = np.concatenate((vec, b), axis=0)
    return vec


def vectograd(vec, grad):
    le = len(grad)
    for i in range(0, le):
        a = grad[i]
        b = a.numpy()
        if len(a.shape) == 2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            c = vec[0:da * db]
            c = c.reshape(da, db)
            lev = len(vec)
            vec = vec[da * db:lev]
        else:
            da = int(a.shape[0])
            c = vec[0:da]
            lev = len(vec)
            vec = vec[da:lev]
        grad[i] = 0 * grad[i] + c
    return grad




def quantd(vec, v2, b):
    n = len(vec)
    r = max(abs(vec - v2))
    delta = r / (np.floor(2 ** b) - 1)
    quantv = v2 - r + 2 * delta * np.floor((vec - v2 + r + delta) / (2 * delta))
    return quantv





#########################################################################
# alg=0:QLAG;alg=1:GD; alg=2:rQGD;alg=3:sGD

tic = time.time()
# alg=0 #cl=0.05;alpha=0.02;Iter=5000

nalg = 1;
Iter = 5000;
Iter = 8000;
C = 100;
ck = 0.8;
Iter = 4000;
Loss = np.zeros(Iter)
Cr = np.zeros(Iter)
Gradnorm = np.zeros( Iter)

beta = 0.001;



alg = 0
b = 4;
s = 10;
p = 0.5
alpha = 0.02;
alpah = 0.02
# alpha=ck*alpha
cl = 0.01;
M = 10;
D = 10
(mnist_images, mnist_labels), (mnist_ta, mnist_tb) = tf.keras.datasets.mnist.load_data()
mnist_ta = mnist_ta / 255
Ntr = mnist_images.__len__()
Nte = mnist_ta.__len__()
Mi = int(Ntr / M)
Datatr = M * [0];
for m in range(0, M):
    datr = tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[m * Mi:(m + 1) * Mi, tf.newaxis] / 255, tf.float32),
         tf.cast(mnist_labels[m * Mi:(m + 1) * Mi], tf.int64)))
    datr = datr.batch(Mi)
    Datatr[m] = datr

Datate = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_ta[..., tf.newaxis], tf.float32),
     tf.cast(mnist_tb, tf.int64)))
Datate = Datate.batch(1)

nl = len(mnist_tb)
mnistl = np.eye(10)[mnist_tb]

# Build the model

'''
mnist_model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
'''
regularizer = tf.contrib.layers.l2_regularizer(scale=0.9)
tf.random.set_random_seed(1234)
mnist_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(120, activation=tf.nn.relu,kernel_regularizer=regularizer),
    # tf.keras.layers.Dense(200, activation=tf.nn.relu),
    # tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
    tf.keras.layers.Dense(10)
])
mnist_model.compile(optimizer=tf.train.GradientDescentOptimizer(alpha),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

# mnist_model.compile(metrics='accuracy')

for images, labels in Datatr[1].take(1):
    print("Logits: ", mnist_model(images[0:1]).numpy())

# optimizer = tf.train.AdamOptimizer()
optimizer = tf.train.GradientDescentOptimizer(alpha)

le = len(mnist_model.trainable_variables)
nv = 0
for i in range(0, le):
    a = mnist_model.trainable_variables[i]
    if (len(a.shape) == 2):
        da = int(a.shape[0])
        db = int(a.shape[1])
        nv = nv + da * db
    if (len(a.shape) == 1):
        da = int(a.shape[0])
        nv = nv + da

clock = np.zeros(M)
e = np.zeros(M)
ehat = np.zeros(M)
theta = np.zeros(nv)
thetat = np.zeros(nv)
dtheta = np.zeros((nv, Iter))
# mtheta=np.zeros((M,nv,Iter))
gr = np.zeros((M, nv))
mgr = np.zeros((M, nv))
dL = np.zeros((M, nv))

dsa = np.zeros(nv)
gm = np.zeros((M, nv))
D = 10;
ksi = np.ones((D, D + 1));  # ksi=np.matrix(ksi);
for i in range(0, D + 1):
    if (i == 0):
        ksi[:, i] = np.ones(D);
    if (i <= D and i > 0):
        ksi[:, i] = 1 / i * np.ones(D);
ksi = ck * ksi
# ksi=0.002*ksi


Ind = np.zeros((M, Iter))


loss_history = np.zeros(Iter)
lossfg = np.zeros(Iter)
lossfr = np.zeros(Iter)
lossfr2 = np.zeros(Iter)
grnorm = np.zeros(Iter)

for k in range(0, Iter):
    if k == 0:
        thetat = gradtovec(mnist_model.trainable_variables)
    if k >= 1:
        thetat = var
    me = np.zeros(M)
    var = gradtovec(mnist_model.trainable_variables)
    if (k >= 1):
        dtheta[:, k] = var - theta
    theta = var

    for m in range(0, M):
        for (batch, (images, labels)) in enumerate(Datatr[m].take(1)):
            with tf.GradientTape() as tape:
                logits = mnist_model(images, training=True)
                loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
                for i in range(0, len(mnist_model.trainable_variables)):
                    if i == 0:
                        l2_loss = cl * tf.nn.l2_loss(mnist_model.trainable_variables[i])
                    if i >= 1:
                        l2_loss = l2_loss + cl * tf.nn.l2_loss(mnist_model.trainable_variables[i])

                # l2_loss = tf.losses.get_regularization_loss()
                loss_value = loss_value + l2_loss
            grads = tape.gradient(loss_value, mnist_model.trainable_variables)
            vec = gradtovec(grads)
            bvec = vec

            vec = quantd(vec, mgr[m, :], b)
            gr[m, :] = vec
            dvec = vec - bvec
            e[m] = (dvec.dot(dvec))

        for d in range(0, D):
            if (k - d >= 0):
                if (k <= D):
                    me[m] = me[m] + ksi[d, k] * dtheta[:, k - d].dot(dtheta[:, k - d])
                if (k > D):
                    me[m] = me[m] + ksi[d, D] * dtheta[:, k - d].dot(dtheta[:, k - d])
        dL[m, :] = gr[m, :] - mgr[m, :]
        if ((dL[m, :].dot(dL[m, :])) >= (1 / (alpha ** 2 * M ** 2)) * me[m] + 3 * (e[m] + ehat[m]) or clock[
            m] == C):
            Ind[m, k] = 1

        if (Ind[m, k] == 1):
            mgr[m, :] = gr[m, :]
            ehat[m] = e[m]
            clock[m] = 0
            dsa = dsa + dL[m, :]
        if (Ind[m, k] == 0):
            clock[m] = clock[m] + 1

        if m == 0:
            g = grads
            loss = loss_value.numpy() / M
        else:
            g = [a + b for a, b in zip(g, grads)]
            loss = loss + loss_value.numpy() / M
    lossfr2[k] = l2_loss
    # lossfg[k]=loss
    lossfr[k] = 0
    for i in range(0, len(mnist_model.trainable_variables)):
        # loss= loss+ cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
        lossfr[k] = lossfr[k] + cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
    loss_history[k] = loss

    # dsa=dsa-beta/alpha*(theta-thetat)
    ccgrads = vectograd(dsa, grads)
    # grr = copy.deepcopy(mnist_model.trainable_variables)
    # grr2 = [c * cl * 2 for c in grr]
    # ccgrads = [a + b for a, b in zip(cgrads, grr2)]
    # ccgrads=cgrads
    for i in range(0, len(ccgrads)):
        grnorm[k] = grnorm[k] + tf.nn.l2_loss(ccgrads[i]).numpy()
    optimizer.apply_gradients(zip(ccgrads, mnist_model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())

acc = mnist_model.evaluate(mnist_ta, mnistl)
Acc= acc[1]
lossfrv = loss_history - lossfg
plt.plot(loss_history)
plt.xlabel('Iterations#')
plt.ylabel('Loss [entropy]');
plt.show()

deloss = np.array(loss_history)
ll = len(deloss)
lossopt = deloss[ll - 1]
deloss = deloss - lossopt





plt.figure();
plt.stem(Ind[0, :].T);
plt.show();
plt.figure();
plt.stem(Ind[5, :].T);
plt.show();
plt.figure();
plt.stem(Ind[9, :].T);
plt.show();

Inds = sum(Ind)
commrou = np.zeros(Iter)
for i in range(0, Iter):
    if i == 0:
        commrou[i] = Inds[i]
    if i >= 1:
        commrou[i] = commrou[i - 1] + Inds[i]
Loss= loss_history
Cr= commrou
Gradnorm= grnorm


toc = time.time()
runtime = toc - tic

plt.figure();
plt.plot(Cr, Loss, 'r', lw=3, label='LAG');

#plt.yscale('log')
plt.xlabel('Number of communications(uploads)');
plt.ylabel('Loss function');
plt.legend();
plt.show();

reloss = Iter * [0]
for i in range(0, Iter):
    reloss[i] = loss_history[i]
reloss
'''
with open('loss6000.txt','w') as f:
    json.dump(reloss, f)
with open('loss6000.txt','r') as f:
    loss6=json.load(f)
'''
LossGradrou = np.concatenate((Loss, Gradnorm, Cr), axis=0)
'''
if b == 8:
    np.savetxt('eLossGradrouAllQpltb8a2ck.txt', LossGradrou)
    LossGradrou = np.loadtxt('eLossGradrouAllQpltb8a2ck.txt')
if b == 4:
    np.savetxt('eLossGradrouAllQpltb4ck.txt', LossGradrou)
    LossGradrou = np.loadtxt('eLossGradrouAllQpltb4ck.txt')
'''

float = 32;
Bit = np.zeros( Iter)
Bit= b * nv * Cr;

minl = Loss.min()
float = 32
deLoss = Loss - minl



k = 0
while (deLoss[k] > 10 ** (-6) and k < Iter - 1):
    k = k + 1
index = k

'''
[Cr[0, int(index[0])], Cr[1, int(index[1])], Cr[2, int(index[2])], Cr[3, int(index[3])]]  # communication #
[b * nv * Cr[0, int(index[0])], float * nv * Cr[1, int(index[1])], b * nv * Cr[2, int(index[2])],
 float * nv * Cr[3, int(index[3])]]
'''

plt.figure();
plt.plot(deLoss, 'r', lw=3, label='LAG');
plt.ylim(10 ** (-6), 10 ** 1);
plt.xlim(-100, 3000);
plt.yscale('log');
#plt.xscale('log')
plt.xlabel('Number of iterations');
plt.ylabel('Loss function');
plt.legend();
plt.show();

plt.figure();
plt.plot(Cr, deLoss, 'r', lw=3, label='LAG');
plt.ylim(10 ** (-6), 10 ** 1);

plt.yscale('log');
plt.xscale('log')
plt.xlabel('Number of communications(uploads)');
plt.ylabel('Loss function');
plt.legend();
plt.show();

plt.figure();
plt.plot(b * Cr, deLoss, 'r', lw=2.5, label='LAG');
plt.ylim(10 ** (-6), 10 ** 1);

plt.yscale('log');
plt.xscale('log')
plt.xlabel('Number of bits');
plt.ylabel('Loss function');
plt.legend();
plt.show();

