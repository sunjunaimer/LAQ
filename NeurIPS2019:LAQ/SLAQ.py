from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import time
import dill
import json
import copy
import math

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.setrecursionlimit(6000)
#################################################################################
def gradtovec(grad):
    vec=np.array([])
    le=len(grad)
    for i in range(0,le):
        a=grad[i]
        b = a.numpy()

        if len(a.shape)==2:
            da = int(a.shape[0])
            db = int(a.shape[1])
            b=b.reshape(da*db)
       # else:
        #    da = int(a.shape[0])
        #    b.reshape(da)
        vec=np.concatenate((vec,b),axis=0)
    return vec

def vectograd(vec,grad):
    le=len(grad)
    for i in range(0,le):
        a=grad[i]
        b=a.numpy()
        if len(a.shape)==2:
            da=int(a.shape[0])
            db=int(a.shape[1])
            c=vec[0:da*db]
            c=c.reshape(da,db)
            lev=len(vec)
            vec=vec[da*db:lev]
        else:
            da=int(a.shape[0])
            c=vec[0:da]
            lev = len(vec)
            vec = vec[da:lev]
        grad[i]=0*grad[i]+c
    return grad


'''
def quantr(vec,s):
    norv=vec.dot(vec)
    norv=norv**(0.5)
    #norv=1
    norvec=vec/norv
    le=len(vec)
    q=np.zeros(le)
    num=0
    nbit=0
    for i in range(0,le):
        k=0
        sign=-1
        if norvec[i]>=0:
            sign=1
        while(abs(norvec[i])>=k/s):
            k=k+1
        rr=np.random.uniform(0,1)
        if (rr>k-abs(norvec[i]/(1/s))):
            q[i]=sign*(k-1)
        else:]
            q[i]=sign*k
        if q[i]!=0:
            num=num+1
        tem=abs(q[i])
        while (tem>=2):
            tem = math.log(tem, 2)
            nbit=nbit+tem

    quantv=q/s*norv
    return quantv,num,nbit
'''


def quantd(vec,v2,b):
    n=len(vec)
    r=max(abs(vec-v2))
    delta=r/(np.floor(2**b)-1)
    quantv=v2-r+2*delta*np.floor((vec-v2+r+delta)/(2*delta))
    return quantv








#########################################################################
#alg=0:QLAG;alg=1:GD; alg=2:rQGD;alg=3:sGD
float=32
tic=time.time()
#alg=0 #cl=0.05;alpha=0.02;Iter=5000
minibatch=500
nalg=1;Iter=1500;C=100;ec=0.01;ck=0.8;ec=0.5;ade=0.2;#ck=1
Loss=np.zeros(Iter)
Cr=np.zeros(Iter)
Gradnorm=np.zeros(Iter)
#Acc=np.zeros(nalg)
Ncs=np.zeros(Iter)   #communication of sparsification
Ncr=np.zeros(Iter)
Nbitr=np.zeros(Iter)
NNbitr=np.zeros(Iter)
NNcs=np.zeros(Iter)
Iter=1500

b=3;s=10000;p=0.5
alpha=0.008;alpha=0.008;
cl=0.01;
M=10;D=10
(mnist_images, mnist_labels), (mnist_ta,mnist_tb)= tf.keras.datasets.mnist.load_data()
mnist_ta=mnist_ta/255
Ntr=mnist_images.__len__()
Nte=mnist_ta.__len__()
Mi=int(Ntr/M)
Datatr=M*[0];
for m in range(0,M):
    datr=tf.data.Dataset.from_tensor_slices(
        (tf.cast(mnist_images[m*Mi:(m+1)*Mi,tf.newaxis]/255, tf.float32),
        tf.cast(mnist_labels[m*Mi:(m+1)*Mi],tf.int64)))
    datr=datr.batch(minibatch)
    Datatr[m]=datr



Datate = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_ta[...,tf.newaxis], tf.float32),
   tf.cast(mnist_tb,tf.int64)))
Datate=Datate.batch(1)

nl=len(mnist_tb)
mnistl=np.eye(10)[mnist_tb]

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

#tf.keras.layers.Dense(200, activation=tf.nn.relu),

  #tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
  tf.keras.layers.Dense(10)
])
mnist_model.compile(optimizer=tf.train.GradientDescentOptimizer(alpha),
              loss='categorical_crossentropy',
              metrics=['accuracy'])





#mnist_model.compile(metrics='accuracy')

for images,labels in Datatr[1].take(1):
  print("Logits: ", mnist_model(images[0:1]).numpy())

#optimizer = tf.train.AdamOptimizer()
optimizer=tf.train.GradientDescentOptimizer(alpha)



le=len(mnist_model.trainable_variables)
nv=0
for i in range (0,le):
    a =mnist_model.trainable_variables[i]
    if (len(a.shape)==2):
        da=int(a.shape[0])
        db=int(a.shape[1])
        nv=nv+da*db
    if (len(a.shape)==1):
        da=int(a.shape[0])
        nv=nv+da


clock = np.zeros(M)
e = np.zeros(M)
ehat = np.zeros(M)
theta=np.zeros(nv)
dtheta=np.zeros((nv,D))
#mtheta=np.zeros((M,nv,Iter))
gr=np.zeros((M,nv))
mgr=np.zeros((M,nv))
dL=np.zeros((M,nv))

dsa=np.zeros(nv)
gm=np.zeros((M,nv))
D=10;ksi=np.ones((D,D+1));#ksi=np.matrix(ksi);
for i in range(0,D+1):
    if (i==0):
        ksi[:,i]=np.ones(D);
    if (i<=D and i>0):
        ksi[:, i] = 1/i*np.ones(D);
ksi=ck*ksi


Ind=np.zeros((M,Iter))



loss_history = np.zeros(Iter)
lossfg=np.zeros(Iter)
lossfr=np.zeros(Iter)
lossfr2=np.zeros(Iter)
grnorm=np.zeros(Iter)




for k in range(0,Iter):
    me=np.zeros(M)
    var=gradtovec(mnist_model.trainable_variables)
    if (k>=1):
        #dtheta[:,k]=var-theta
        dtheta[:, 0:D - 1] = dtheta[:, 1:D];
        dtheta[:, D - 1] = var - theta
    theta = var

    for m in range(0,M):
        Datatr[m] = Datatr[m].shuffle(10)
        for (batch, (images,labels)) in enumerate(Datatr[m].take(1)):
            with tf.GradientTape() as tape:
                logits=mnist_model(images, training=True)
                loss_value = tf.losses.sparse_softmax_cross_entropy(labels, logits)
                for i in range(0,len(mnist_model.trainable_variables)):
                    if i==0:
                        l2_loss=cl*tf.nn.l2_loss(mnist_model.trainable_variables[i])
                    if i>=1:
                        l2_loss=l2_loss+cl*tf.nn.l2_loss(mnist_model.trainable_variables[i])

                #l2_loss = tf.losses.get_regularization_loss()
                loss_value=loss_value+l2_loss
            grads=tape.gradient(loss_value, mnist_model.trainable_variables)
            vec=gradtovec(grads)
            bvec=vec

            vec=quantd(vec,mgr[m,:],b)

            gr[m,:]=vec
            dvec = vec - bvec
            e[m] = (dvec.dot(dvec))


        NNbitr[k]=NNbitr[k]+Nbitr[k]
        NNcs[k]=NNcs[k]+Ncs[k]


        for d in range(1,D):
            if (k-d>=0):
                if (k<=D):
                    me[m]=me[m]+ksi[d,k]*dtheta[:,k-d].dot(dtheta[:,k-d])
                if (k>D):
                    me[m]=me[m]+ksi[d,D]*dtheta[:,D-d].dot(dtheta[:,D-d])
        dL[m,:]=gr[m,:]-mgr[m,:]
        if ((dL[m,:].dot(dL[m,:]))>=(1/(alpha**2*M**2))*me[m]+ec*3*(e[m]+ehat[m])+ade or clock[m]==C):
            Ind[m,k]=1

        if (Ind[m,k]==1):
            mgr[m,:]=gr[m,:]
            ehat[m] = e[m]
            clock[m] = 0
            dsa=dsa+dL[m,:]

        if (Ind[m, k] == 0):
            clock[m]=clock[m]+1

        if m==0:
            g=grads
            loss=loss_value.numpy()/M
        else:
            g=[a+b for a,b in zip(g,grads)]
            loss=loss+loss_value.numpy()/M
    lossfr2[k]=l2_loss
    #lossfg[k]=loss
    lossfr[k] = 0
    for i in range(0, len(mnist_model.trainable_variables)):
        #loss= loss+ cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
        lossfr[k]=lossfr[k]+cl * np.linalg.norm(mnist_model.trainable_variables[i].numpy()) ** 2
    loss_history[k]=loss

    ccgrads=vectograd(dsa, grads)
    #grr = copy.deepcopy(mnist_model.trainable_variables)
    #grr2 = [c * cl * 2 for c in grr]
    #ccgrads = [a + b for a, b in zip(cgrads, grr2)]
    #ccgrads=cgrads
    for i in range(0,len(ccgrads)):
        grnorm[k]=grnorm[k]+tf.nn.l2_loss(ccgrads[i]).numpy()
    optimizer.apply_gradients(zip(ccgrads, mnist_model.trainable_variables),
                              global_step=tf.train.get_or_create_global_step())

acc=mnist_model.evaluate(mnist_ta,mnistl)
Acc=acc[1]
lossfrv=loss_history-lossfg
plt.plot(loss_history)
plt.xlabel('Iterations#')
plt.ylabel('Loss [entropy]');
plt.show()

deloss=np.array(loss_history)
ll=len(deloss)
lossopt=deloss[ll-1]
deloss=deloss-lossopt
plt.plot(deloss)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
plt.yscale('log')
plt.show()

plt.plot(Acc)
plt.xlabel('Batch #')
plt.ylabel('Accuracy')
plt.show()
'''
s = 0;acc = 0
for ba, (ta, tb) in enumerate(Datate.take(10000)):
  logit = mnist_model(ta[0:1]).numpy()
  dig = logit.argmax()
  if (dig == tb.numpy()[0]):
      acc = acc + 1
  s = s + 1
acc=acc/s
'''




datatesta = tf.data.Dataset.from_tensor_slices(
  tf.cast(mnist_ta[...,tf.newaxis]/255, tf.float32))

datatestb = tf.data.Dataset.from_tensor_slices(
   tf.cast(mnist_tb,tf.int64))

toc=time.time()
runtime=toc-tic
plt.figure();plt.stem(Ind[0,:].T);plt.show();
plt.figure();plt.stem(Ind[5,:].T);plt.show();
plt.figure();plt.stem(Ind[9,:].T);plt.show();


Inds = sum(Ind)
commrou = np.zeros(Iter)
for i in range(0, Iter):
    if i == 0:
        commrou[i] = Inds[i]
    if i >= 1:
        commrou[i] = commrou[i - 1] + Inds[i]

Loss = loss_history
Cr = commrou

Gradnorm=grnorm



tnbitr=np.zeros(Iter)
tnbits=np.zeros(Iter)
for i in range(0,Iter):
    if i==0:
        tnbitr[i]=NNbitr[i]
        tnbits[i]=NNcs[i]*float
    if i>=1:
        tnbitr[i]=tnbitr[i-1]+NNbitr[i]
        tnbits[i] = tnbits[i-1]+NNcs[i]  * float

plt.figure();plt.plot(Cr[0],Loss[0],'r',lw=3,label='QLAG');plt.yscale('log')
plt.xlabel('Number of communications(uploads)');plt.ylabel('Loss function');
plt.legend();plt.show();




reloss=Iter*[0]
for i in range(0,Iter):
    reloss[i]=loss_history[i]
reloss
'''
with open('loss6000.txt','w') as f:
    json.dump(reloss, f)
with open('loss6000.txt','r') as f:
    loss6=json.load(f)
'''

Bit=np.zeros(Iter)
Bit=b*nv*Cr;
LossGradrou=np.concatenate((Loss,Gradnorm,Cr,Bit),axis=0)
'''
if b==20:
    np.savetxt('sgdLossGradrouAllQpltb20.txt',LossGradrou)
    LossGradrou=np.loadtxt('sgdLossGradrouAllQpltb20.txt')
if b==8:
    np.savetxt('sgdLossGradrouAllQpltb8L20a.txt', LossGradrou)
    LossGradrou = np.loadtxt('sgdLossGradrouAllQpltb8L20.txt')

if b==3:
    np.savetxt('sgdLossGradrouAllQpltb3L.txt', LossGradrou)
    LossGradrou = np.loadtxt('sgdLossGradrouAllQpltb3L.txt')
'''


minl=Loss.min()
float=32
deLoss=Loss-minl

plt.figure();plt.plot(deLoss,'r',lw=3,label='SLAQ');plt.yscale('log');plt.xscale('log')
plt.xlabel('Number of iterations');plt.ylabel('Loss function');
plt.legend();plt.show();

plt.figure();plt.plot(Cr,deLoss,'r',lw=3,label='QLAG');plt.yscale('log');plt.xscale('log')
plt.xlabel('Number of communications(uploads)');plt.ylabel('Loss function');
plt.legend();plt.show();

plt.figure();plt.plot(b*nv*Cr,deLoss,'r',lw=3,label='SLAQ');plt.yscale('log');plt.xscale('log')
plt.xlabel('Number of bits');plt.ylabel('Loss function');
plt.legend();plt.show();


'''

plt.figure();plt.plot(Gradnorm[0],'r',lw=3,label='QLAG');plt.plot(Gradnorm[1],'b',lw=2,linestyle=':',label='SGD');
plt.plot(Gradnorm[2],'g',lw=3,label='QSGD');plt.plot(Gradnorm[3],'m',lw=2,linestyle=':',label='SSGD');plt.yscale('log');
plt.xlabel('Number of iterations');plt.ylabel('Gradient norm');
plt.legend();plt.show();

plt.figure();plt.plot(Cr[0],Gradnorm[0],'r',lw=3,label='QLAG');plt.plot(Cr[1],Gradnorm[1],'b',lw=2,linestyle=':',label='SGD');
plt.plot(Cr[2],Gradnorm[2],'g',lw=3,label='QSGD');plt.plot(Cr[3],Gradnorm[3],'m',lw=2,linestyle=':',label='SSGD');plt.yscale('log');
plt.xlabel('Number of communications(uploads)');plt.ylabel('Gradient norm');
plt.legend();plt.show();

plt.figure();plt.plot(b*nv*Cr[0],Gradnorm[0],'r',lw=3,label='QLAG');plt.plot(float*nv*Cr[1],Gradnorm[1],'b',lw=2,linestyle=':',label='SGD');
plt.plot(tnbitr,Gradnorm[2],'g',lw=3,label='QSGD');plt.plot(tnbits,Gradnorm[3],'m',lw=2,linestyle=':',label='SSGD');plt.yscale('log');
plt.xlabel('Number of bits');plt.ylabel('Gradient norm');
plt.legend();plt.show();

[b*nv*Cr[0,Iter-1],float*nv*Cr[1,Iter-1],tnbitr[Iter-1],tnbits[Iter-1]]











plt.figure();plt.plot(Loss[0],'r',lw=2.5,label='QCSGD');plt.plot(Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');
plt.xlabel(r'Number of iterations',fontsize=17);plt.ylabel(r'$f$',fontsize=13.5);
plt.grid(True, which="both",ls=':')
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdlossiter.eps');plt.savefig('figures2/sgdlossiter.pdf');
plt.show();



plt.figure();plt.plot(Cr[0],Loss[0],'r',lw=2.5,label='QCSGD');plt.plot(Cr[1],Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Cr[2],Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Cr[3],Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');#plt.yscale('log');#plt.xscale('log')
plt.xlabel(r'Number of communications',fontsize=17);plt.ylabel(r'$f$',fontsize=13.5);
plt.grid(True, which='both',ls=':')
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdlosscom.eps');plt.savefig('figures2/sgdlosscom.pdf')
plt.show();


plt.figure();fig,ax=plt.subplots();plt.rc('text', usetex=True)
plt.plot(Bit[0],Loss[0],'r',lw=2.5,label='QCSGD');plt.plot(Bit[1],Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Bit[2],Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Bit[3],Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');#plt.yscale('log');#plt.xscale('log')
plt.grid(True, which='both',ls=':')
plt.xlabel(r'Number of bits',fontsize=17);plt.ylabel(r'$f$',fontsize=13.5);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
#ax.set(ylabel=r'$\mathcal{L}-\mathcal{L}^*$') #||\nabla\mathcal{L}(\theta^k)||
#ax.yaxis.label.set_size(15)
plt.savefig('figures2/sgdlossbit.eps');plt.savefig('figures2/sgdlossbit.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Gradnorm[0],'r',lw=2.5,label='QCSGD');plt.plot(Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of iterations',fontsize=16);plt.ylabel(r'$||\nabla f||$',fontsize=13.5);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormiter.eps');plt.savefig('figures2/sgdnormiter.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Cr[0],Gradnorm[0],'r',lw=2.5,label='QCSGD');plt.plot(Cr[1],Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Cr[2],Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Cr[3],Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of communications',fontsize=16);plt.ylabel(r'$||\nabla f||$',fontsize=13.5);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormcom.eps');plt.savefig('figures2/sgdnormcom.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Bit[0],Gradnorm[0],'r',lw=2.5,label='QCSGD');plt.plot(Bit[1],Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Bit[2],Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Bit[3],Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of bits',fontsize=16);plt.ylabel(r'$||\nabla f||$',fontsize=13.5);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormbit.eps');plt.savefig('figures2/sgdnormbit.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.rc('text', usetex=True)
plt.plot(b*nv*Cr[0],deLoss[0],'r',lw=2.5,label='QCSGD');plt.plot(float*nv*Cr[1],deLoss[1],'b',lw=2.5,label='SGD');
plt.plot(b*nv*Cr[2],deLoss[2],'g',lw=2.5,label='QSGD');plt.plot(float*nv*Cr[3],deLoss[3],'m',lw=2.5,label='SSGD');plt.yscale('log');plt.xscale('log')
plt.grid(True, which='both',ls=':')
plt.xlabel(r'Number of bits',fontsize=17);plt.ylabel(r'$f$',fontsize=13.5);
plt.legend(fontsize=16);plt.ylim(10**(-6),10**1);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
#ax.set(ylabel=r'$\mathcal{L}-\mathcal{L}^*$') #||\nabla\mathcal{L}(\theta^k)||
#ax.yaxis.label.set_size(15)
plt.savefig('figures2/sgdlossbit.eps');plt.savefig('figures2/sgdlossbit.pdf')
plt.show();
# Restore the model's state
#E_model.load_weights('my_model.h5')

#E_model.trainable_variabels=mnist_model.trainable_variables
#E_model.non_trainable_variables=mnist_model.non_trainable_variables

#E_model.set_weights(mnist_model.get_weights())
#a=E_model.evaluate(mnist_ta,mnist_tb)














plt.figure();plt.plot(Loss[0],'r',lw=2.5,label='SLAQ');plt.plot(Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');
plt.xlabel(r'Number of iterations',fontsize=17);plt.ylabel('Loss~' +r'($f$)',fontsize=17);
plt.grid(True, which="both",ls=':')
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdlossiter.eps');plt.savefig('figures2/sgdlossiter.pdf');
plt.show();



plt.figure();plt.plot(Cr[0],Loss[0],'r',lw=2.5,label='SLAQ');plt.plot(Cr[1],Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Cr[2],Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Cr[3],Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');#plt.yscale('log');#plt.xscale('log')
plt.xlabel(r'Number of communications',fontsize=17);plt.ylabel('Loss~' +r'($f$)',fontsize=17);
plt.grid(True, which='both',ls=':')
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdlosscom.eps');plt.savefig('figures2/sgdlosscom.pdf')
plt.show();


plt.figure();fig,ax=plt.subplots();plt.rc('text', usetex=True)
plt.plot(Bit[0],Loss[0],'r',lw=2.5,label='SLAQ');plt.plot(Bit[1],Loss[1],'b',lw=2.5,label='SGD');
plt.plot(Bit[2],Loss[2],'orange',linestyle='--',lw=2.5,label='QSGD');plt.plot(Bit[3],Loss[3],'g',linestyle=':',lw=2.5,label='SSGD');#plt.yscale('log');#plt.xscale('log')
plt.grid(True, which='both',ls=':')
plt.xlabel(r'Number of bits',fontsize=17);plt.ylabel('Loss~' +r'($f$)',fontsize=17);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
#ax.set(ylabel=r'$\mathcal{L}-\mathcal{L}^*$') #||\nabla\mathcal{L}(\theta^k)||
#ax.yaxis.label.set_size(15)
plt.savefig('figures2/sgdlossbit.eps');plt.savefig('figures2/sgdlossbit.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Gradnorm[0],'r',lw=2.5,label='SLAQ');plt.plot(Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of iterations',fontsize=16);plt.ylabel('Gradient norm~'+r'$(||\nabla f||)$',fontsize=17);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormiter.eps');plt.savefig('figures2/sgdnormiter.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Cr[0],Gradnorm[0],'r',lw=2.5,label='SLAQ');plt.plot(Cr[1],Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Cr[2],Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Cr[3],Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of communications',fontsize=16);plt.ylabel('Gradient norm~'+r'$(||\nabla f||)$',fontsize=17);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormcom.eps');plt.savefig('figures2/sgdnormcom.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.plot(Bit[0],Gradnorm[0],'r',lw=2.5,label='SLAQ');plt.plot(Bit[1],Gradnorm[1],'b',lw=2.5,label='SGD');
plt.plot(Bit[2],Gradnorm[2],'g',lw=2.5,label='QSGD');plt.plot(Bit[3],Gradnorm[3],'m',lw=2.5,label='SSGD');plt.yscale('log');
plt.grid(True, which='both',ls=':')
plt.xlabel('Number of bits',fontsize=16);plt.ylabel('Gradient norm~'+r'$(||\nabla f||)$',fontsize=17);
plt.legend(fontsize=16);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
plt.savefig('figures2/sgdnormbit.eps');plt.savefig('figures2/sgdnormbit.pdf')
plt.show();

plt.figure();fig,ax=plt.subplots();plt.rc('text', usetex=True)
plt.plot(b*nv*Cr[0],deLoss[0],'r',lw=2.5,label='SLAQ');plt.plot(float*nv*Cr[1],deLoss[1],'b',lw=2.5,label='SGD');
plt.plot(b*nv*Cr[2],deLoss[2],'g',lw=2.5,label='QSGD');plt.plot(float*nv*Cr[3],deLoss[3],'m',lw=2.5,label='SSGD');plt.yscale('log');plt.xscale('log')
plt.grid(True, which='both',ls=':')
plt.xlabel(r'Number of bits',fontsize=17);plt.ylabel(r'$f$',fontsize=13.5);
plt.legend(fontsize=16);plt.ylim(10**(-6),10**1);
plt.rc('xtick',labelsize=18);plt.rc('ytick',labelsize=17)
#ax.set(ylabel=r'$\mathcal{L}-\mathcal{L}^*$') #||\nabla\mathcal{L}(\theta^k)||
#ax.yaxis.label.set_size(15)
plt.savefig('figures2/sgdlossbit.eps');plt.savefig('figures2/sgdlossbit.pdf')
plt.show();

'''





#a=E_model.evaluate(mnist_ta,mnistl)


#cf=np.array([1486.18,1497.47,271.02,406.90,361.90,326.22,282.77,271.56,1525.71,276.77])


''''
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import numpy as np
import time
import dill
import json
import copy
b=3;float=32;alpha=0.02;nalg=4;Iter=1000
mnist_model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      #tf.keras.layers.Dense(512, activation=tf.nn.relu),
      #tf.keras.layers.Dropout(0.2),
    #tf.keras.layers.Dense(120, activation=tf.nn.relu,kernel_regularizer=regularizer),
    tf.keras.layers.Dense(200, activation=tf.nn.relu),
      #tf.keras.layers.Dense(10, kernel_regularizer=regularizer)
      tf.keras.layers.Dense(10)
    ])
mnist_model.compile(optimizer=tf.train.GradientDescentOptimizer(alpha),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
le=len(mnist_model.trainable_variables)
nv=0
for i in range (0,le):
    a =mnist_model.trainable_variables[i]
    if (len(a.shape)==2):
        da=int(a.shape[0])
        db=int(a.shape[1])
        nv=nv+da*db
    if (len(a.shape)==1):
        da=int(a.shape[0])
        nv=nv+da


#LossGradrou = np.loadtxt('optallerror\sgdLossGradrouAllQpltb3.txt')
LossGradrou = np.loadtxt('optallerror\onsgdLossGradrouAllQpltb8i1000.txt')

Loss=LossGradrou[0:4,:]
Gradnorm=LossGradrou[4:8,:]
Cr=LossGradrou[8:12,:]
Bit=LossGradrou[12:16,:]
minl=Loss.min()
float=32
deLoss=Loss-minl


index=np.zeros(nalg)
for i in range(0,nalg):
    k=0
    while(deLoss[i,k]>10**(-6) and k<Iter-1):
        k=k+1
    index[i]=k

[Cr[0,int(index[0])], Cr[1,int(index[1])],Cr[2,int(index[2])],Cr[3,int(index[3])]] #communication #
[b*nv*Cr[0,int(index[0])],float*nv* Cr[1,int(index[1])],b*nv*Cr[2,int(index[2])],float*nv*Cr[3,int(index[3])]]
'''

