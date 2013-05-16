#! /usr/bin/env python

import numpy as np
from scipy.io import loadmat
from dipy.reconst.dti import Tensor
from dipy.reconst.dni import EquatorialInversion
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.recspeed import peak_finding
from visualize_dsi import show_blobs
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.core.geometry import vec2vec_rotmat
from scipy.optimize import fmin as fmin_powell
from scipy.optimize import leastsq
from scipy.io import savemat
from time import time
from dipy.core.geometry import sphere2cart, cart2sphere


def SingleTensor(bvals,gradients,S0,evals,evecs,snr=None):
    """ Simulated signal with a Single Tensor
     
    Parameters
    ----------- 
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    S0 : double,
    evals : array, shape (3,) eigen values
    evecs : array, shape (3,3) eigen vectors
    snr : signal to noise ratio assuming gaussian noise. 
        Provide None for no noise.
    
    Returns
    --------
    S : simulated signal    
    
    """
    S=np.zeros(len(gradients))
    D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=S0*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    return S
    
def MultiTensor(bvals,gradients,S0,mf,mevals,mevecs):
    S=np.zeros(len(gradients))
    m=len(mf)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            S[i+1]+=S0*f*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    return S

def ODF(vecs,mf,mevals,mevecs):
    odf=np.zeros(len(vecs))
    m=len(mf)
    for (i,v) in enumerate(vecs):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            iD=np.linalg.inv(D)
            nD=np.linalg.det(D)
            upper=(np.dot(np.dot(v.T,iD),v))**(-3/2.)
            lower=4*np.pi*np.sqrt(nD)
            odf[i]+=f*upper/lower
    return odf

def lambda_ranges():
    #print 'max', 1*10**(-3),'to',2*10**(-3)
    #print 'other', 0.1*10**(-3),'to',0.6*10**(-3)
    lmin=np.linspace(0.1,0.6,10)*10**(-3)
    lmax=np.linspace(1,2,10)*10**(-3)
    f=np.linspace(0.1,1,10)
    return lmax,lmin,f 

def count_peaks(PK):
    return np.sum(PK>0,axis=-1)

def all_evecs(e0):
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    mat=vec2vec_rotmat(axes[2],e0)
    e1=np.dot(mat,axes[0])
    e2=np.dot(mat,axes[1])
    return np.array([e0,e1,e2])

def opt2(params,bvals,bvecs,signal,mevecs):
    mf=[params[0],1-params[0]]
    mevals=np.zeros((2,3))
    mevals[0,0]=params[1]
    mevals[0,1:]=params[2]
    mevals[1,0]=params[3]
    mevals[1,1:]=params[4]
    mevals=mevals*10**(-3)
    S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)
    return np.sum(np.sqrt((S-signal)**2))

def opt3(params,bvals,bvecs,signal,mevecs):
    mf=[params[0],params[1],1-params[0]-params[1]]
    mevals=np.zeros((3,3))
    mevals[0,0]=params[2]
    mevals[0,1:]=params[3]
    mevals[1,0]=params[4]
    mevals[1,1:]=params[5]
    mevals[2,0]=params[6]
    mevals[2,1:]=params[7]
    mevals=mevals*10**(-3)
    S=MultiTensor(bvals,bvecs,1.,mf,mevals,mevecs)
    return np.sum(np.sqrt((S-signal)**2))

def unpackopt2(xopt):
    params=xopt
    mf=[params[0],1-params[0]]
    mevals=np.zeros((2,3))
    mevals[0,0]=params[1]
    mevals[0,1:]=params[2]
    mevals[1,0]=params[3]
    mevals[1,1:]=params[4]
    mevals=mevals*10**(-3)
    return mf, mevals

def unpackopt3(xopt):
    params=xopt
    mf=[params[0],params[1],1-params[0]-params[1]]
    mevals=np.zeros((3,3))
    mevals[0,0]=params[2]
    mevals[0,1:]=params[3]
    mevals[1,0]=params[4]
    mevals[1,1:]=params[5]
    mevals[2,0]=params[6]
    mevals[2,1:]=params[7]
    mevals=mevals*10**(-3)
    return mf,mevals


def load_data(test,typ,snr):

    if test=='train':
        fname='/home/eg309/Software/Hardi/'+typ+'__SNR='+snr+'__SIGNAL.mat'
    if test=='test':
        fname='/home/eg309/Software/Hardi/TestData/'+typ+'__SNR='+snr+'__SIGNAL.mat'

    fgrads='/home/eg309/Software/Hardi/gradient_list_257_clean.txt'
    fvertices='/home/eg309/Software/Hardi/TrainingData/ODF_XYZ.mat'
    vertices=loadmat(fvertices)
    vertices=np.ascontiguousarray(vertices['ODF_XYZ'])

    ffaces='/home/eg309/Software/Hardi/TrainingData/FACES.mat'
    faces=loadmat(ffaces)
    faces=np.ascontiguousarray(faces['K'])
    faces=faces-1 #from matlab to numpy indexing

    DATA=loadmat(fname)
    dat=np.ascontiguousarray(DATA['E'])

    grads=np.loadtxt(fgrads)
    odf_sphere=(vertices.astype(np.float32),faces.astype(np.uint16))

    bvals=np.zeros(515)
    bvals[0]=0
    bvals[1:258]=grads[:,3]
    bvals[258:]=grads[:,3]
    bvecs=np.zeros((515,3))
    bvecs[0,:]=np.zeros(3)
    bvecs[1:258,:]=grads[:,:3]
    bvecs[258:,:]=-grads[:,:3]

    data=np.zeros(dat.shape[:3]+(515,))
    data[:,:,:,0]=1
    data[:,:,:,1:258]=dat.copy()
    data[:,:,:,258:]=dat.copy()

    return data,bvals,bvecs,odf_sphere

def dump():

    """
    #EIT
    ei=EquatorialInversion(data,bvals,bvecs,
                odf_sphere=odf_sphere,
                mask=None,
                half_sphere_grads=False,
                auto=False,
                save_odfs=True,
                fast=True)

    ei.radius=np.arange(0,5,0.2)
    ei.gaussian_weight=0.02
    ei.set_operator('laplap')#laplacian
    ei.update()
    ei.fit() 

    #DSI
    ds=DiffusionSpectrum(data,bvals,bvecs,            
                odf_sphere=odf_sphere,
                mask=None,
                half_sphere_grads=False,
                save_odfs=True)
    """
    pass


def analyze_peaks(data,ten,qg):

    PK=qg.PK
    IN=qg.IN
    M=count_peaks(PK)
    R={}
    for index in np.ndindex(M.shape):
        #print index, M[index]
        if M[index]==0:
            mf=[0]
            mevals=[ten.evals[index]]
            mevecs=[ten.evecs[index]]
            directions=[get_phi_theta(ten.evecs[index][:,0])]
            odf=ODF(qg.odf_vertices,mf,mevals,mevecs)            
        if M[index]==1:
            mf=[1.]
            mevals=[ten.evals[index]]
            mevecs=[ten.evecs[index]]
            directions=[get_phi_theta(ten.evecs[index][:,0])]
            odf=ODF(qg.odf_vertices,mf,mevals,mevecs)
        if M[index]==2:
            e0=qg.odf_vertices[np.int(qg.IN[index+(0,)])]
            e1=qg.odf_vertices[np.int(qg.IN[index+(1,)])]
            signal = data[index]
            mevecs=[all_evecs(e0).T,all_evecs(e1).T]
            mf=[0.5,0.5]
            mevals=np.array(([0.0015,0.0003,0.0003],
                    [0.0015,0.0003,0.0003]))
            directions=[get_phi_theta(e0),
                        get_phi_theta(e1)]
            odf=qg.ODF[index]
            odf=odf#-0.4*odf.max()
            odf=odf/np.float(odf.sum())

        if M[index]==3:
            e0=qg.odf_vertices[np.int(qg.IN[index+(0,)])]
            e1=qg.odf_vertices[np.int(qg.IN[index+(1,)])]
            e2=qg.odf_vertices[np.int(qg.IN[index+(2,)])]
            signal = data[index]
            mevecs=[all_evecs(e0).T,all_evecs(e1).T,all_evecs(e2).T]
            mf=[0.33,0.33,0.33]
            mevals=np.array(([0.0015,0.0003,0.0003],
                        [0.0015,0.0003,0.0003],
                        [0.0015,0.0003,0.0003]))
            directions=[get_phi_theta(e0),
                        get_phi_theta(e1),
                        get_phi_theta(e2)]
            odf=qg.ODF[index]
            odf=odf#-0.4*odf.max()
            odf=odf/np.float(odf.sum())
        R[index]={'m':M[index],'f':mf,'evals':mevals,'evecs':mevecs,'odf':odf,'directions':directions}

    return M,R

def get_phi_theta(e):
    r,theta,phi=cart2sphere(e[0],e[1],e[2])
    phi=np.mod(phi,2*np.pi)
    theta=np.mod(theta,np.pi)
    return np.array([phi,theta])

def show_no_fibs(M,R):
    for index in np.ndindex(M.shape):
        print index
        print R[index]['m']

def revised_peak_no(odf,odf_faces,peak_thr):
    peaks,inds=peak_finding(odf,odf_faces)
    ibigp=np.where(peaks>peak_thr*peaks[0])[0]
    l=len(ibigp)                
    if l>3:
        l=3               
    if l==0:
        return np.sum(peaks[l]/np.float(peaks[0])>0)                         
    if l>0:                    
        return np.sum(peaks[:l]/np.float(peaks[0])>0)

def best_smoother():

    for smoo in np.linspace(3,5,10):
        gqs=GeneralizedQSampling(data,bvals,bvecs,smoo,
                        odf_sphere=odf_sphere,
                        mask=None,
                        squared=True,
                        auto=False,
                        save_odfs=True)
        gqs.peak_thr=0.5
        gqs.fit()
        gqs.ODF[gqs.ODF<0]=0.
        
        odf=gqs.ODF[0,0,0]

        print smoo, np.sum((direct_odf/direct_odf.max() - odf/odf.max())**2)


def example(type):

    if type=='1a':
        mf=[1.]
        mevals=np.array([[ 0.002 ,  0.0006,  0.0006]])
        mevecs=[np.array([[ 0.53140014,  0.72508361,  0.43802701],
                [-0.84668511,  0.43802701,  0.30208724],
                [ 0.02717085, -0.53140013,  0.84668509]])]
    if type=='1b':
        mf=[1.]
        mevals=np.array([[ 0.002 ,  0.0001,  0.0001]])
        mevecs=[np.array([[ 0.53140014,  0.72508361,  0.43802701],
                [-0.84668511,  0.43802701,  0.30208724],
                [ 0.02717085, -0.53140013,  0.84668509]])]
    if type=='2a':
        mf=[0.5,0.5]
        mevals=np.array([[ 0.002 ,  0.0006,  0.0006],
                [ 0.002 ,  0.0006,  0.0006]])
        mevecs=[np.array([[ 0.53140014,  0.72508361,  0.43802701],
                [-0.84668511,  0.43802701,  0.30208724],
                [ 0.02717085, -0.53140013,  0.84668509]]),
                np.array([[-0.99880177, -0.04874515, -0.00435084],
                [-0.00414365, -0.00435084,  0.99998195],
                [-0.0487632 ,  0.99880177,  0.00414365]])]    
    if type=='2b':
        mf=[0.5,0.5]
        mevals=np.array([[ 0.002 ,  0.0006,  0.0006],
                [ 0.002 ,  0.0006,  0.0006]])
        mevecs=[np.array([[-0.99880177, -0.04874515, -0.00435084],
                [-0.00414365, -0.00435084,  0.99998195],
                [-0.0487632 ,  0.99880177,  0.00414365]]),
                np.array([[ -2.57094204e-03,   9.99996692e-01,  -7.28549000e-05],
                [-5.66298328e-02,  -7.28549000e-05,   9.98395234e-01],
                [9.98391926e-01,   2.57094757e-03,   5.66299545e-02]])]
    if type=='3':
        mf=[0.33,0.33,0.33]
        mevals=np.array([[ 0.002 ,  0.0006,  0.0006],
                [ 0.002 ,  0.0006,  0.0006],
                [ 0.002 ,  0.0006,  0.0006]])
        mevecs=[np.array([[ 0.53140014,  0.72508361,  0.43802701],
                [-0.84668511,  0.43802701,  0.30208724],
                [ 0.02717085, -0.53140013,  0.84668509]]),
                np.array([[-0.99880177, -0.04874515, -0.00435084],
                [-0.00414365, -0.00435084,  0.99998195],
                [-0.0487632 ,  0.99880177,  0.00414365]]),
                np.array([[ -2.57094204e-03,   9.99996692e-01,  -7.28549000e-05],
                [-5.66298328e-02,  -7.28549000e-05,   9.98395234e-01],
                [9.98391926e-01,   2.57094757e-03,   5.66299545e-02]])]
    return mf,mevals,mevecs

def get_all_odfs(M,R,sphsize):
    ODF=np.zeros(M.shape+(sphsize,))
    for index in np.ndindex(M.shape):
        odf=R[index]['odf']
        if np.sum(odf)>40:
            ODF[index]=odf*4*np.pi/724.#/np.float(odf.sum())
        else:
            ODF[index]=odf
    return ODF

def save_for_mat(test,typ,snr,M,R,sphsize=724):

    if test=='train':
        fname='/home/eg309/Software/Hardi/Results/Training/interm_'+typ+'__SNR='+snr+'__SIGNAL.mat'
    if test=='test':
        fname='/home/eg309/Software/Hardi/Results/Testing/interm_'+typ+'__SNR='+snr+'__SIGNAL.mat'

    F=np.zeros(M.shape+(3,))
    ODF=np.zeros(M.shape+(sphsize,))
    L=np.zeros(M.shape+(3,3))
    RO=np.zeros(M.shape+(3,2))

    for index in np.ndindex(M.shape):
        for m in range(M[index]):
            F[index][m]=R[index]['f'][m]
            L[index][m]=R[index]['evals'][m][::-1]            
            RO[index][m]=R[index]['directions'][m]

        ODF[index]=R[index]['odf']
        #ODF[index]=ODF[index]/np.float(ODF[index].sum())
    asf=np.asfortranarray
    data = {'results':{'M':asf(M),
                    'F':asf(F),
                    'L':asf(L), 
                    'R':asf(RO),
                    'ODF':asf(ODF)}}
    savemat(fname,data)


if __name__ == '__main__':

    test='test'
    #types=['Training_SF']
    types=['Testing_SF','Testing_IV']
    #SNRs=['10','20','30']
    SNRs=['05','10','15','20','25','30','35','40']
    #smooth=[3.,3.3,3.5]
    smooth=[2.,3.,3.2,3.3,3.5,3.5,3.8,4.]

    types=['Testing_IV']
    #SNRs=['10','20','30']
    SNRs=['30']
    #smooth=[3.,3.3,3.5]
    smooth=[3.5]

    save=False
    show=True
    
    for typ in types:
        for (i,snr) in enumerate(SNRs):
            data,bvals,bvecs,odf_sphere=load_data(test,typ,snr)#'3D_SF'
            #data=data[4,4,0]    
            #mf,mevals,mevecs=example('1b')    
            #signal=MultiTensor(bvals,bvecs,S0=1.,mf=mf,mevals=mevals,mevecs=mevecs)
            #data=signal    
            #data=data[None,None,None,:]
            data=data[:,4:40,:,:]
            #ten
            ten = Tensor(100*data, bvals, bvecs)
            FA = ten.fa()
            #GQI
            gqs=GeneralizedQSampling(data,bvals,bvecs,smooth[i],
                            odf_sphere=odf_sphere,
                            mask=None,
                            squared=True,
                            auto=False,
                            save_odfs=True)
            gqs.peak_thr=0.5
            gqs.fit()
            gqs.ODF[gqs.ODF<0]=0.
            #manipulate
            qg=gqs
            #pack_results
            M,R=analyze_peaks(data,ten,qg)
            if test=='train':
                K=np.load('trainSF.npy')
                print 'SNR',snr, 'smooth',smooth[i],\
                    'Missed',np.sum(np.abs(M-K)>0), \
                    'Success',100*(np.float(np.prod(M.shape))-np.sum(np.abs(M-K)>0))/np.float(np.prod(M.shape)),'%'
            if save==True:
                save_for_mat(test,typ,snr,M,R)
            #show ODFs
            if show==True:
                ODFs=get_all_odfs(M,R,len(qg.odf_vertices))
                show_blobs(ODFs[:,:,0,:][:,:,None,:],qg.odf_vertices,qg.odf_faces,size=1.5,scale=1.,norm=True)



