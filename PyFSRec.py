'''
Focal Series Reconstruction

An object definition to handle the reconstruction of a focal series
'''

import numpy as np
import numpy.matlib
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
import configparser
import ast
import argparse
import scipy
from scipy import signal
from scipy.ndimage import median_filter
from scipy.optimize import leastsq
import sys
from numpy import genfromtxt
import os.path

debug=False # debugging
debugalign=False # debugging to show alignment progress 
stepana=False # focus step analysis
prealignonly=False
nonlinear=False

'''
-----------------------------------------------------------------------------------
  A few routines that are required for alignment purposes
-----------------------------------------------------------------------------------
'''
def nextint(x):
    if (x < 0):
        return math.floor(x)
    else:
        return math.ceil(x)

def create_circular_mask(h, w, center=None, radius=None):
     
    # define a circular cut-off mask with sub-pixel centering
    # inefficient code, useful for small patches only
    # returns a pixel mask

    # Parameters
    #----------
    # h, w : integers, 
    #    height and width of the patch
    # center : list or np.ndarray
    #    optional center offset, in fractional pixels
    # radius : float
    #    mask radius in pixels
    #    if no radius is given then the maximum radius around the center fitting into the patch is chosen
    
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def shiftfft2d(fftdata, deltax, deltay, phase=0.):

    # FFT-based sub-pixel image shift.
    # returns fft multiplied by the phase wedge that relates to th image shift by deltax, deltay in pixels in the real space domain
    
    # Parameters
    #----------
    # fftdata : np.ndarray
    #    fft of a 2D image
    # phase : float
    #    optional phase offset, in radians

    ny,nx = fftdata.shape

    xfreq = deltax * np.fft.fftfreq(nx)[np.newaxis,:]
    yfreq = deltay * np.fft.fftfreq(ny)[:,np.newaxis]
    freq_grid = xfreq + yfreq

    kernel = np.exp(-1j*2.*np.pi*freq_grid-1j*phase)
    
    return fftdata * kernel


def shift2d(data, deltax, deltay, phase=0.):

    # FFT-based sub-pixel image shift.
    # source: https://image-registration.readthedocs.io/en/latest/_modules/image_registration/fft_tools/shift.html

    # Parameters
    #----------
    # data : np.ndarray
    #    2D image
    # phase : float
    #    Phase, in radians
    #if np.any(np.isnan(data)):
    #    data = np.nan_to_num(data)

    ny,nx = data.shape

    xfreq = deltax * np.fft.fftfreq(nx)[np.newaxis,:]
    yfreq = deltay * np.fft.fftfreq(ny)[:,np.newaxis]
    freq_grid = xfreq + yfreq

    kernel = np.exp(-1j*2.*np.pi*freq_grid-1j*phase)

    result = np.fft.ifft2( np.fft.fft2(data) * kernel )
    
    return np.real(result)

def iterate_COM(corr, thresh=None, framesz=None, maxiter=None):
    
    # iterate the center of mass in a correlation image
    # corr: np.ndarray
    #    2D correlation image in fft order
    # thresh: scalar float 
    #    convergence boundary, pixel convergence, default is 0.1 pixels
    # framesz: pixel radius for refinement, default is 3
    # maxiter: maximum iterations, default is 10
    #
    # plot conventions
    # plot markers: green=target center
    #               red = iterated center of mass
    #               upon convergence the distance between both is less than thresh
    #
    # returns: array [dx,dy] 
    #     correlation center
    finished=False
    if thresh is None:
        thresh=0.1
    if framesz is None:
        framesz=3
    if maxiter is None:
        maxiter=10
    iter=1
    framexhalf=framesz
    frameyhalf=framesz
    # for debugging: mark center
    corrsh=corr # the copy corrsh will be shifted by integer amounts
    # corrsh[0,0]=0.
    # shift correlation image by framexhalf, frameyhalf and cut a smaller frame for analysis
    corrsh=np.roll(np.roll(corrsh,framexhalf,axis=1),frameyhalf,axis=0)
    # now take an odd size frame of 2xframexhalf+1, 2xframeyhalf+1 
    # the old center should be now at framexhalf, frameyhalf
    # for debugging: mark center as 0 int
    corrmask=corrsh[0:(framexhalf*2)+1,0:(frameyhalf*2+1)]
    startim=corrmask
    if debugalign:
        plt.imshow(corrmask)
    maxpos=np.where(corrmask == np.amax(corrmask))
    sumcx=0. # sumcx and sumcy contain the integer part of the displacement,
    sumcy=0. # so far we didn't shift, so they are zero
    if debugalign:
        print("Initial center ", maxpos[1][0],maxpos[0][0])
        print("Initial dx, dy ", maxpos[1][0]-framexhalf,maxpos[0][0]-frameyhalf)
        plt.scatter(maxpos[1][0], maxpos[0][0], s=5, c='red', marker='o')
    gradx=np.linspace(-framexhalf, framexhalf, (2*framexhalf+1))
    grady=np.linspace(-frameyhalf, frameyhalf, (2*frameyhalf+1))
    dx=maxpos[1][0]-framexhalf
    dy=maxpos[0][0]-frameyhalf
    corrsh=np.roll(np.roll(corrsh,-round(dx),axis=1),-round(dy),axis=0)
    if debugalign:
        plt.show()
    # prepare circular mask
    rad=np.min([framexhalf,frameyhalf])
    cmask=create_circular_mask(corrmask.shape[1], corrmask.shape[0], center=[framexhalf,frameyhalf], radius=rad)
    while not(finished):
        # Shift corrsh by the integer part of the new dx and dy 
        corrmask=corrsh[0:(framexhalf*2)+1,0:(frameyhalf*2+1)]
        sumcx+=round(dx) # sumcx and sumcy contain the integer part of the displacement
        sumcy+=round(dy)
        if debugalign:
            plt.imshow(cmask*corrmask)  
        # mark target center of gravity for orientation
        if debugalign:
            plt.scatter(framexhalf, frameyhalf, s=5, c='green', marker='o')        
        cx=np.sum((corrmask*cmask)/np.sum(corrmask*cmask)*gradx)
        cy=np.sum(np.transpose(corrmask*cmask)/np.sum(corrmask*cmask)*grady)
        # mark current center of gravity for orientation
        if debugalign:
            plt.scatter(framexhalf+cx, frameyhalf+cy, s=5, c='red', marker='o')  
            plt.show()
        # correction of shift is cx, cy
        dx += cx
        dy += cy
        # shift corrsh
        corrsh=shift2d(corrsh,-cx,-cy)
        corrsh=corrsh.real
        # report
        if debugalign:
            print("Center of mass correction (iteration ",iter,"): ",cx,",",cy)
            print("Total dx, dy ", dx,dy)
        iter +=1
        if ((math.sqrt(cx*cx + cy*cy) < thresh)): 
            finished=True
            if debugalign:
                print("Refinement below threshold (", thresh,")")
        elif (iter > maxiter):
            finished=True
            if debugalign:
                print("Maximum iteration count (", maxiter,") reached")
    # plot center onto original cross-correlation
    # corrsh=corr # the copy corrsh will be shifted by integer amounts
    # shift correlation image by framexhalf, frameyhalf and cut a smaller frame for analysis
    if debugalign:
        plt.imshow(startim)
        plt.scatter(framexhalf+dx, frameyhalf+dy, s=5, c='red', marker='o')
        plt.show()
    return np.array([dx,dy])



def image_corr(im1, im2, maskzero=None):
   import matplotlib.pyplot as plt
   # cross-correlate two images
   # returns: cross-correlation, displacement
   #
   # im1, im2= images
   # maskzero = experimental, f set the the central pixel in the correlation is replace by the median of the neighbouring pixels
   # 
   
   sx,sy = im1.shape
   # the type cast into 'float' is to avoid overflows
   im1_gray = im1.astype('float')
   im2_gray = im2.astype('float')

   # get rid of the averages, otherwise the results are not good
   im1_gray -= np.mean(im1_gray)
   im2_gray -= np.mean(im2_gray)

   # calculate the correlation image; note the flipping of onw of the images
   corrimg=np.abs(scipy.signal.fftconvolve(im1_gray, im2_gray[::-1,::-1], mode='same'))
   if not(maskzero is None):
       # replace center pixel by meadian of the neighbour pixels
       # this is to avoid a spike by fix pattern noise
       # CAUTION, does not perform as expected, requires debugging
       med  = np.median([corrimg[sx//2-2:sx//2-1,sy//2],corrimg[sx//2+1:sx//2+2,sy//2],corrimg[sx//2,sy//2-2:sy//2-1],corrimg[sx//2,sy//2+1:sy//2+2]])
       corrimg[sx//2,sy//2]=med
   x, y =np.unravel_index(np.argmax(corrimg), corrimg.shape)
   displ=[x-sx//2,y-sy//2]
   maxval=np.abs(corrimg[x,y])
   if debugalign:
       fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
       ax1.imshow(im1_gray)
       ax1.set_title('image')
       ax2.imshow(im1_gray)
       ax2.set_title('reference')
       ax3.imshow(np.abs(corrimg))
       ax3.set_title('cross correlation')
       ax4.imshow(np.abs(corrimg))
       ax4.set_title("dx, dy={0:5d},{1:5d}".format(displ[0],displ[1]))
       ax4.scatter(x, y, s=5, c='red', marker='o')
       plt.show()  # render the plot
   return corrimg, displ,maxval

# phase correlation
def phase_corr(a, b):
    G_a = np.fft.fft2(a) 
    G_b = np.fft.fft2(b)
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    R /= np.abs(R)
    corrimg = np.abs(np.fft.ifft2(R))
    sx,sy = corrimg.shape
    corrimg=np.roll(corrimg,(math.ceil(sx/2)),axis=0)
    corrimg=np.roll(corrimg,(math.ceil(sy/2)),axis=1)
    x, y =np.unravel_index(np.argmax(corrimg), corrimg.shape)
    displ=[x-sx//2,y-sy//2]
    if debug:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(a)
        ax1.set_title('image')
        ax2.imshow(b)
        ax2.set_title('reference')
        ax3.imshow(np.abs(corrimg))
        ax3.set_title('cross correlation')
        ax4.imshow(np.abs(corrimg))
        ax4.set_title("dx, dy={0:5d},{1:5d}".format(displ[0],displ[1]))
        plx=sx//2-displ[0]
        ply=sy//2-displ[1]
        ax4.scatter(sx//2, sy//2, s=5, c='green', marker='o')
        ax4.scatter(plx, ply, s=5, c='red', marker='o')
        plt.show()  # render the plot
    return corrimg, displ, np.abs(corrimg[x,y])

def stdnormalize(a):
    avg=np.average(a)
    stddev=np.std(a)
    a=(a-avg)/stddev
    return a

def cross_corr(a, b):
    G_a = np.fft.fft2(stdnormalize(a)) 
    G_b = np.fft.fft2(stdnormalize(b))
    conj_b = np.ma.conjugate(G_b)
    R = G_a*conj_b
    corrimg = np.abs(np.fft.ifft2(R))
    sx,sy = corrimg.shape
    corrimg=np.roll(corrimg,(math.ceil(sx/2)),axis=0)
    corrimg=np.roll(corrimg,(math.ceil(sy/2)),axis=1)
    x, y =np.unravel_index(np.argmax(corrimg), corrimg.shape)
    displ=[x-sx//2,y-sy//2]
    if debugalign:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(a)
        ax1.set_title('image')
        ax2.imshow(b)
        ax2.set_title('reference')
        ax3.imshow(np.abs(corrimg))
        ax3.set_title('cross correlation')
        ax4.imshow(np.abs(corrimg))
        ax4.set_title("dx, dy={0:5d},{1:5d}".format(displ[0],displ[1]))
        plx=sx//2-displ[0]
        ply=sy//2-displ[1]
        ax4.scatter(sx//2, sy//2, s=5, c='green', marker='o')
        ax4.scatter(plx, ply, s=5, c='red', marker='o')
        
        plt.show()  # render the plot
    return corrimg, displ, np.abs(corrimg[x,y])


# define 2D gaussian, scaled between 0 and 1
def gauss2d(x=0, y=0, mx=0, my=0, sx=10, sy=10):
    return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def bandpass(dim,lo=0.02,hi=0.5):
    # define a band pass filter, a smoothed tophat calculated 
    # as the product of a gaussian and an inverted gaussian
    # the high pass and low pass cut-offs as a fraction of the Nyquist frequency
    XDim=dim[0]
    YDim=dim[1]
    minx=int(XDim/2)
    miny=int(YDim/2)
    x = np.linspace(-minx,-minx+XDim-1,num=XDim)
    y = np.linspace(-miny,-miny+YDim-1,num=YDim)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    lopass = gauss2d(x, y,0,0,hi*minx,hi*miny)
    hipass = 1-gauss2d(x, y,0,0,lo*minx,lo*miny)
    bp=np.roll(np.roll(np.multiply(lopass,hipass),-minx),-miny,axis=0)
    bp[0,0]=1. # preserve the mean value
    return bp

'''

 Routines for principal component analysis

'''

def ImagePCA(images,crop=None):
    # images: array(xdim,ydim,Nim) of 2D images
    # crop is a 4tupel with crop information, (startx,starty,widthx,widthy)
    dimx, dimy, Nim = images.shape
    # for PCA each image is a 1D row in A, we have Nim rows
    if crop:
        A=np.zeros((crop[2]*crop[3],Nim),dtype=np.single)
        for ii in range(0,Nim):
            tmp=images[crop[0]:crop[0]+crop[2],crop[1]:crop[1]+crop[3],ii]
            A[:,ii]=tmp.reshape(crop[2]*crop[3])
    else:
        A=np.zeros((dimx*dimy,Nim),dtype=np.single)
        for ii in range(0,Nim):
            tmp=images[:,:,ii]
            A[:,ii]=tmp.reshape(dimx*dimy)
    M=np.mean(A.T, axis=1) # subtract image mean
    C=A-M
    # calculate covariance matrix
    V=np.cov(C.T)
    # eigendecomposition of the covariance matrix
    eigenvals, eigenvecs=np.linalg.eig(V)
    pca = eigenvecs.dot(A.T)
    if crop:
        return eigenvals, pca.reshape(Nim,crop[2],crop[3])
    else:
        return eigenvals, pca.reshape(Nim,dimx,dimy)
    

def binArray(data, axis, binstep, binsize, func=np.nanmean):
#
#    binning of a multidimensional array 
#    author: Alexandre Kempf
#    source: https://stackoverflow.com/questions/21921178/binning-a-numpy-array
#
#    data is the array
#    axis is the axis you want to bin
#    binstep is the number of points between each bin (allow overlapping bins)
#    binsize is the size of each bin
#    func is the function you want to apply to the bin (np.max for maxpooling, np.mean for an average ...)
#
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def find_outlier_pixels(data,tolerance=10,worry_about_edges=True):
    #This function finds the hot or dead pixels in a 2D dataset. 
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    
    blurred = median_filter(data, size=3)
    difference = data - blurred
    threshold = tolerance*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    fixed_image = np.copy(data) #This is the image with the hot pixels removed
    for y,x in zip(hot_pixels[0],hot_pixels[1]):
        fixed_image[y,x]=blurred[y,x]

    if worry_about_edges == True:
        height,width = np.shape(data)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(data[index-1:index+2,0:2])
            diff = np.abs(data[index,0] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(data[index-1:index+2,-2:])
            diff = np.abs(data[index,-1] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(data[0:2,index-1:index+2])
            diff = np.abs(data[0,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(data[-2:,index-1:index+2])
            diff = np.abs(data[-1,index] - med)
            if diff>threshold: 
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(data[0:2,0:2])
        diff = np.abs(data[0,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(data[0:2,-2:])
        diff = np.abs(data[0,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(data[-2:,0:2])
        diff = np.abs(data[-1,0] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(data[-2:,-2:])
        diff = np.abs(data[-1,-1] - med)
        if diff>threshold: 
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med

    return hot_pixels,fixed_image

# standard quadratic function for extremum refinement

def parabola(params, x):
    a, b, c = params
    return a * x * x + b * x + c


# Error function, that is, the difference between the value obtained by the fitted curve and the actual value
def parabolaerror(params, x, y):
    return parabola(params, x) - y



'''
-----------------------------------------------------------------------------------
  Focal series reconstruction
  Object definition code
-----------------------------------------------------------------------------------
'''

class FSR(object):

    # Initializer / Instance Attributes
    def __init__(self):
        # initialize instance attributes, their values will be partially overwritten by the method ReadParameterFile below
        self.inpath = ""
        self.outpath = ""
        self.infilenameprefix = ""
        self.outfilenameprefix = ""
        self.NImages=0
        self.dim = [512,512]
        self.framewidth=[256,256]
        self.frameoffset=[0,0] # offset of the rfernec quarter frame
        self.roi = [self.dim[0]//4,self.dim[1]//4,self.dim[0]//4+self.dim[0]//2,self.dim[1]//4+self.dim[1]//2]
        self.sampling=[1.,1.]
        self.recsampling=[1.,1.]
        self.foci=None   # foci of selected images 
        self.allfoci=None # foci of all images
        self.Cs=0.
        self.C5=0.
        self.imagespread=20 # image spread in pm
        self.reclimit=10.
        self.voltage=80.
        self.lamda=0.00417560
        self.semiconv=0.1 # semiconvergence in mrad
        self.focspread=3. # defocus spread in nm
        self.distpow1rec=None
        self.distpow2rec=None
        self.distpow3rec=None
        self.distpow4rec=None
        self.data=None # original data
        self.sfftdata=None # fft of shifted images, for temporary usage
                           # a shift is just a phase factor on the fft, so no interpolation artefacts
        self.fftdata=None # fft of images
        self.fftsimdata=None # fft of simulated images
        self.alignfilt=None # alignment filter
        self.alignpass=[2.,6.] # alignment top-hat filter pass values in 1/nm
        self.outfilt=None # output filter
        self.outpass=None # output top-hat filter pass values in 1/nm
        self.refinementfilttype = 'gaussian'
        self.refinementfilt=None # step analysis filter
        self.refinementpass=[2.,6.] # step analysis top-hat filter pass values in 1/nm
        self.autofocfilt=None # autofocus filter
        self.autofocpass=[2.,6.] # autofocus top-hat filter pass values in 1/nm
        self.alignprecision=0.1 # alignment precision in pix
        self.alignfilttype='gaussian' # alternatives: 'tophat', 'gaussian'
        self.displacements=None # total image displacements
        self.prealignswitch=True
        self.invertforalign=False
        self.commonframe=False
        self.fixhotpix=False
        self.NMaxPreali=10
        self.savealignedimages=False
        self.saveintermediaterec=False
        self.NLinear=10
        self.envelopeswitch=True
        self.tn=None
        self.fn=None
        self.psi=None
        self.fftpsiscatt=None
        # linear envelope functions
        self.temporal_env = None
        self.spatial_env = None
        self.imagespread_env = None
        self.tot_env = None
        self.invtot_env = None
        self.linit=1  # linear iteration counter
        # choose correlation function
        self.imcorrelation=cross_corr
        self.wavcorrelation=cross_corr
        self.comkernel = 3 # radius for com refinement in pix
        self.commaxiter = 5
        self.ravg=0
        self.envlimit=0.05 # envelope limit below which the transfer will be masked, info limit would be 0.135
                           # beware: this is another cut-off then reclimit
        self.dispcorrelation='sequential' # sequential, sequential-pca or sequential-com
        self.stepreffocfact=0.5
        self.steprefrangefact=3.
        return

    def ReadParameterFile(self, fname):
        # read a parameter file to populate the values of the object data
        # uses the configparser module: https://docs.python.org/3/library/configparser.html
        print("Reading parameter file ",fname)
        config = configparser.ConfigParser()
        config.read(fname)
        config.sections()
        #print(config.sections())
        self.inpath=config['INPUT/OUTPUT']['inputpath']
        if config.has_option('INPUT/OUTPUT','outputpath'):
            self.outpath=config['INPUT/OUTPUT']['outputpath']
        else:
            self.outpath=self.inpath
        self.infilenameprefix=config['INPUT/OUTPUT']['inputfilename']
        self.outfilenameprefix = config['INPUT/OUTPUT']['outputfilename']
        self.NImages=int(config['INPUT/OUTPUT']['nimages'])
        s='[{0:4d},{1:4d}]'.format(0,self.NImages-1)
        # print(s)
        self.imrange=[0,self.NImages]
        tmpl=ast.literal_eval(config['IMAGEDATA'].get('range',s))
        # in version 0.4 range is a list of ranges, we need to expand it to a list
        #print(config['IMAGEDATA'].get('range',s))
        #print(tmpl)
        i=iter(tmpl)
        res=[]
        for d in i:
            #print(d)
            #print(isinstance(d, list))
            if isinstance(d, list):
                a, b = map(int, d)
                res=[*res,*range(a,b+1)]
            else:
                res.append(int(d))
        self.imrange=res
        #print(res)
        #        
        self.dim = ast.literal_eval(config['IMAGEDATA']['dimensions'])
        self.sampling = ast.literal_eval(config['IMAGEDATA']['sampling'])
        self.recsampling[0] = 1./(self.sampling[0]*self.dim[0])
        self.recsampling[1] = 1./(self.sampling[1]*self.dim[1])
        if config.has_option('IMAGEDATA','area'):
            self.framewidth = ast.literal_eval(config['IMAGEDATA']['area'])
        else:
            self.framewidth=[self.dim[0]//2,self.dim[1]//2]
        if config.has_option('IMAGEDATA','offset'):
            self.frameoffset = ast.literal_eval(config['IMAGEDATA']['offset'])
        if config.has_option('IMAGEDATA','fixhotpix'):
            self.fixhotpix = config.getboolean('IMAGEDATA','fixhotpix')
        else:
            self.ffixhotpix=False
        self.voltage=float(config['OPTICS']['voltage'])
        self.lamda=1.2264263/math.sqrt(self.voltage*(1000.+0.97847538*self.voltage))
        self.Cs=float(config['OPTICS']['c3'])
        self.C5=float(config['OPTICS']['c5'])
        self.foci=np.array(ast.literal_eval(config['OPTICS']['foci']),dtype=np.float32)
        # in version 0.5: You can also just specify first and last focus in case of a linear focus ramp
        # set fous values and check number of foci
        # first all image focus values are requested, for all self.NImages
        # then only the part in the reconstruction range is taken
        # it means in turn that the focus specification in the input parameter file
        # always refers to the full data set, not the subset.
        NumSteps = np.size(self.imrange, 0)
        Numfoci = np.size(self.foci, 0)
        # print("NumSteps, Numfoci = {0:5d}, {1:5d}".format(NumSteps,Numfoci))
        if (Numfoci != self.NImages):
            # check if Numfoci == 2, meaning assume a linear ramp from the first image
            # to the last image (self.NImages)
            if (Numfoci == 2):
                rampfoci=np.copy(self.foci)
                focusrange=np.abs(rampfoci[-1]-rampfoci[0])
                step=focusrange/(self.NImages-1)
                if (rampfoci[0] > rampfoci[-1]):
                    self.foci=rampfoci[0]-np.arange(0,self.NImages,1)*step
                    # decreasing values, need to reverse the array
                else:
                    self.foci=rampfoci[0]+np.arange(0,self.NImages)*step
                print("Linear focus ramp - start, end, step = {0:8.3f}, {1:8.3f}, {2:8.3f}".format(self.foci[0],self.foci[-1],step))
        self.allfoci=np.copy(self.foci)
        # from the linear ramp of self.NImages take only the part that matches the image range sequence
        self.foci=self.foci[self.imrange]
        NumSteps=np.size(self.foci, 0)
        # check that number of foci matches image range
        if (NumSteps != np.size(self.imrange, 0)):
                print("Fatal error - Mismatch in number of focus values.")
                print("Number of input images,reconstruction images,foci = {0:5d},{1:5d},{2:5d}".format(self.NImages,np.size(self.imrange, 0),NumSteps))
                sys.exit()
        self.focspread=float(ast.literal_eval(config['OPTICS']['focusspread']))
        self.imagespread=float(ast.literal_eval(config['OPTICS']['imagespread']))
        self.semiconv=np.array(ast.literal_eval(config['OPTICS']['semiconvergence']),dtype=np.float32)
        self.reclimit=float(config['RECONSTRUCTION']['limit'])
        self.eps=float(config['RECONSTRUCTION']['filtercutoff'])
        self.alignpass = ast.literal_eval(config['RECONSTRUCTION']['alignmentfilter'])
        self.alignprecision=float(config['RECONSTRUCTION']['alignmentprecision'])
        self.prealignswitch=config.getboolean('RECONSTRUCTION','prealignment')
        self.envelopeswitch=config.getboolean('RECONSTRUCTION','envelopes')
        self.savealignedimages=config.getboolean('RECONSTRUCTION','savealigned')
        self.saveintermediaterec=config.getboolean('RECONSTRUCTION','saveintermediaterec')
        if config.has_option('RECONSTRUCTION','outputfilter'):
                self.outpass = ast.literal_eval(config['RECONSTRUCTION']['outputfilter'])
        self.NLinear=int(config['RECONSTRUCTION']['lineariterations'])
        if config.has_section('STEPREFINEMENT'):
            self.stepreffocfact=float(config['STEPREFINEMENT'].get('stepmultiplicator',0.5))
            self.steprefrangefact=float(config['STEPREFINEMENT'].get('rangemultiplicator',2.))
            if config.has_option('STEPREFINEMENT','refinementfilter'):
                self.refinementpass = ast.literal_eval(config['STEPREFINEMENT']['refinementfilter'])
        avgstep, intercept, r_value, p_value, std_err=scipy.stats.linregress(np.arange(NumSteps,dtype=np.single),self.foci)
        resfreq=np.sqrt(1/(self.lamda*np.abs(avgstep)))
        if config.has_section('ALIGNMENT'):
            self.comkernel=int(config['ALIGNMENT'].get('comkernel',3))
            self.NMaxPreali=int(config['ALIGNMENT'].get('maxprealign',10))
            self.dispcorrelation=config['ALIGNMENT'].get('methprealign','sequential')
            if config.has_option('ALIGNMENT','raverage'):
                self.ravg=int(config['ALIGNMENT']['raverage'])
            if config.has_option('ALIGNMENT','commonframe'):
                self.commonframe=config.getboolean('ALIGNMENT','commonframe')
                if self.commonframe:
                    if (self.ravg <= 1):
                        sys.exit("Common frame alignment requires raverage greater or equal to 2. Check parameter file.")
            if config.has_option('ALIGNMENT','invertprealign'):
                self.invertforalign=config.getboolean('ALIGNMENT','invertprealign')
        self.autofocrange=50.
        self.autofocfilt=None
        self.autofocpass=None
        if config.has_section('AUTOCORRECTION'):
            if config.has_option('AUTOCORRECTION','autofocusrange'):
                self.autofocrange=float(config['AUTOCORRECTION'].get('autofocusrange',50.))
            if config.has_option('AUTOCORRECTION','autofocusfilter'):
                self.autofocpass = ast.literal_eval(config['AUTOCORRECTION']['autofocusfilter'])
        # report
        print("+ --- Input parameters")
        print("Path                           =",self.inpath)
        print("Filename prefix                =",self.infilenameprefix)
        print("Output path                    =",self.outpath)
        print("Output filename prefix         =",self.outfilenameprefix)
        print("No. images                     =",self.NImages)
        print("No. images for reconstruction  =",np.size(self.imrange, 0))
        print("Image range                    =",self.imrange)
        print("Image dimensions (pix)         =",self.dim)
        print("Image sampling (nm)            =",self.sampling)
        print("Reciprocal sampling (1/nm)     = [{0:6.4f},{1:6.4f}]".format(self.recsampling[0],self.recsampling[1]))
        print("Voltage (kV)                   =",self.voltage)
        print("Wavelength (nm)                = {0:9.6f}".format(self.lamda))
        print("Spherical Aberration (nm)      =",self.Cs)
        print("First Focus value (nm)         ={0:6.2f}".format(self.foci[0]))
        print("Last Focus value (nm)          ={0:6.2f}".format(self.foci[-1]))
        print("Reconstruction limit (1/nm)    ={0:6.2f}".format(self.reclimit))
        print("Resonance frequency (1/nm)     ={0:6.2f}".format(resfreq))
        print("Low frequency limit (1/nm)     ={0:6.2f}".format(0.5/np.sqrt(np.pi*self.lamda*np.abs(self.foci[np.size(self.imrange, 0)-1]-self.foci[0]))))
        print("Filter cutoff (%)              =",self.eps*100)
        print("Pre-alignment COM kernel (pix) =",self.comkernel)
        print("Pre-alignment method           =",self.dispcorrelation)
        print("Pre-alignment to common frame  =",self.commonframe)
        
        #for ii in range(0,NumSteps):
        #    print(self.foci[ii])
        #exit(0)
        return

    def SetNLinearIterations(self, liniter):
        self.NLinear=liniter
        return

    def SetMaxPreAlignIterations(self, maxprealign):
        self.NMaxPreali=maxprealign
        return
    
    def PreviewRefArea(self):
        #
        # Save a preview of an image with the reference area
        #
        print("+ ---Saving first image for a preview of the reference area")
        ii=0
        tmp = np.fromfile(self.inpath + self.infilenameprefix + str(self.imrange[ii]).zfill(3) + ".raw", dtype=np.single, count=-1)
        tmp=tmp.reshape((self.dim[0],self.dim[1]))
        if self.fixhotpix:
            hot_pixels,tmp=find_outlier_pixels(tmp)
        fig, ax = plt.subplots()
        im = ax.imshow(tmp, cmap=cm.gray,
               origin='upper')
        # Create a Rectangle patch
        startx = (self.dim[0]- self.framewidth[0])//2 + self.frameoffset[0]
        starty = (self.dim[1]- self.framewidth[1])//2 + self.frameoffset[1]
        rect = patches.Rectangle((starty,startx),self.framewidth[1],self.framewidth[0],linewidth=1,edgecolor='y',facecolor='none')
        plt.xlabel('Dim 1')
        plt.ylabel('Dim 0')
        # Add the patch to the Axes
        ax.add_patch(rect)
        outfname=self.outpath+self.outfilenameprefix+"_referenceframe.png"
        plt.savefig(outfname, dpi=300)
        print("Preview saved to "+outfname)
        return
    
    def PrepareSamplingGrid(self):
        #
        # prepare the sampling grids in reciprocal space and their powers
        # results are stored in self.distpow1rec ... self.distpow4rec
        #
        # no formal parameters needed, the object instance knows them
        print("+ --- Preparing reciprocal space sampling grids")
        distx = np.roll(np.arange(self.dim[0],dtype=np.int64)-math.ceil(self.dim[0]/2),(math.ceil(self.dim[0]/2)),axis=0)
        # dist-x = distance from centre at dimx/2 in FFT order, pay attention it is integer now
        REPdistx = numpy.matlib.repmat(distx, self.dim[0], 1)
        distpow2_x = REPdistx * REPdistx  
        distpow2 = distpow2_x + distpow2_x.transpose()
        # distpow2 is the square distance from the centre, in FFT order, in pixel ccordinates
        distpow1=np.sqrt(distpow2)
        distpow3=distpow2*distpow1
        distpow4=distpow2*distpow2
        # reciprocal space sampling arrays g, g^3, g^4, in pixel ccordinates
        self.distpow1rec = self.recsampling[0] * distpow1
        self.distpow2rec = (self.recsampling[0] ** 2) * distpow2  
        self.distpow3rec = (self.recsampling[0] ** 3) * distpow3 
        self.distpow4rec = (self.recsampling[0] ** 4) * distpow4 
        if debug:
            print("DEBUG: Saving sampling grids ", end="")
            print(".", end="")
            outfname=self.outpath+self.outfilenameprefix+"_g1_float64.raw"
            fh = open(outfname, "bw")
            (self.distpow1rec).astype("float64").tofile(fh)
            print(".", end="")
            outfname=self.outpath+self.outfilenameprefix+"_g2_float64.raw"
            fh = open(outfname, "bw")
            (self.distpow2rec).astype("float64").tofile(fh)
            print(".", end="")
            outfname=self.outpath+self.outfilenameprefix+"_g3_float64.raw"
            fh = open(outfname, "bw")
            (self.distpow3rec).astype("float64").tofile(fh)
            print(".", end="")
            outfname=self.outpath+self.outfilenameprefix+"_g4_float64.raw"
            fh = open(outfname, "bw")
            (self.distpow4rec).astype("float64").tofile(fh)
            print()
        return
            
    def PrepareAlignmentFilter(self):
        #
        # tophat or gaussian filter in reciprocal space for alignment purposes
        #
        print("+ --- Preparing alignment filter ")
        ny=self.recsampling[0]*self.dim[0]/2
        hi=self.alignpass[1]/ny
        lo=self.alignpass[0]/ny
        if (self.alignfilttype == 'gaussian'):
            print("Gaussian bandpass filter, low - high sigma (% of Ny): {0:6.2f}-{1:6.2f}".format(lo*100,hi*100))
            self.alignfilt=bandpass(self.dim,lo,hi)
        else:
            print("Tophat filter, low - high (% of Ny): "+str(lo*100)+" - "+str(hi*100))
            self.alignfilt=np.copy(self.distpow2rec)
            self.alignfilt = (self.distpow2rec <= (self.alignpass[1] * self.alignpass[1])) *  (self.distpow2rec > (self.alignpass[0] * self.alignpass[0]))
        return

    def PrepareOutputFilter(self):
        #
        # tophat or gaussian filter in reciprocal space for alignment purposes
        #
        if not (self.outpass is None):
            print("+ --- Preparing output filter ")
            ny=self.recsampling[0]*self.dim[0]/2
            hi=self.alignpass[1]/ny
            lo=self.alignpass[0]/ny
            print("Tophat filter, low - high (% of Ny): "+str(lo*100)+" - "+str(hi*100))
            self.outfilt=np.copy(self.distpow2rec)
            self.outfilt = (self.distpow2rec <= (self.outpass[1] * self.outpass[1])) *  (self.distpow2rec > (self.outpass[0] * self.outpass[0]))
        return
    
    def PrepareRefinementFilter(self):
        #
        # tophat or gaussian filter in reciprocal space for refinement purposes
        #
        print("+ --- Preparing refinement filter ")
        ny=self.recsampling[0]*self.dim[0]/2
        hi=self.refinementpass[1]/ny
        lo=self.refinementpass[0]/ny
        if (self.refinementfilttype == 'gaussian'):
            print("Gaussian bandpass filter for refinement, low - high sigma (% of Ny): {0:6.2f}-{1:6.2f}".format(lo*100,hi*100))
            self.refinementfilt=bandpass(self.dim,lo,hi)
        else:
            print("Tophat filter for refinement, low - high (% of Ny): "+str(lo*100)+" - "+str(hi*100))
            self.refinementfilt=np.copy(self.distpow2rec)
            self.refinementfilt = (self.distpow2rec <= (self.refinementpass[1] * self.refinementpass[1])) *  (self.distpow2rec > (self.refinementpass[0] * self.refinementpass[0]))
        return

    
    def PrepareAutoFocusFilter(self):
        #
        # tophat or gaussian filter in reciprocal space for refinement purposes
        #
        print("+ --- Preparing autofocus filter ")
        ny=self.recsampling[0]*self.dim[0]/2
        hi=self.autofocpass[1]/ny
        lo=self.autofocpass[0]/ny
        if (self.refinementfilttype == 'gaussian'):
            print("Gaussian bandpass filter for refinement, low - high sigma (% of Ny): {0:6.2f}-{1:6.2f}".format(lo*100,hi*100))
            self.autofocfilt=bandpass(self.dim,lo,hi)
        else:
            print("Tophat filter for refinement, low - high (% of Ny): "+str(lo*100)+" - "+str(hi*100))
            self.autofocfilt=np.copy(self.distpow2rec)
            self.autofocfilt = (self.distpow2rec <= (self.autofocpass[1] * self.autofocpass[1])) *  (self.distpow2rec > (self.autofocpass[0] * self.autofocpass[0]))
        return

    
    def PrepareTransferFctns(self):
        print("+ --- Preparing transfer functions ")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        WAF_st = (2 * numpy.pi * 0.25 * ((self.lamda) ** 3)) * self.Cs * self.distpow4rec  # the second term of wave aberration functions
        WAF_st_3D = np.repeat(WAF_st[:, :, np.newaxis], NumSteps, axis=2)
        Z_reshape = (self.foci).reshape((1, 1, NumSteps))
        distpow2rec_3D = np.repeat(self.distpow2rec[:, :, np.newaxis], NumSteps, axis=2) 
        WAF_ft = (2 * numpy.pi * 0.5 * self.lamda) * distpow2rec_3D * Z_reshape  # the first term of wave aberration functions
        WAF = WAF_ft + WAF_st_3D  # wave aberration functions for the set of defocus values Z=(Z1,Z2,Z3, …., length of Z)
        self.tn = np.exp(-1j * WAF)
        # IMPORTANT: filter transfer functions to cut off high frequency oscillations
        # print("transfer function cutoff: ",self.reclimit)
        filt = (self.distpow2rec <= (self.reclimit * self.reclimit))
        for ii in range(0, NumSteps):
            self.tn[:, :, ii] = self.tn[:, :, ii] * filt
        #if debug:
        #    print("DEBUG: Saving transfer functions", end="")
        #    for ii in range(0, NumSteps):
        #        print(".", end="")
        #        if (NumSteps <= self.NImages):
        #            outfname=self.outpath+self.outfilenameprefix+"_tn_complex64_"+str(self.imrange[ii]).zfill(3)+".raw"
        #        else:
        #            # that's the case when StepAnalysis calls this function
        #            outfname=self.outpath+self.outfilenameprefix+"_tn_complex64_"+str(ii).zfill(3)+".raw"
        #        fh = open(outfname, "bw")
        #        (self.tn[:, :, ii]).astype(np.csingle).tofile(fh)
        #    print()
        return

    def PrepareFilterFctns(self):
        print("+ --- Preparing backprojection filters")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        tncc = (self.tn).conj()
        tn_p2 = self.tn *self. tn
        tncc_p2 = tncc * tncc
        # calculate the denominator
        # sum up the squared transfer functions
        tn_p2sum = tn_p2.sum(axis=2)
        tncc_p2sum = tncc_p2.sum(axis=2)
        Denom=np.zeros([self.dim[0], self.dim[1]],dtype=np.csingle)
        # D = numpy.real(NumSteps- (1. / NumSteps) * tn_p2sum * tncc_p2sum)
        # Denom = (NumSteps- (1. / NumSteps) * tn_p2sum * tncc_p2sum)
        Denom = NumSteps-(1. / NumSteps)* tn_p2sum * tncc_p2sum
        # analyze D for small numbers
        indicesofnonsmall = (numpy.abs(Denom) > self.eps).nonzero()
        indicesofsmall = (numpy.abs(Denom) <= self.eps).nonzero()
        #
        # create an array for filter functions
        #
        self.fn = np.zeros((self.dim[0], self.dim[1], NumSteps), dtype=np.csingle)
        for ii in range(0, NumSteps):
            tmp = (tncc[:, :, ii] - self.tn[:, :, ii] * (1. / NumSteps) * tncc_p2sum)
            tmp[indicesofnonsmall] = tmp[indicesofnonsmall] / Denom[indicesofnonsmall]
            tmp[indicesofsmall] = 0.0
            self.fn[:, :, ii] = tmp
        if debug:
            print("DEBUG: Saving filter functions", end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                outfname=self.outpath+self.outfilenameprefix+"_fn_complex64_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                (self.fn[:, :, ii]).astype(np.csingle).tofile(fh)
            print()
            print("DEBUG: Saving filter denominator function", end="")
            outfname=self.outpath+self.outfilenameprefix+"_fndenom_complex64.raw"
            fh = open(outfname, "bw")
            Denom.astype(np.csingle).tofile(fh)
            print()
            return
        
    def PrepareEnvelopeFctns(self):
        print("+ --- Preparing envelope functions ")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        self.temporal_env = np.zeros([self.dim[0],self.dim[1]],dtype=np.double)
        self.spatial_env = np.zeros([self.dim[0],self.dim[1], NumSteps],dtype=np.double)
        self.imagespread_env = np.zeros([self.dim[0],self.dim[1]],dtype=np.double)
        self.tot_env = np.zeros([self.dim[0],self.dim[1], NumSteps],dtype=np.double)
        self.invtot_env = np.zeros([self.dim[0],self.dim[1], NumSteps],dtype=np.double)
        #
        self.temporal_env=np.exp(-((0.5 * numpy.pi * self.lamda * self.focspread) ** 2) * self.distpow4rec)
        self.imagespread_env=np.exp(-2. * (numpy.pi * 0.001*self.imagespread) ** 2 * self.distpow2rec)
        
        filt = (self.distpow2rec <= (self.reclimit * self.reclimit))
        tmp=np.zeros([self.dim[0],self.dim[1]],dtype=np.double)
        envcutoff=np.zeros(NumSteps)
        for ii in range(0, NumSteps):
            # self.spatial_env[:, :, ii] = np.exp(-((numpy.pi * 0.001*self.sc) ** 2) * (self.foci[ii] * self.distpow1rec + self.Cs * self.lamda*self.lamda * self.distpow3rec + self.C5 * self.lamda^4 * self.distpow3rec * self.distpow2rec) ** 2)
            self.spatial_env[:, :, ii] = np.exp(-((numpy.pi * 0.001*self.semiconv) ** 2) * (self.foci[ii] * self.distpow1rec + self.Cs * self.lamda*self.lamda * self.distpow3rec) ** 2)
            self.tot_env[:,:,ii]=self.spatial_env[:,:,ii] * self.imagespread_env * self.temporal_env
            # IMPORTANT: filter transfer functions to cut off high frequency oscillations
            # When does the envelope fall below a certain value %?
            indicesofsmall = (numpy.abs(self.tot_env[:,:,ii]) <= self.envlimit).nonzero()
            tmp=1./self.tot_env[:,:,ii]
            tmp[indicesofsmall]=0.
            envcutoff[ii]=np.amin(self.distpow1rec[indicesofsmall])
            self.invtot_env[:,:,ii]=tmp
        print("Minimum/Maximum envelope cutoff frequency (1/nm): {0:8.3f}, {1:8.3f}".format(np.amin(envcutoff),np.amax(envcutoff)))    
        # if debug:
        #     print("DEBUG: Saving envelope functions", end="")
        #     outfname=self.outpath+self.outfilenameprefix+"_ispreadenv_float32.raw"
        #     fh = open(outfname, "bw")
        #     (self.imagespread_env).astype(np.single).tofile(fh)
        #     outfname=self.outpath+self.outfilenameprefix+"_tempenv_float32.raw"
        #     fh = open(outfname, "bw")
        #     (self.temporal_env).astype(np.single).tofile(fh)
        #     for ii in range(0, NumSteps):
        #         print(".", end="")
        #         outfname=self.outpath+self.outfilenameprefix+"_spatialenv_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
        #         fh = open(outfname, "bw")
        #         (self.spatial_env[:, :, ii]).astype(np.single).tofile(fh)
        #         outfname=self.outpath+self.outfilenameprefix+"_invtotenv_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
        #         fh = open(outfname, "bw")
        #         (self.invtot_env[:, :, ii]).astype(np.single).tofile(fh)
        #     print()
        return


    def LoadImages(self):
        print("+ --- Loading  images ")
        print("Loading  and transforming images: ", end="")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        self.data = np.empty([self.dim[0], self.dim[1], NumSteps])
        self.fftdata = np.empty([self.dim[0], self.dim[1], NumSteps], dtype=np.cdouble)
        # load and prepare fourier transforms
        for ii in range(0, NumSteps):
            print(".", end="")
            tmp = np.fromfile(self.inpath + self.infilenameprefix + str(self.imrange[ii]).zfill(3) + ".raw", dtype=np.single, count=-1)
            if self.fixhotpix:
                tmp = np.reshape(tmp, (self.dim[0], self.dim[1]))
                hot_pixels,tmp=find_outlier_pixels(tmp)
                tmp = np.reshape(tmp, (self.dim[0]*self.dim[1]))
            #  normalize image intensity
            tmp = tmp/tmp.mean()
            self.data[:, :, ii] = np.reshape(tmp, (self.dim[0], self.dim[1]))
            self.fftdata[:, :, ii] = np.fft.fft2(self.data[:, :, ii])            
        # initialize essential arrays
        self.sfftdata = np.copy(self.fftdata)
        self.fftsimdata = np.copy(self.fftdata)
        self.psi=np.zeros([self.dim[0], self.dim[1]], dtype=np.cdouble)
        self.displacements=np.zeros([2, NumSteps], dtype=np.single)
        print()        
        return

    def PreAlignImagesLowDose(self):
        if not(self.prealignswitch):
            print("Warning: Pre-aligning is switched off. ")
            return
        print("+ --- Pre-aligning low dose images: ")
        # pre-align images using a sequential cross-correlation
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        # take central quarter of images for a common frame
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        print("")
        print("Reference frame coordinates:")
        print("lower left=[{0:4d},{1:4d}]".format(startx,starty))
        print("upper right=[{0:4d},{1:4d}]".format(startx+cropx,starty+cropy))
        # if contrastinversion is expected, prepare a contrast inversion array of appropritae sign
        # first analyse the imaginary part of the transfer function, is it mostly positive or negative over the alignment filter?
        #
        invert=np.zeros(NumSteps,dtype=np.int32)
        if self.invertforalign:
            print("")
            print("Pre-alignment contrast inversion")
            print("------------------")
            print("   image    flag        ")
            print("------------------")
            for ii in range(0,NumSteps):
                if np.sum((self.tn[:,:,ii]*self.alignfilt).imag) < 0:
                    invert[ii]=-1
                else:
                    invert[ii]=1
                print("{0:5d}    {1:}".format(self.imrange[ii],invert[ii]))    
        print("")
        print("Coarse pixel-wise alignment ")
        print("----------------------")
        print(" image    dx      dy     ")
        print("----------------------")
        #
        for ii in range(0, NumSteps):
            # reference image creation
            count=0
            refim=np.zeros([cropx,cropy],dtype=np.single)
            for kk in range(ii-self.ravg,ii-1):
                # exclude the image ii
                if (kk >= 0):
                    tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                    if (invert[kk] == -1):
                        refim -= (tmpim[startx:startx+cropx,starty:starty+cropy]).real
                    else:
                        refim += (tmpim[startx:startx+cropx,starty:starty+cropy]).real
                    count +=1
                    # print("kk={0:d}".format(kk)) 
            for kk in range(ii+1,ii+self.ravg):
                if (kk < NumSteps):
                    tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                    if (invert[kk] == -1):
                        refim -= (tmpim[startx:startx+cropx,starty:starty+cropy]).real
                    else:
                        refim += (tmpim[startx:startx+cropx,starty:starty+cropy]).real
                    count +=1
                    # print("kk={0:d}".format(kk)) 
            # refim /= count # we don't need to normalize, adding is fine
            if debug:
                outfname=self.outpath+self.outfilenameprefix+"_refim_"+str(self.imrange[ii]).zfill(3)+".raw"
                # print("Saving reference image to {0}.".format(outfname)) 
                fh = open(outfname, "bw")
                refim.astype(np.single).tofile(fh)
            if (ii > 0):
                # cumulate shifts
                self.displacements[0,ii] = self.displacements[0,ii-1]
                self.displacements[1,ii] = self.displacements[1,ii-1]
            im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
            im=invert[ii]*im[startx:startx+cropx,starty:starty+cropy]
            # correlate images
            cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
            self.displacements[0,ii] += - maxpos[0]
            self.displacements[1,ii] += - maxpos[1]
            del refim
            if debug:
                outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                cim.astype(np.single).tofile(fh)
        meandispl = (self.displacements).mean(axis=1)
        self.displacements[0,:] -= meandispl[0]
        self.displacements[1,:] -= meandispl[1]
        for ii in range(0, NumSteps):
            print("{0:5d}{1:8.3f}{2:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii]))  
        print("---------------------------------------")
        # check shift for boundary wrap in the reference frame
        minx=nextint(np.amin( -self.displacements[0,:]+startx))
        maxx=nextint(np.amax( -self.displacements[0,:]+startx+cropx))
        miny=nextint(np.amin( -self.displacements[1,:]+starty))
        maxy=nextint(np.amax( -self.displacements[1,:]+starty+cropy))
        print("Common frame coordinates:")
        print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
        print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
        if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
        # apply shifts to fft
        print("Applying shifts ",end="")
        for ii in range(0, NumSteps):
            print(".", end="")
            self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
        print()
        # sys.exit("finished prealignment")
        # do more rounds
        jj=1
        # smax=np.amax(np.amax(np.abs(self.displacements),axis=1),axis=0)
        smax=1
        while (smax > 0.5) and (jj < self.NMaxPreali):
            print("Refining pixel-wise alignment ")
            corrdisp=np.zeros([2,NumSteps])
            print("------------------------------------")
            print(" image    dx      dy    ddx   ddy   ")
            print("------------------------------------")
            if (self.dispcorrelation=='sequential-com'):
                for ii in range(0, NumSteps):
                    # reference image creation
                    refim=np.zeros([cropx,cropy],dtype=np.single)
                    for kk in range(ii-self.ravg,ii):
                        if (kk >= 0):
                            tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                            refim += invert[kk]*(tmpim[startx:startx+cropx,starty:starty+cropy]).real
                            count +=1
                    for kk in range(ii+1,ii+self.ravg):
                        if (kk < NumSteps):
                            tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                            refim += invert[kk]*(tmpim[startx:startx+cropx,starty:starty+cropy]).real
                            count +=1
                    refim /= count
                    if (ii > 0):
                        corrdisp[0,ii] = corrdisp[0,ii-1]
                        corrdisp[1,ii] = corrdisp[1,ii-1]
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=invert[ii]*im[startx:startx+cropx,starty:starty+cropy]
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    cim=np.roll(cim,-(cropx // 2),axis=0)
                    cim=np.roll(cim,-(cropy // 2),axis=1)
                    fracdisp=np.zeros([2])
                    fracdisp=iterate_COM(cim,self.alignprecision,self.comkernel,self.commaxiter)
                    corrdisp[0,ii] += - fracdisp[0] 
                    corrdisp[1,ii] += - fracdisp[1]
                    del refim
            if (self.dispcorrelation=='sequential'):
                for ii in range(1, NumSteps):
                    # reference image creation
                    refim=np.zeros([cropx,cropy],dtype=np.single)
                    for kk in range(ii-self.ravg,ii):
                        if (kk >= 0):
                            tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                            refim += invert[kk]*(tmpim[startx:startx+cropx,starty:starty+cropy]).real
                            count +=1
                    for kk in range(ii+1,ii+self.ravg):
                        if (kk < NumSteps):
                            tmpim = np.fft.ifft2(self.sfftdata[:, :, kk]*self.alignfilt)
                            refim += invert[kk]*(tmpim[startx:startx+cropx,starty:starty+cropy]).real
                            count +=1
                    refim /= count
                    if (ii > 0):
                        corrdisp[0,ii] = corrdisp[0,ii-1]
                        corrdisp[1,ii] = corrdisp[1,ii-1]
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=invert[ii]*im[startx:startx+cropx,starty:starty+cropy]
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    corrdisp[0,ii] += - maxpos[0] 
                    corrdisp[1,ii] += - maxpos[1]
                    del refim
            if (self.dispcorrelation=='sequential-pca'):
                # get new pca ref
                eigenvals,pca=ImagePCA(self.data, [startx,starty,cropx,cropy])
                refim=pca[0,:,:]
                for ii in range(0, NumSteps):
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=im[startx:startx+cropx,starty:starty+cropy]
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    corrdisp[0,ii] += - maxpos[0] 
                    corrdisp[1,ii] += - maxpos[1]
            meandispl = corrdisp.mean(axis=1)
            corrdisp[0,:] -= meandispl[0]
            corrdisp[1,:] -= meandispl[1]
            for ii in range(0, NumSteps):
                    self.displacements[0,ii] += -corrdisp[0,ii]
                    self.displacements[1,ii] += -corrdisp[1,ii]
                    print("{0:5d}{1:8.3f}{2:8.3f}{3:8.3f}{4:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii],corrdisp[0,ii],corrdisp[1,ii]))
            print("---------------------------------------")
            smax=np.amax(np.amax(np.abs(corrdisp),axis=1),axis=0)
            print("Maximum displacement in x,y: {0:5.3f}".format(smax))
            srms=np.std(corrdisp,axis=1)
            print("RMS displacement in x,y: {0:5.3f},{1:5.3f}".format(srms[0],srms[1]))
            # check shift for boundary wrap in the reference frame
            minx=nextint(np.amin( -self.displacements[0,:]+startx))
            maxx=nextint(np.amax( -self.displacements[0,:]+startx+cropx))
            miny=nextint(np.amin( -self.displacements[1,:]+starty))
            maxy=nextint(np.amax( -self.displacements[1,:]+starty+cropy))
            print("Common frame coordinates:")
            print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
            print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
            if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
            # apply shifts to fft
            print("Applying shifts ",end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
                if debug:
                    outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                    fh = open(outfname, "bw")
                    tmp=np.fft.ifft2(self.sfftdata[:, :, ii])
                    tmp.astype(np.csingle).tofile(fh)
            print()
            jj += 1
        # ------------------ end sequential correlation    
        if (smax > 1):
            print("Warning: Prealignment did not converge to less than one pixel shift for at least one image.")
        for ii in range(0, NumSteps):
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
        return


    def PreAlignImagesLowDoseCommonFrame(self):
        if not(self.prealignswitch):
            print("Warning: Pre-aligning is switched off. ")
            return
        print("+ --- Pre-aligning low dose images to common aligned average: ")
        # pre-align images using a sequential cross-correlation
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        # take central quarter of images for a common frame
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        print("")
        print("Reference frame coordinates:")
        print("lower left  =[{0:4d},{1:4d}]".format(startx,starty))
        print("upper right =[{0:4d},{1:4d}]".format(startx+cropx,starty+cropy))
        print("frame width =[{0:4d},{1:4d}]".format(self.framewidth[0],self.framewidth[1]))
        print("frame center=[{0:4d},{1:4d}]".format(math.floor((startx+startx+cropx)/2),math.floor((starty+starty+cropy)/2)))
        # if contrastinversion is expected, prepare a contrast inversion array of appropritae sign
        # first analyse the imaginary part of the transfer function, is it mostly positive or negative over the alignment filter?
        #
        invert=np.zeros(NumSteps,dtype=np.int32)
        if self.invertforalign:
            print("")
            print("Pre-alignment contrast inversion")
            print("------------------")
            print("   image    flag        ")
            print("------------------")
            for ii in range(0,NumSteps):
                if np.sum((self.tn[:,:,ii]*self.alignfilt).imag) < 0:
                    invert[ii]=-1
                else:
                    invert[ii]=1
                    print("{0:5d}    {1:}".format(self.imrange[ii],invert[ii]))    
            print("")
        print("Creating boxcar averages ")
        tmpim=np.zeros([self.dim[0],self.dim[1],NumSteps],dtype=np.single)
        for ii in range(0, NumSteps):
              if (invert[ii] == -1):
                    tmpim[:,:,ii] = -(np.fft.ifft2(self.sfftdata[:, :, ii])).real
              else:
                    tmpim[:,:,ii] = (np.fft.ifft2(self.sfftdata[:, :, ii])).real
        print("Averaging over {0} neighboring images.".format(self.ravg))
        bavg_mean = binArray(tmpim, 2, self.ravg, self.ravg, np.mean)
        print(bavg_mean.shape)
        #print("Calculating principal component.")
        #eigenvals, pca = ImagePCA(bavg_mean)
        #print("Eigenvalues:")
        #print(eigenvals)
        #refim=pca[0,:,:]
        #refim=bavg_mean[:,:,0]
        x,y,NumAvg= bavg_mean.shape
        if debug:
                print("Saving boxcar average images") 
                for kk in range(0, NumAvg):
                        outfname=self.outpath+self.outfilenameprefix+"_boxcaravg_float32_"+str(kk).zfill(3)+".raw"
                        fh = open(outfname, "bw")
                        bavg_mean[:,:,kk].astype(np.single).tofile(fh)
        # Pre-align the boxcar averaged images        
        #
        print("Pre-aligning averaged images.")
        displacements=np.zeros([2,NumAvg])
        ref=np.fft.ifft2(np.fft.fft2(bavg_mean[:,:,0])*self.alignfilt)
        ref=ref[startx:startx+cropx,starty:starty+cropy]
        for ii in range(1, NumAvg):
            displacements[0,ii] = displacements[0,ii-1]
            displacements[1,ii] = displacements[1,ii-1]
            im=np.fft.ifft2(np.fft.fft2(bavg_mean[:,:,ii])*self.alignfilt)
            im=im[startx:startx+cropx,starty:starty+cropy]            
            cim,maxpos,maxval=self.imcorrelation(im.real,ref.real)
            displacements[0,ii] += - maxpos[1]
            displacements[1,ii] += - maxpos[0]
            ref=im
        meandispl = (displacements).mean(axis=1)
        displacements[0,:] -= meandispl[0]
        displacements[1,:] -= meandispl[1]
        print("")
        print("Coarse, sequential pixel-wise alignment ")
        print("----------------------")
        print(" image    dx      dy     ")
        print("----------------------")
        for ii in range(0, NumAvg):
            print("{0:5d}{1:8.3f}{2:8.3f}".format(ii,displacements[0,ii],displacements[1,ii]))  
        print("---------------------------------------")
        # apply shifts to fft
        print("Applying shifts ",end="")
        for ii in range(0, NumAvg):
            print(".", end="")
            bavg_mean[:,:,ii]=(np.fft.ifft2(shiftfft2d(np.fft.fft2(bavg_mean[:, :, ii]),+displacements[0,ii],+displacements[1,ii]))).real
        print()
        if debug:
            print("Saving aligned boxcar average images") 
            for kk in range(0, NumAvg):
                outfname=self.outpath+self.outfilenameprefix+"_boxcaravg_ali_float32_"+str(kk).zfill(3)+".raw"
                fh = open(outfname, "bw")
                bavg_mean[:,:,kk].astype(np.single).tofile(fh)
        # Refine alignment: take average frame as reference and align to that
        # loop
        print("Refining pre-alignment of averaged images.")
        refdisplacements=np.zeros([2,NumAvg])
        fracdisp=np.zeros([2])
        jj=1
        smax=2. # do at least one round
        while (smax > 0.1) and (jj < self.NMaxPreali):
            refim=np.mean(bavg_mean, axis=2)
            refdisplacements[:,:]=0.
            fracdisp[:]=0.
            # re-run alignment
            ref=refim[startx:startx+cropx,starty:starty+cropy]
            fracdisp=np.zeros([2])
            for ii in range(0, NumAvg):
                im=np.fft.ifft2(np.fft.fft2(bavg_mean[:,:,ii])*self.alignfilt)
                im=im[startx:startx+cropx,starty:starty+cropy]
                cim,maxpos,maxval=self.imcorrelation(im.real, ref.real)
                cim=np.roll(cim,-(cropx // 2),axis=0)
                cim=np.roll(cim,-(cropy // 2),axis=1)
                fracdisp=iterate_COM(cim,self.alignprecision,self.comkernel,self.commaxiter)
                refdisplacements[0,ii] = - fracdisp[0]
                refdisplacements[1,ii] = - fracdisp[1]
                displacements[0,ii] += - fracdisp[0]
                displacements[1,ii] += - fracdisp[1]
            print("")
            print("Refined alignment to common frame ")
            print("---------------------------------------")
            print(" image    dx    dy    ddx      ddy     ")
            print("---------------------------------------")
            for ii in range(0, NumAvg):
                print("{0:5d}{1:8.3f}{2:8.3f}{3:8.3f}{4:8.3f}".format(ii,displacements[0,ii],displacements[1,ii],refdisplacements[0,ii],refdisplacements[1,ii]))  
            print("---------------------------------------")
            smax=np.amax(np.amax(np.abs(refdisplacements),axis=1),axis=0)
            print("Maximum displacement in x,y: {0:5.3f}".format(smax))
            srms=np.std(refdisplacements,axis=1)
            print("RMS displacement in x,y: {0:5.3f},{1:5.3f}".format(srms[0],srms[1]))

            # apply shifts to fft
            print("Applying shifts ",end="")
            for ii in range(0, NumAvg):
                print(".", end="")
                bavg_mean[:,:,ii]=(np.fft.ifft2(shiftfft2d(np.fft.fft2(bavg_mean[:, :, ii]),+refdisplacements[0,ii],+refdisplacements[1,ii]))).real
            print()

            jj += 1
        if debug:
            print("Saving aligned boxcar average images") 
            for kk in range(0, NumAvg):
                outfname=self.outpath+self.outfilenameprefix+"_boxcaravg_ali_float32_"+str(kk).zfill(3)+".raw"
                fh = open(outfname, "bw")
                bavg_mean[:,:,kk].astype(np.single).tofile(fh)
            # Refine alignment: take average frame as reference and align to that
            refim=np.mean(bavg_mean, axis=2)
            if debug:
                print("Saving aligned average of boxcar image")
                outfname=self.outpath+self.outfilenameprefix+"_avgboxcaravg_ali_float32.raw"
                fh = open(outfname, "bw")
                refim.astype(np.single).tofile(fh)
        # check shift for boundary wrap in the reference frame
        minx=nextint(np.amin( -displacements[0,:]+startx))
        maxx=nextint(np.amax( -displacements[0,:]+startx+cropx))
        miny=nextint(np.amin( -displacements[1,:]+starty))
        maxy=nextint(np.amax( -displacements[1,:]+starty+cropy))
        print("Common frame coordinates:")
        print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
        print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
        if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
        # apply shifts to fft
        # now we have a common reference, we can predict the single image shift by interpolation
        # this means estimating self.displacements[0,ii]
        print("")
        print("Applying shifts to original series ",end="")
        print("----------------------")
        print(" image    dx      dy     ")
        print("----------------------")
        for ii in range(0, NumSteps):
            kk= ii // self.ravg
            if (kk >= NumAvg):
                kk=NumAvg-1
            print("{0:5d}{1:8.3f}{2:8.3f}".format(ii,displacements[0,kk],displacements[1,kk]))
            self.displacements[0,ii]= -displacements[0,kk] # self.displacements has the opposite sign!!!
            self.displacements[1,ii]= -displacements[1,kk]
            self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],displacements[0,kk],displacements[1,kk])
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
            if debug:
                outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                tmp=np.abs(np.fft.ifft2(self.sfftdata[:, :, ii]))
                tmp.astype(np.csingle).tofile(fh)
        print("----------------------")
        print()
        # now the original images are pre-aligned, we can refine the alignment, but with the common frame image
                # do more rounds
        jj=1
        #smax=np.amax(np.amax(np.abs(self.displacements),axis=1),axis=0)
        smax=2. # do at least one round
        #while (smax > 0.5) and (jj < self.NMaxPreali):
        ref=refim[startx:startx+cropx,starty:starty+cropy]
        print("Refining alignment of individual images")
        while (smax > 0.5) and (jj < self.NMaxPreali):
            print(" ")
            print("Iteration "+str(jj))
            corrdisp=np.zeros([2,NumSteps])
            print("------------------------------------")
            print(" image    dx      dy    ddx   ddy   ")
            print("------------------------------------")
            for ii in range(1, NumSteps):
                im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                im=im[startx:startx+cropx,starty:starty+cropy]
                if (invert[ii] == -1):
                    im=-1.*im
                cim,maxpos,maxval=self.imcorrelation(im.real, ref.real)
                cim=np.roll(cim,-(cropx // 2),axis=0)
                cim=np.roll(cim,-(cropy // 2),axis=1)
                fracdisp=np.zeros([2])
                fracdisp=iterate_COM(cim,self.alignprecision,self.comkernel,self.commaxiter)
                corrdisp[0,ii] += - fracdisp[0] 
                corrdisp[1,ii] += - fracdisp[1]
                print("{0:5d}{1:8.3f}{2:8.3f}{3:8.3f}{4:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii],corrdisp[0,ii],corrdisp[1,ii]))
                self.displacements[0,ii] += -corrdisp[0,ii]
                self.displacements[1,ii] += -corrdisp[1,ii]
            print("---------------------------------------")
            smax=np.amax(np.amax(np.abs(corrdisp),axis=1),axis=0)
            print("Maximum displacement in x,y: {0:5.3f}".format(smax))
            srms=np.std(corrdisp,axis=1)
            print("RMS displacement in x,y: {0:5.3f},{1:5.3f}".format(srms[0],srms[1]))
            # check shift for boundary wrap in the reference frame
            minx=nextint(np.amin( -self.displacements[0,:]+startx))
            maxx=nextint(np.amax( -self.displacements[0,:]+startx+cropx))
            miny=nextint(np.amin( -self.displacements[1,:]+starty))
            maxy=nextint(np.amax( -self.displacements[1,:]+starty+cropy))
            print("Common frame coordinates:")
            print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
            print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
            if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
            # apply shifts to fft
            print("Applying shifts ",end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
            print()
            jj += 1
        # ------------------ end sequential correlation
        if (smax > 1):
            print("Warning: Prealignment did not converge to less than one pixel shift for at least one image.")
        for ii in range(0, NumSteps):
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
            if debug:
                outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                tmp=np.abs(np.fft.ifft2(self.sfftdata[:, :, ii]))
                tmp.astype(np.csingle).tofile(fh)
        return
        # sys.exit("finished prealignment")

    def SaveBoxcarSumImages(self, len):
        print(len)
        if (len > 1):
            NumSteps = np.size(self.foci, 0)  # the number of defocus steps
            print("+ --- Creating boxcar sums of pre-aligned images: ")
            # shift images
            print("Summing over {0} neighboring images.".format(len))
            count=0
            maxcount=math.floor(NumSteps/len)
            bcim=np.zeros([self.dim[0],self.dim[1],maxcount],dtype=np.single)
            for count in range(0,maxcount):
                print(count)
                kk=count*len
                for jj in range(kk,kk+len):
                    bcim[:,:,count] += (np.fft.ifft2(self.sfftdata[:, :, jj])).real
                    print(jj)
                    # print("kk={0:d}".format(kk))
            x,y,NumAvg= bcim.shape
            print("Saving boxcar sum images") 
            for kk in range(0, NumAvg):
                outfname=self.outpath+self.outfilenameprefix+"_boxcar"+str(len)+"_float32_"+str(kk).zfill(3)+".raw"
                fh = open(outfname, "bw")
                bcim[:,:,kk].astype(np.single).tofile(fh)
        return
                
    
    def PreAlignImages(self):
        if not(self.prealignswitch):
            print("Warning: Pre-aligning is switched off. ")
            return
        if self.commonframe:
            self.PreAlignImagesLowDoseCommonFrame()
            return
        #if (self.ravg > 0):
        #    self.PreAlignImagesLowDose()
        #    return
        print("+ --- Pre-aligning images: ")
        # pre-align images using a sequential cross-correlation
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        # take central quarter of images for a common frame
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        # if contrastinversion is expected, prepare a contrast inversion array of appropritae sign
        # first analyse the imaginary part of the transfer function, is it mostly positive or negative over the alignment filter?
        #
        invert=np.zeros(NumSteps,dtype=bool)
        if self.invertforalign:
            print("")
            print("Pre-alignment contrast inversion")
            print("------------------")
            print("   image    flag        ")
            print("------------------")
            for ii in range(0,NumSteps):
                if np.sum((self.tn[:,:,ii]*self.alignfilt).imag) < 0:
                    invert[ii]=True
                print("{0:5d}    {1:}".format(self.imrange[ii],invert[ii]))    
        print("")
        print("Reference frame coordinates:")
        print("lower left=[{0:4d},{1:4d}]".format(startx,starty))
        print("upper right=[{0:4d},{1:4d}]".format(startx+cropx,starty+cropy))
        print("")
        print("Coarse pixel-wise alignment ")
        print("----------------------")
        print(" image    dx      dy     ")
        print("----------------------")
        refim=np.fft.ifft2(self.sfftdata[:, :, 0]*self.alignfilt)
        refim=refim[startx:startx+cropx,starty:starty+cropy]
        if invert[0]:
            refim=-1.*refim
        for ii in range(1, NumSteps):
            self.displacements[0,ii] = self.displacements[0,ii-1]
            self.displacements[1,ii] = self.displacements[1,ii-1]
            im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
            im=im[startx:startx+cropx,starty:starty+cropy]
            if invert[ii]:
                im=-1.*im
            cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
            self.displacements[0,ii] += - maxpos[0]
            self.displacements[1,ii] += - maxpos[1]
            refim=im
            if debug:
                outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                cim.astype(np.single).tofile(fh)
        meandispl = (self.displacements).mean(axis=1)
        self.displacements[0,:] -= meandispl[0]
        self.displacements[1,:] -= meandispl[1]
        for ii in range(0, NumSteps):
            print("{0:5d}{1:8.3f}{2:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii]))  
        print("---------------------------------------")
        # check shift for boundary wrap in the reference frame
        minx=nextint(np.amin( -self.displacements[0,:]+startx))
        maxx=nextint(np.amax( -self.displacements[0,:]+startx+cropx))
        miny=nextint(np.amin( -self.displacements[1,:]+starty))
        maxy=nextint(np.amax( -self.displacements[1,:]+starty+cropy))
        print("Common frame coordinates:")
        print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
        print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
        if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
        # apply shifts to fft
        print("Applying shifts ",end="")
        for ii in range(0, NumSteps):
            print(".", end="")
            self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
        print()
        # do more rounds
        kk=1
        #smax=np.amax(np.amax(np.abs(self.displacements),axis=1),axis=0)
        smax=2. # do at least one round
        while (smax > 0.5) and (kk < self.NMaxPreali):
            print("Refining pixel-wise alignment ")
            corrdisp=np.zeros([2,NumSteps])
            print("------------------------------------")
            print(" image    dx      dy    ddx   ddy   ")
            print("------------------------------------")
            if (self.dispcorrelation=='sequential-com'):
                refim=np.fft.ifft2(self.sfftdata[:, :, 0]*self.alignfilt) # shifted in the first round
                refim=refim[startx:startx+cropx,starty:starty+cropy]
                if invert[0]:
                    refim=-1.*refim
                for ii in range(1, NumSteps):
                    corrdisp[0,ii] = corrdisp[0,ii-1]
                    corrdisp[1,ii] = corrdisp[1,ii-1]
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=im[startx:startx+cropx,starty:starty+cropy]
                    if invert[ii]:
                        im=-1.*im
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    cim=np.roll(cim,-(cropx // 2),axis=0)
                    cim=np.roll(cim,-(cropy // 2),axis=1)
                    fracdisp=np.zeros([2])
                    fracdisp=iterate_COM(cim,self.alignprecision,self.comkernel,self.commaxiter)
                    corrdisp[0,ii] += - fracdisp[0] 
                    corrdisp[1,ii] += - fracdisp[1]
                    refim=im
            if (self.dispcorrelation=='sequential'):
                refim=np.fft.ifft2(self.sfftdata[:, :, 0]*self.alignfilt) # shifted in the first round
                refim=refim[startx:startx+cropx,starty:starty+cropy]
                if invert[0]:
                    refim=-1.*refim
                for ii in range(1, NumSteps):
                    corrdisp[0,ii] = corrdisp[0,ii-1]
                    corrdisp[1,ii] = corrdisp[1,ii-1]
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=im[startx:startx+cropx,starty:starty+cropy]
                    if invert[ii]:
                        im=-1.*im
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    corrdisp[0,ii] += - maxpos[0] 
                    corrdisp[1,ii] += - maxpos[1]
                    refim=im
            if (self.dispcorrelation=='sequential-pca'):
                # get new pca ref
                eigenvals,pca=ImagePCA(self.data, [startx,starty,cropx,cropy])
                refim=pca[0,:,:]
                for ii in range(0, NumSteps):
                    im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
                    im=im[startx:startx+cropx,starty:starty+cropy]
                    cim,maxpos,maxval=self.imcorrelation(im.real, refim.real)
                    corrdisp[0,ii] += - maxpos[0] 
                    corrdisp[1,ii] += - maxpos[1]
            meandispl = corrdisp.mean(axis=1)
            corrdisp[0,:] -= meandispl[0]
            corrdisp[1,:] -= meandispl[1]
            for ii in range(0, NumSteps):
                    self.displacements[0,ii] += -corrdisp[0,ii]
                    self.displacements[1,ii] += -corrdisp[1,ii]
                    print("{0:5d}{1:8.3f}{2:8.3f}{3:8.3f}{4:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii],corrdisp[0,ii],corrdisp[1,ii]))
            print("---------------------------------------")
            smax=np.amax(np.amax(np.abs(corrdisp),axis=1),axis=0)
            print("Maximum displacement in x,y: {0:5.3f}".format(smax))
            srms=np.std(corrdisp,axis=1)
            print("RMS displacement in x,y: {0:5.3f},{1:5.3f}".format(srms[0],srms[1]))
            # check shift for boundary wrap in the reference frame
            minx=nextint(np.amin( -self.displacements[0,:]+startx))
            maxx=nextint(np.amax( -self.displacements[0,:]+startx+cropx))
            miny=nextint(np.amin( -self.displacements[1,:]+starty))
            maxy=nextint(np.amax( -self.displacements[1,:]+starty+cropy))
            print("Common frame coordinates:")
            print("lower left=[{0:4d},{1:4d}]".format(minx,miny))
            print("upper right=[{0:4d},{1:4d}]".format(maxx,maxy))
            if ((minx < 0) or (miny<0) or (maxx > self.dim[0]) or (maxy > self.dim[1])):
                print("Fatal error: shifts too large, check drift and frame selection.")
                exit(0)
            # apply shifts to fft
            print("Applying shifts ",end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
                if debug:
                    outfname=self.outpath+self.outfilenameprefix+"_preali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                    fh = open(outfname, "bw")
                    tmp=np.fft.ifft2(self.sfftdata[:, :, ii])
                    tmp.astype(np.csingle).tofile(fh)
            print()
            kk += 1
        # ------------------ end sequential correlation    
        if (smax > 1):
            print("Warning: Prealignment did not converge to less than one pixel shift for at least one image.")
        for ii in range(0, NumSteps):
            self.fftsimdata[:, :, ii]=self.sfftdata[:, :, ii]
        return

    def LoadPreviousFoci(self, filename=None):
        print("+ --- Loading previous focus values: ")
        if not filename:
            infname=self.outpath+self.outfilenameprefix+"_foci.csv"
        else:
            infname=filename
        print("Loading focus data from ", infname)
        if os.path.isfile(infname):
            focdata = genfromtxt(infname, delimiter=',')
            if (np.size(self.foci, 0) == np.size(focdata[:,1], 0)):
                self.foci=focdata[:,1]
            else:
                print("WARNING: Mismatch in number of foci, continuing with data in parameter file.") 
        else:
            print("WARNING: File not found, continuing with data in parameter file.")
        return

    def LoadPreviousDisplacements(self, filename=None):
        # print("+ --- Loading previous displacement values: ")
        if not filename:
            infname=self.outpath+self.outfilenameprefix+"_dxy.csv"
        else:
            infname=filename
        print("+ --- Loading displacement data from ", infname)
        if os.path.isfile(infname):
            NumIm=np.size(self.foci, 0)
            dxydata = genfromtxt(infname, delimiter=',')
            if (NumIm == np.size(dxydata[:,0], 0)):
               # align
                print("------------------------")
                print(" image    dx      dy    ")
                print("------------------------")
                for ii in range(0, NumIm):
                    self.displacements[0,ii] = dxydata[ii,1]
                    self.displacements[1,ii] = dxydata[ii,2]
                    print("{0:5d}{1:8.3f}{2:8.3f}".format(int(dxydata[ii,0]),self.displacements[0,ii],self.displacements[1,ii]))
                    self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
                print("------------------------")
                #sys.exit("Debug STOP")
            else:
                sys.exit("Mismatch in number of displacements. Check file displacement data.")            
        else:
            # no file found
            # give a warning and continue only if pre-alignment is off.
            if self.prealignswitch:
                print("File not found:", infname)
                print("Possible fixes:", infname)
                print(" - Retrieve or create the file.")
                print(" - Set Prealignment=Off under [RECONSTRUCTION] in the configuration file.")
                sys.exit("Program halted with critical error.")
            else:
                print("WARNING: File not found") 
                print("WARNING: Prealignment is off, continuing with zero shifts.")
                print("------------------------")
                print(" image    dx      dy    ")
                print("------------------------")
                NumIm=np.size(self.foci, 0)
                for ii in range(0, NumIm):
                    self.displacements[0,ii] = 0.
                    self.displacements[1,ii] = 0.
                    print("{0:5d}{1:8.3f}{2:8.3f}".format(ii,self.displacements[0,ii],self.displacements[1,ii]))
                    self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
                print("------------------------")
        return

    def LoadPreviousWav(self, filename=None):
        infname=self.outpath+self.outfilenameprefix+".wav"
        print("Loading wave function data from ", infname)
        if os.path.isfile(infname):
            tmp=np.fromfile(infname, dtype=np.csingle, count=-1, offset=0)
            tmp=np.reshape(tmp,(self.dim[0],self.dim[1]))
            self.psi=tmp.astype(np.cdouble)
        else:
            sys.exit("File not found.")
    
    def AlignImages(self):
        print("+ --- Aligning images: ")
        # align images to simulated images, then apply phase factor to fftdata in order to update sfftdata
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        # take central quarter of images 
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        deltadispl=np.zeros((2,NumSteps),dtype=np.single)
        print("--------------------------------------")
        print(" image    dx      dy     ddx     ddy  ")
        print("--------------------------------------")
        for ii in range(0, NumSteps):
            im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.alignfilt)
            im=im[startx:startx+cropx,starty:starty+cropy]
            simim=np.fft.ifft2(self.fftsimdata[:, :, ii]*self.alignfilt)
            simim=simim[startx:startx+cropx,starty:starty+cropy]
            cim,maxpos,maxval=self.wavcorrelation(im.real, simim.real)
            cimr=np.roll(cim.real,-(cropx // 2),axis=0)
            cimr=np.roll(cimr,-(cropy // 2),axis=1)
            fracdisp=np.zeros([2])
            fracdisp=iterate_COM(cimr,self.alignprecision,self.comkernel,self.commaxiter)
            ddx =  fracdisp[0]
            ddy =  fracdisp[1]
            deltadispl[0,ii]=ddx
            deltadispl[1,ii]=ddy
            print("{0:5d}{1:8.3f}{2:8.3f}{3:8.3f}{4:8.3f}".format(self.imrange[ii],self.displacements[0,ii],self.displacements[1,ii],ddx,ddy))
            self.displacements[0,ii] += ddx
            self.displacements[1,ii] += ddy
        print("---------------------------------------")
        smax=np.amax(np.amax(np.abs(deltadispl),axis=1),axis=0)
        print("Maximum displacement in x,y: {0:5.3f}".format(smax))
        srms=np.std(deltadispl,axis=1)
        print("RMS displacement in x,y: {0:5.3f},{1:5.3f}".format(srms[0],srms[1]))
        # apply shifts to fft
        print("Applying shifts ",end="")
        for ii in range(0, NumSteps):
            print(".", end="")
            self.sfftdata[:, :, ii]=shiftfft2d(self.fftdata[:, :, ii],-self.displacements[0,ii],-self.displacements[1,ii])
        print()
        if debug:
            print("DEBUG: Saving intermediate fourier transforms", end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                outfname=self.outpath+self.outfilenameprefix+"_sfft_complex64_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                (self.sfftdata[:, :, ii]).astype(np.csingle).tofile(fh)
            print()
        return
         
        
    def BackProjection(self, scale=True):
        print("+ --- Projecting filtered sum of Fourier transforms: ")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        tmp=np.empty([self.dim[0], self.dim[1]], dtype=np.cdouble)
        # if envelopes are used then here in the linear reconstruction, they are correct only for the linear imaging
        # for the nonlinear refinement it will be important not to modify the images or their fft for the proper comparison
        # with the forward calculation
        for ii in range(0, NumSteps):
            if (self.envelopeswitch):
                tmp += self.fn[:, :, ii] * self.sfftdata[:, :, ii]* self.invtot_env[:, :, ii]
            else:
                tmp += self.fn[:, :, ii] * self.sfftdata[:, :, ii]
        self.fftpsiscatt=tmp
        # take care of dc term, the average of the real part should be 1, for normalized input images
        # Use Parseval's theorem, sum of squares should be 1 for image.
        # Envelopes take out contrast, the corresponding term ends up in the dc part in the images, not so in the wavefunction.
        # The DC part in the images therefore has a meaning, but the total number of electrons is supposed to be constant
        # So we can use Parseval's theorem to normalize.
        # be careful: the fft sum of squares is dim[0]*dim[1] larger, because of fft normalization
        # so the sum of squares of the fft has to be dim[0]*dim[1]
        # tmp[0,0]=np.sqrt(totalsumsq-sumsq)
        #/self.dim[0]/self.dim[1]
        #
        # the correct way
        sumsq=np.sum(tmp*np.conj((tmp))) # fft sumsq
        totalsumsq=(self.dim[0]*self.dim[1])**2
        #
        # print("Sum of squares of the scattered wavefunction in Fourier space: ", sumsq)
        # print("Target sum of squares of the wavefunction in Fourier space: ", totalsumsq)
        print("Fraction of scattered electrons (%): {0:5.3f}".format(np.abs(sumsq/totalsumsq)*100))
        # print("DC term in Fourier space: ", np.sqrt(totalsumsq-sumsq))
        if scale:
            const=np.abs(np.sqrt(totalsumsq-sumsq)/(self.dim[0]*self.dim[1]))
        else:
            # leave constant, for refinement purposes
            const=0.
        # print("Constant in real space: ", const)
        tmp=np.fft.ifft2(tmp)
        self.psi = const + tmp        
        # print("Sum of squares of the wavefunction in Real space: ", np.sum(self.psi*np.conj((self.psi))))
        
        if self.saveintermediaterec:
            outfname=self.outpath+self.outfilenameprefix+"_"+str(self.linit).zfill(3)+".wav"
            print("Saving wave function to ", outfname)
            fh = open(outfname, "bw")
            (self.psi).astype(np.csingle).tofile(fh)
        return
    
    def Simulate(self):
       # Simulate images from a wave funnction
       # the routine takes a wavefunction, pre-caalcuated transfer functions tn and
       # precalulated envelope functions tot_env
       # (these are calculated using the current set of foci in self.foci)
       #
        print("+ --- Simulating images: ")
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        tmp=np.fft.fft2(self.psi) # the exit plane wave function
        del self.fftsimdata
        self.fftsimdata=np.zeros([self.dim[0], self.dim[1], NumSteps],dtype=np.csingle)
        # calclulate fft of propagated psi in fourier space 
        for ii in range(0, NumSteps):
            print(".", end="")
            if (self.envelopeswitch):
                self.fftsimdata[:, :, ii]=tmp*self.tn[:, :, ii] * self.tot_env[:, :, ii]
            else:
            # multiply fft of wavefunction with transfer functions
                self.fftsimdata[:, :, ii]=tmp*self.tn[:, :, ii] 
            # now the image intensity
            # psi(0)*psi(g)+conj(psi(0)*psi(-g))
            # or the full convolution: psi*conj(psi)
            tmpout=np.fft.ifft2(self.fftsimdata[:, :, ii])
            self.fftsimdata[:, :, ii]=np.fft.fft2(tmpout*np.conj(tmpout))
            #self.fftsimdata[0, 0, ii]=1.
        print()
        if debug:
            # check if the number of images matches the input range, then it means we are saving the debig information for the recostruction loop, otherwise it might be the refinement loop
            rangelabels=False
            if (NumSteps == np.size(self.imrange, 0)):
                rangelabels = True
            print("DEBUG: Saving intermediate simulated images", end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                if rangelabels:
                    outfname=self.outpath+self.outfilenameprefix+"_simim_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                else:
                   outfname=self.outpath+self.outfilenameprefix+"_simim_float32_"+str(ii).zfill(3)+".raw" 
                fh = open(outfname, "bw")
                tmpout=np.fft.ifft2(self.fftsimdata[:, :, ii])
                (tmpout.real).astype(np.single).tofile(fh)
            print()
        return


    def LinearIteration(self):
        for self.linit in range(1, self.NLinear+1):
            print("---------------------------------------")
            print("+ --- Linear Iteration {0:3d} of {1:3d}".format(self.linit,self.NLinear))
            print("---------------------------------------")
            #print("L --- Image alignment")
            myfsr.AlignImages()
            #print("L --- Backprojection")
            myfsr.BackProjection()
            #print("L --- Projection")
            myfsr.Simulate()
        return

    def ContinueIteration(self):
        # started implementing 23.05.2020
        # continue a previous iteration
        # - load previous dxy
        # - apply to loaded images
        # - load previous wavefunction
        # - run simulate once
        # - start iteration: Alignment -> Backproject -> Simulate (i.e. self.LinearIteration)
        return

    def StepAnalysis(self):
        # propagate the wavefunction, simulate a focal series
        # BEWARE!
        # This routine overwrites many arrays required for reconstruction. It is a final step!
        print("---------------------------------------")
        print("+ --- Focus step analysis: ")
        print("---------------------------------------")
        origfoci=np.copy(self.foci)
        #print("DEBUG: Input focus values in linear scheme in input format for {0:4d} images:".format(self.NImages))
        #print("[", end="")
        #for ii in range(0,self.NImages):
        #    print("{0:6.2f}".format(origfoci[ii]), end="")
        #    if (ii < self.NImages-1):
        #        print(",",end="")
        #print("]")
        step=0.25/(self.lamda*self.reclimit*self.reclimit)
        # create a new focus array
        # calculate focus range
        # Zmin ... Zmax
        # range=Zmin-Zmax
        # centre=0.5*(Zmin+Zmax)
        # newrange=2*range
        # newZmin=Zmin-range/2
        # newZmax=Zmax+range/2
        # step: step if newZmax > nwZmin, else -step 
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        piquarterstep=1./(4*self.lamda*self.reclimit*self.reclimit)
        focrange=np.abs(self.foci[-1]-self.foci[0])
        Nfocsteps=np.int32(2*np.abs(focrange/piquarterstep))
        tmp=np.arange((self.foci[-1]+self.foci[0])/2-self.steprefrangefact/2*focrange,(self.foci[-1]+self.foci[0])/2+self.steprefrangefact/2*focrange,piquarterstep*self.stepreffocfact)
        if (tmp[0] > tmp[-1]):
            # decreasing values, need to reverse the array
            self.foci=np.copy(tmp[::-1])
        else:
            self.foci=np.copy(tmp)
        Numfocsteps=np.size(self.foci, 0)        
        print("Simulating", Numfocsteps, " images")
        #print("DEBUG: Simulation focus values in linear scheme in input format for {0:4d} images:".format(self.NImages))
        #print("[", end="")
        #for ii in range(0,Numfocsteps):
        #    print("{0:6.2f}".format(self.foci[ii]), end="")
        #    if (ii < Numfocsteps):
        #        print(",",end="")
        #print("]")
        self.PrepareTransferFctns()
        self.PrepareEnvelopeFctns()
        self.PrepareRefinementFilter()
        self.Simulate()
        # now cross-correlate
        # one by one
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        print("--------------------------------------------------------------")
        print(" image    sttdev   mindev    minrms  index   initial  refined")
        print("--------------------------------------------------------------")
        corrvec=np.zeros(Numfocsteps,dtype=np.single)
        rmsvec=np.zeros(Numfocsteps,dtype=np.single)
        reffoc=np.zeros(NumSteps)
        for ii in range(0,NumSteps):
            im=np.fft.ifft2(self.sfftdata[:, :, ii]*self.refinementfilt)
            im=im[startx:startx+cropx,starty:starty+cropy]
            for jj in range(0,Numfocsteps):
                # correlate images and take maximum, alignfilt may be too strict! should be rec filter
                simim=np.fft.ifft2(self.fftsimdata[:, :, jj]*self.refinementfilt)
                simim=simim[startx:startx+cropx,starty:starty+cropy]
                # do a proper cross-correlation on normalised data
                # as a criterion you can use the cross correlation maximum
                # or the mean absolute difference between the normalised images
                #cim,maxpos,maxval=cross_corr(im.real, simim.real)
                corrvec[jj]=np.average(np.abs(im.real-simim.real))
                rmsvec[jj]=np.std(np.abs(im.real-simim.real))
            # print("corrvec=",corrvec)    
            ind=np.argmin(corrvec)
            minval=0
            minval=corrvec[ind]
            reffoc[ii]=self.foci[ind]
            print("{0:5d}{1:10.3f}{2:10.3f}{3:10.3f}{4:6d}{5:10.3f}{6:10.3f}".format(self.imrange[ii],np.std(im.real),minval,rmsvec[ind],ind,origfoci[ii],reffoc[ii]))
        print("-----------------------------------------------------")
        slope, intercept, r_value, p_value, std_err=scipy.stats.linregress(self.imrange,reffoc)
        print("+ -- Linear regression")
        print("Focus step (nm)        : {0:8.3f}".format(slope))
        print("First focus (nm)       : {0:8.3f}".format(intercept))
        print("Standard deviation (nm): {0:8.3f}".format(std_err))
        print("Correlation coefficient: {0:8.3f}".format(r_value**2))
        print("")
        a=(np.arange(0,(self.NImages),1,dtype=np.single))*slope+intercept
        print("Focus values in linear scheme in input format for {0:4d} images:".format(self.NImages))
        print("[", end="")
        for ii in range(0,self.NImages):
            print("{0:6.2f}".format(a[ii]), end="")
            if (ii < self.NImages-1):
                print(",",end="")
        print("]")
        print()
        print("Refined focus values in input format:")
        a[self.imrange]=reffoc
        self.allfoci=a
        print("[", end="")
        for ii in range(0,self.NImages):
            print("{0:6.2f}".format(a[ii]), end="")
            if (ii < self.NImages-1):
                print(",",end="")
        print("]")
        
        print()
        # save alignment data
        outfname=self.outpath+self.outfilenameprefix+"_foci.csv"
        print("Saving refined focus data to ", outfname)
        np.savetxt(outfname, np.transpose((self.imrange,reffoc)), delimiter=',', fmt='%f' , header='Refined defocus values')
        return
        
    def AutoFocus(self):
        # propagate the wavefunction, simulate a focal series
        # BEWARE!
        # This routine overwrites many arrays required for reconstruction. It is a separate call!
        print("---------------------------------------")
        print("+ --- Finding Minimum Contrast Focus: ")
        print("---------------------------------------")
        origfoci=np.copy(self.foci)
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        piquarterstep=1./(4*self.lamda*self.reclimit*self.reclimit) # focal step that corresponds to a phase shift of Pi/4 at the reconstruction limit
        focrange=np.abs(self.foci[-1]-self.foci[0])
        Nfocsteps=np.int32(2*np.abs(focrange/piquarterstep)) # search the whole defous range
        tmp=np.arange(- self.autofocrange,self.autofocrange,piquarterstep)
        if (tmp[0] > tmp[-1]):
            # decreasing values, need to reverse the array
            self.foci=np.copy(tmp[::-1])
        else:
            self.foci=np.copy(tmp)
        Numfocsteps=np.size(self.foci, 0)        
        print("Propagating to ", Numfocsteps, " focus values.")
        print("Focus search start (nm): {0:8.3f}".format(self.foci[0]))
        print("Focus search end   (nm): {0:8.3f}".format(self.foci[-1]))
        print("Focus search step  (nm): {0:8.3f}".format(piquarterstep))
        self.PrepareTransferFctns()
        # self.PrepareEnvelopeFctns() # better without, to suppress noise
        self.envelopeswitch=False
        self.PrepareAutoFocusFilter()
        self.Simulate()
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        rmsvec=np.zeros(Numfocsteps,dtype=np.single)
        reffoc=np.zeros(NumSteps)
        for jj in range(0,Numfocsteps):
            simim=np.fft.ifft2(self.fftsimdata[:, :, jj]*self.autofocfilt)
            simim=simim[startx:startx+cropx,starty:starty+cropy]
            rmsvec[jj]=np.std(np.abs(simim.real))         
        print()
        ind=np.argmin(rmsvec)
        minval=0
        minval=rmsvec[ind]
        # refine the minimum value with a quadratic fit
        i0=ind-2
        i1=ind+2
        if (i0 < 0):
            i1+=i0
            i0=0
        if (i1 >= Numfocsteps):
            i1=Numfocsteps-1
            i0=Numfocsteps-5
        # parabola fit, experimental code    
        #Y=rmsvec[i0:i1]
        #X=np.arange(i0,i1)
        #p0=[minval,0,-0.2]
        #Para = leastsq(parabolaerror, p0, args=(X, Y))
        #a, b, c = Para[0]
        #print "a=",a," b=",b," c=",c
        #print "cost:" + str(Para[1])
        #print("y="+str(round(a,2))+"x*x+"+str(round(b,2))+"x+"+str(c))
        #
        # save csv file in order to find the focus from the simulated images
        outfname=self.outpath+self.outfilenameprefix+"_focus_rms.csv"
        print("Saving focus rms data to ", outfname)
        np.savetxt(outfname, np.transpose((self.foci,rmsvec)), delimiter=',', fmt='%f' , header='rmsvec')
        print("Minimum contrast focus: {:.3f} nm".format(self.foci[ind]))
        print("Focus values in input format:")
        print("[", end="")
        for ii in range(0,self.NImages):
            print("{0:6.2f}".format(self.allfoci[ii]-self.foci[ind]), end="")
            if (ii < self.NImages-1):
                print(",",end="")
        print("]")
        # save alignment data
        outfname=self.outpath+self.outfilenameprefix+"_autofocus.csv"
        print("Saving focus rms data to ", outfname)
        np.savetxt(outfname, np.transpose((self.foci,rmsvec)), delimiter=',', fmt='%f' , header='rmsvec')
        print("Correcting wave function.")
        # take transferfunction of that focus and correct the wave function
        tmp=np.fft.fft2(self.psi)*self.tn[:, :, ind] # the corrected exit plane wave function in fourier space 
        self.psi = np.fft.ifft2(tmp) # the corrected exit plane wave function in real space
        return
        
    
    def Output(self, psionly=False):
        # save wavefunction
        outfname=self.outpath+self.outfilenameprefix+".wav"
        print("Saving wave function to ", outfname)
        fh = open(outfname, "bw")
        (self.psi).astype(np.csingle).tofile(fh)
        if not (self.outfilt is None):
            outfname=self.outpath+self.outfilenameprefix+"_filt.wav"
            print("Saving filtered wave function to ", outfname)
            fh = open(outfname, "bw")
            tmp=np.fft.fft2(self.psi)*self.outfilt # the corrected exit plane wave function in fourier space 
            tmp = np.mean(self.psi)+np.fft.ifft2(tmp) # the corrected exit plane wave function in real space
            (tmp).astype(np.csingle).tofile(fh)
        if psionly:
            return
        # save alignment data
        outfname=self.outpath+self.outfilenameprefix+"_dxy.csv"
        print("Saving displacement data to ", outfname)
        np.savetxt(outfname, np.transpose((self.imrange,self.displacements[0,:],self.displacements[1,:])), delimiter=',', fmt='%f' , header='Image displacements')
        # save aligned images
        if self.savealignedimages:
            NumSteps = np.size(self.foci, 0)  # the number of defocus steps
            print("Output: Saving aligned images ", end="")
            for ii in range(0, NumSteps):
                print(".", end="")
                outfname=self.outpath+self.outfilenameprefix+"_ali_float32_"+str(self.imrange[ii]).zfill(3)+".raw"
                fh = open(outfname, "bw")
                tmp=(np.fft.ifft2(self.sfftdata[:, :, ii])).real
                tmp.astype(np.single).tofile(fh)
            print()    
        return


# define sub-class for non-linear reconstruction, inheriting linear FSR
class NonLinFSR(FSR):

    def __init__(self, method="LW"):
        FSR.__init__(self)
        self.method=method
        self.linpsi=None
        self.fftlinpsiscatt=None
        self.sfftorigdata=None
        self.NNonLinear=10
        self.regpar=0.05

    def PrepareNonLinIteration(self):
        # secure data from overwriting
        self.linpsi=np.copy(self.psi)
        self.totpsi=np.copy(self.psi)
        self.fftlinpsiscatt=np.copy(self.fftpsiscatt)
        self.sfftorigdata=np.copy(self.sfftdata)
        return
        
    def RecDifferenceImages(self):
        self.PrepareNonLinIteration()
        # subtract images and simulated images, then apply phase factor to fftdata in order to update sfftdata
        NumSteps = np.size(self.foci, 0)  # the number of defocus steps
        # take central quarter of images 
        cropx = self.framewidth[0]
        cropy = self.framewidth[1]
        startx = (self.dim[0]-cropx)//2 + self.frameoffset[0]
        starty = (self.dim[1]-cropy)//2 + self.frameoffset[1]
        for kk in range(1, self.NNonLinear+1):
            print("---------------------------------------")
            print("+ --- Non-linear Iteration {0:3d} of {1:3d}".format(kk,self.NNonLinear))
            print("---------------------------------------")
            print("+ --- Calculating difference images: ")
            totrms=0.
            print("--------------------------------------")
            print(" image    rms                         ")
            print("--------------------------------------")
            for ii in range(0, NumSteps):
                im=np.fft.ifft2(self.sfftorigdata[:, :, ii]) # the original data
                simim=np.fft.ifft2(self.fftsimdata[:, :, ii]) # the current simulation
                # the dangereous line, overwriting sfftdata!
                diffim=im-simim
                self.sfftdata[:, :, ii]=np.fft.fft2(self.regpar*diffim) # the correction of the data, to be reconstructed and added 
                rms=np.std(diffim[startx:startx+cropx,starty:starty+cropy])
                totrms += rms
                print("{0:5d}{1:8.3f}".format(self.imrange[ii],rms))
            print("---------------------------------------")
            print("rms per image = {0:8.3f}".format(totrms/NumSteps))
            #print("L --- Backprojection")
            self.BackProjection(scale=False)
            # the correction for the wavefunction
            # the correction term
            # the power needs to be corrected, otherwise the comparison between images and simulated images will diverge
            # normalize to the total number of electrons, i.e. the mean value of psi must stay 1
            # self.totpsi is normalized, that means the total power of the correction term should be zero
            # the power of the transmitted term in the correction is the negative of the power of the correction
            # the power in the corrected wavefunction
            self.totpsi += (self.psi-np.mean(self.psi))
            # now simulate and align from the new total 
            self.psi = np.copy(self.totpsi)
            self.Simulate() # simulate from the new total wavefunction
            self.sfftdata=self.sfftorigdata # get back to original data for alignment
            # self.AlignImages()
            # self.fftpsiscatt=tmp
            if debug:
                print("DEBUG: Saving difference image fourier transforms", end="")
                for ii in range(0, NumSteps):
                    print(".", end="")
                    outfname=self.outpath+self.outfilenameprefix+"_fftdiffim_"+str(self.imrange[ii]).zfill(3)+".raw"
                    fh = open(outfname, "bw")
                    (self.sfftdata[:, :, ii]).astype(np.csingle).tofile(fh)
                print("DEBUG: Saving difference image fourier transforms", end="")
                
                print()
            if self.saveintermediaterec:
                outfname=self.outpath+self.outfilenameprefix+"_"+str(kk).zfill(3)+"nonlin.wav"
                print("Saving wave function to ", outfname)
                fh = open(outfname, "bw")
                (self.totpsi).astype(np.csingle).tofile(fh)
                outfname=self.outpath+self.outfilenameprefix+"_"+str(kk).zfill(3)+"nonlincorr.wav"
                print("Saving wave function to ", outfname)
                fh = open(outfname, "bw")
                (self.psi).astype(np.csingle).tofile(fh)
        return        

    
'''

Main Program

'''
# parse command line arguments          
parser = argparse.ArgumentParser(description='A focal series reconstruction code.',epilog="(c) L. Houben, I. Biran. Weizmann Institute of Science.  Version 1.0, Aug 2021.")
parser.add_argument('--prm', help='Full path to the input configuration file.', required=True)
parser.add_argument('--debug', help='Enter debugging mode.',action="store_true")
parser.add_argument('--showalignment', help='Enter debugging mode for alignment.',action="store_true")
parser.add_argument('--stepanalysis', help='Focus step analysis for a previous output.',action="store_true")
parser.add_argument('--autofocus', help='Determine the minimum contrast focus of the wave function for a previous output.',action="store_true")
parser.add_argument('--useprevious', help='Continue from the output of a previous iteration.',action="store_true")
parser.add_argument('--prealignonly', help='Exit program after prealignment.',action="store_true")
parser.add_argument('--boxcar', help='Save boxcar sum images after pre-alignment or loading previous alignment data.')
parser.add_argument('--showarea', help='Export the first image with the reference area marked.',action="store_true")
parser.add_argument('--liniter', help='Number of linear iterations.')
parser.add_argument('--nonlinear', help='Nonlinear refinement (EXPERIMENTAL).',action="store_true")
args = parser.parse_args()
debug=args.debug
showarea=args.showarea
debugalign=args.showalignment
stepana=args.stepanalysis
autofocus=args.autofocus
continueprevious=args.useprevious
#print(args)
#if args.prealignonly:
# print("prealign switch")
#    sys.exit("Leaving program after prelignment.")
nonlinear=args.nonlinear
if nonlinear:
    myfsr=NonLinFSR()
else:    
    myfsr=FSR()
print("@ --- Initializing")
myfsr.ReadParameterFile(args.prm)
if args.liniter:
    # overwrite number of linear iterations set in the parameter file
    val=int(args.liniter)
    if (val < 1) or (val > 20):
        sys.exit("Number of linear iterations has to be between 1 and 20.Check input.")
    myfsr.SetNLinearIterations(val)
    #
if showarea:
    myfsr.PreviewRefArea()
    sys.exit("Program halted")
if autofocus:
    print("@ --- Auto focus")
    myfsr.PrepareSamplingGrid()
    myfsr.LoadPreviousWav()
    myfsr.AutoFocus()
    myfsr.PrepareOutputFilter()
    myfsr.Output(psionly=True)
    del myfsr
    sys.exit("")
print("@ --- Preparation")
# preparation steps independent of image sequence
myfsr.PrepareSamplingGrid()
myfsr.PrepareAlignmentFilter()
# preparation steps dependent of image sequence
myfsr.PrepareTransferFctns()
myfsr.PrepareFilterFctns()
myfsr.PrepareEnvelopeFctns()
print("@ --- Data loading")
myfsr.LoadImages()
if stepana:
    print("@ --- Focus Step Analysis")
    # myfsr.LoadPreviousFoci()
    myfsr.LoadPreviousDisplacements()
    myfsr.LoadPreviousWav()
    myfsr.StepAnalysis()
    del myfsr
    sys.exit("")
#   
if continueprevious:
    myfsr.LoadPreviousFoci()
    myfsr.LoadPreviousDisplacements()
else:
    print("@ --- Pre-Alignment")
    myfsr.PreAlignImages()
    if args.prealignonly:
        myfsr.Output()
        del myfsr
        sys.exit("Leaving program after prelignment.")
# boxcar average for a separate iteration? should be done here after the pre-alignment
if args.boxcar:
    print("@ --- Saving boxcar sum images")
    myfsr.SaveBoxcarSumImages(int(args.boxcar))
    sys.exit("Leaving program.")
print("@ --- Reconstruction Loop")
if continueprevious:
    myfsr.LoadPreviousWav()
    myfsr.Simulate() 
# reconstruct
myfsr.LinearIteration()
if nonlinear:
    print("@ --- Nonlinear Loop")
    myfsr.RecDifferenceImages()
print("@ --- Output")
myfsr.PrepareOutputFilter()
myfsr.Output()
#
del myfsr
