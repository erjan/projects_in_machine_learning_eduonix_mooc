import sys,keras,cv2,numpy as np,matplotlib,skimage
from keras.models import Sequential
from keras.layers import  Conv2D,Input
from keras.optimizers import  SGD,Adam
from skimage.measure import  compare_ssim as ssim
from matplotlib import pyplot as plt
import math, os
#python magic function

#define a function to peak signal to noise ratio PNSR
def psnr(target,ref):
    #assume RGB img
    target_data = target.astype(float)
    ref_data = ref.astype(float)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    
    rmse = math.sqrt(np.mean(diff**2.))
    return  20 * math.log10(255./rmse)
#define a function for mean squared error mse
def mse(target,ref):
    #MSE is the sum of the squared dif btw the 2 imgs
    err = np.sum((target.astype('float')-ref.astype('float'))**2)
    err /= float(target.shape[0]*target.shape[1])
    return err

#define a functon that combnes all3 img quatli metrics
def compare_images(target,ref):
    scores=[]
    scores.append(psnr(target,ref))
    scores.append(mse(target,ref))
    scores.append(ssim(target,ref,multichannel=True))
    return scores

#prepare degraded imgs by ntroducng quality distortions via resizing
def prepare_images(path,factor):
    #loop thru the files in a dir
    for file in os.listdir(path):
        #opoen the file
        img = cv2.imread(path +'/'+file)
        #find old new img dimensions
        h,w,c=img.shape
        new_height=h/factor
        new_width = w/factor
        #resize the image down
        img = cv2.resize(img,(new_width,new_height),interpolation =cv2.INTER_LINEAR)
        #resize the img back up
        img = cv2.resize(img,(w,h), interpolation = cv2.INTER_LINEAR)
        #save the img
        print('saving {}'.format(file))
        cv2.imwrite('images/{}'.format(file),img)

prepare_images('source/',2)
#results - output
'''
    saving baboon.bmp
    saving baby_GT.bmp
    saving barbara.bmp
    saving bird_GT.bmp
    saving butterfly_GT.bmp
    saving coastguard.bmp
    saving comic.bmp
    saving face.bmp
    saving flowers.bmp
    saving foreman.bmp
    saving head_GT.bmp
    saving lenna.bmp
    saving monarch.bmp
    saving pepper.bmp
    saving ppt3.bmp
    saving woman_GT.bmp
    saving zebra.bmp
    
'''

#testing the generated images using the image quality metrics
for file in os.listdir('images/'):
    #open  target  and ref imgs
    target = cv2.imread('images/{}'.format(file))
    ref = cv2.imread('source/{}'.format(file))
    #calculate the score
    scores = compare_images(target,ref)
    #print all 3 scores
    print('{}\nPSNR:{}\nMSE:{}\nSSIM:{}\n'.format(file,scores[0],scores[1],scores[2]))
'''
    baboon.bmp
    PSNR:22.1570840834
    MSE:1187.11613333
    SSIM:0.6292775879
    
    baby_GT.bmp
    PSNR:34.3718064097
    MSE:71.2887458801
    SSIM:0.935698787272
    
    barbara.bmp
    PSNR:25.9066298376
    MSE:500.655085359
    SSIM:0.809863264641
    
    bird_GT.bmp
    PSNR:32.8966447287
    MSE:100.123758198
    SSIM:0.953364486603
    
    butterfly_GT.bmp
    PSNR:24.7820765603
    MSE:648.625411987
    SSIM:0.879134476384
    
    coastguard.bmp
    PSNR:27.1616006639
    MSE:375.008877841
    SSIM:0.756950063355
    
    comic.bmp
    PSNR:23.7998615022
    MSE:813.233883657
    SSIM:0.83473354164
    
    face.bmp
    PSNR:30.9922065029
    MSE:155.231897185
    SSIM:0.800843949229
    
    flowers.bmp
    PSNR:27.4545048054
    MSE:350.550939227
    SSIM:0.869728628697
    
    foreman.bmp
    PSNR:30.1445653266
    MSE:188.688348327
    SSIM:0.933268417389
    
    head_GT.bmp
    PSNR:31.0205028482
    MSE:154.22377551
    SSIM:0.801112133073
    
    lenna.bmp
    PSNR:31.4734929787
    MSE:138.948005676
    SSIM:0.846098920052
    
    monarch.bmp
    PSNR:30.1962423653
    MSE:186.456436157
    SSIM:0.943957429343
    
    pepper.bmp
    PSNR:29.8894716169
    MSE:200.103393555
    SSIM:0.835793756846
    
    ppt3.bmp
    PSNR:24.8492616895
    MSE:638.668426391
    SSIM:0.928402394232
    
    woman_GT.bmp
    PSNR:29.3262362808
    MSE:227.812729498
    SSIM:0.933539728047
    
    zebra.bmp
    PSNR:27.9098406393
    MSE:315.658545953
    SSIM:0.891165620933
 '''     
