import tkinter
from tkinter import Button
import math,numpy
import scipy.misc
from scipy.misc.pilutil import Image
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import tkinter.filedialog as tkFileDialog
import cv2
import sys
from scipy import ndimage
import numpy as np
from scipy import signal
from skimage.measure import compare_ssim as ssim
#defining a tkinter window for root,used for gui part
root =tkinter.Tk()

#setting the geometry of the gui window 
root.geometry('950x708+100+100');
#setting title of gui window
root.title("Image Restoration Assignment")

#creating a photoimage object for storing the image in label
image = PhotoImage(file='')
#initializing the image in the label
label=Label(image=image)
global text1
text1=""
#displaying text "Original image" in label2
label2=Label(root,text="Orignal image")
#placing image in gui with coordinates x=180 and y=400
label2.place(x=180,y=400)
#displaying text "Modified image" in label2
# label4=Label(root,text=text1)
# #placing image in gui with coordinates x=180 and y=400
# label4.place(x=600,y=400)

label5=Label(root,text="PSNR")
#placing image in gui with coordinates x=180 and y=400
label5.place(x=720,y=450)

label6=Label(root,text="SSIM")
#placing image in gui with coordinates x=180 and y=400
label6.place(x=720,y=480)


def orignal():
	image=cv2.imread(path)
	
	
	(B,G,R)=cv2.split(image);
	
	# kernel1 = cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png')
	k1=int(w4.get())
	
	if k1 == 1:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png');
	elif k1 == 2:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel2G.png');
	elif k1 == 3:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel3G.png');
	elif k1 == 4:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel4G.png');



	# kernel1=cv2.imread(loc);
	
	(B2,G2,R2)=cv2.split(kernel1);
	
	kb = np.zeros_like(B);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	



	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	

	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	

	kr=np.divide(fr, Hr)
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	
	M = kr.shape[0]
	N = kr.shape[1]
	# H is defined and
	# values in H are initialized to 1
	H1 = numpy.ones((M,N))
	c1 = M/2	#center1
	c2 = N/2	#center2
	rcut =int(w1.get()) #cut off radius
	
	t1 = 1 # the order of BLPF
	t2 = 2*t1
	# defining the convolution function for BLPF
	for i in range(1,M):
		for j in range(1,N):
			r1 = (i-c1)**2 + (j-c2)**2
			# euclidean distance from origin is computed
			r = math.sqrt(r1)
			# eliminating high frequency using cut-off radius 
			if r > rcut:
				H1[i,j] = 1/(1 + (r/rcut)**t1)


	# plt.imshow(H1, cmap='gray')
	# plt.show()	

	kr = np.fft.fftshift(kr)
	kb = np.fft.fftshift(kb)
	kg = np.fft.fftshift(kg)

	# H1 =scipy.misc.toimage(H1)
	H1 = H1 / np.max(H1)
	# print(np.max(H1))

	# performing the convolution
	kr = kr * H1
	kb = kb * H1
	kg = kg * H1
	
	# # computing the magnitude of the inverse FFT
	# # e = abs(fftim.ifft2(con))

	R=np.abs(np.fft.ifft2(kr))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))
	
	
	mer=cv2.merge([R, G, B])
	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img4
	label7=Label(root,text="")
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=450)

	label8=Label(root,text="")
	#placing image in gui with coordinates x=180 and y=400
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing image in gui with coordinates x=180 and y=400
	label4.place(x=600,y=400)
	

def mkimbl():
	# imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/GroundTruth1_1_1.jpg')
	
	imgo = cv2.imread(path)
		
		
	
	
	
	# imgo= imgo[0:380, 0:380]
	(B1,G1,R1)=cv2.split(imgo);
	
	kernel1 = cv2.imread(path2)
	# kernel1 = cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png',0)
	
	# kn = np.zeros_like(B);
	# #kn = np.zeros_like(imager)
	# kn[0:kernel1.shape[0], 0:kernel1.shape[1]] = kernel1
	# kn = kn / (np.sum(kn))

	(B2,G2,R2)=cv2.split(kernel1);
	# b1 = kernel1.copy()
	kb = np.zeros_like(B1);
	# kb= np.pad(B2, [(0, B1.shape[0] - B2.shape[0]), (0, B1.shape[1] - B2.shape[1])], 'constant')
	# # print(kn);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G1);
	# kr= np.pad(R2, [(0, R1.shape[0] - R2.shape[0]), (0, R1.shape[1] - R2.shape[1])], 'constant')
	# # print(kn);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R1);
	# kg= np.pad(G2, [(0, G1.shape[0] - G2.shape[0]), (0, G1.shape[1] - G2.shape[1])], 'constant')
	# # print(kn);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	



	# kb = np.zeros_like(B);
	# kn= np.pad(kernel1, [(0, B.shape[0] - kernel1.shape[0]), (0, B.shape[1] - kernel1.shape[1])], 'constant')
	# print(kn);
	# kn= np.pad(kernel1, [(0, B.shape[0] - kernel1.shape[0]), (0, B.shape[1] - kernel1.shape[1])], 'constant')
	# print(kn);
	# kn[0:kernel1.shape[0], 0:kernel1.shape[1]] = kernel1
	# kn = kn / (np.sum(kn))
	



	fr = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)
	
	# fr = np.fft.fftshift(fr)
	# fb = np.fft.fftshift(fb)
	# fg = np.fft.fftshift(fg)

	# f1 = np.fft.fft2(im2)
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	

	# Hb = np.fft.fftshift(Hb)
	# Hr = np.fft.fftshift(Hr)
	# Hg = np.fft.fftshift(Hg)

	# kr=np.divide(fr, Hr)
	
	# kb=np.divide(fb, Hb)
	# kg=np.divide(fg, Hg)
	# R=np.abs(np.fft.ifft2(kr))
	# B=np.abs(np.fft.ifft2(kb))
	# G=np.abs(np.fft.ifft2(kg))
	
	# mer=cv2.merge([R, G, B])
	










	f=np.multiply(fr,Hr)
	fb=np.multiply(fb,Hb)
	fg=np.multiply(fg,Hg)
	
	# fr = np.fft.fftshift(f)
	# fb = np.fft.fftshift(fb)
	# fg = np.fft.fftshift(fg)

	


	R=np.abs(np.fft.ifft2(f))
	B=np.abs(np.fft.ifft2(fb))
	G=np.abs(np.fft.ifft2(fg))
	
	mer=cv2.merge([R, G, B])
	# cv2.imwrite('messigray.png',mer)
	# mer.save('/home/shubham/Desktop/ip_assign2/my.png')
	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img3
	

def convol(image, kernel,n):
    
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))    
    # convolution output is defined of image size
    output = np.zeros_like(image)
    # Add zero padding to input image
    image_padded = np.zeros((image.shape[0] + 2*n, image.shape[1] + 2*n)) 

    image_padded[n:-n, n:-n] = image
   #for loop is designed to iterate for every pixel in the image 
    for x in range(image.shape[1]):    
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+(2*n+1),x:x+(2*n+1)]).sum()        
    return output		#ouput is returned to calling function 

def psnr(image1, image2):
    mse1 = numpy.mean( (image1 - image2) ** 2 )
    if mse1 == 0:
    	return 100;
    maxp = 255.0#max pixel value
    return 20 * math.log10(maxp / math.sqrt(mse1))
def tinver():
	image=cv2.imread(path)
	
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')

	(B,G,R)=cv2.split(image);
	# kernel1 = cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png',0)
	kernel1 = cv2.imread(path2)
	
	(B2,G2,R2)=cv2.split(kernel1);
	

	kb = np.zeros_like(B);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	


	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	

	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	

	kr=np.divide(fr, Hr)
	
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	









	M = kr.shape[0]
	N = kr.shape[1]
	# H is defined and
	# values in H are initialized to 1
	H1 = numpy.ones((M,N))
	c1 = M/2	#center1
	c2 = N/2	#center2
	rcut =int(w1.get()) #cut off radius
	
	t1 = 1 # the order of BLPF
	t2 = 2*t1
	# defining the convolution function for BLPF
	for i in range(1,M):
		for j in range(1,N):
			r1 = (i-c1)**2 + (j-c2)**2
			# euclidean distance from origin is computed
			r = math.sqrt(r1)
			# eliminating high frequency using cut-off radius 
			if r > rcut:
				H1[i,j] = 1/(1 + (r/rcut)**t1)


	# plt.imshow(H1, cmap='gray')
	# plt.show()	

	kr = np.fft.fftshift(kr)
	kb = np.fft.fftshift(kb)
	kg = np.fft.fftshift(kg)

	# H1 =scipy.misc.toimage(H1)
	H1 = H1 / np.max(H1)
	print(np.max(H1))

	# performing the convolution
	kr = kr * H1
	kb = kb * H1
	kg = kg * H1
	

	R=np.abs(np.fft.ifft2(kr))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))
	

	# o = np.ones_like(s1)
	# # s = (s-np.min(s) * o) * 255/(np.max(s) - np.min(s))
	# # ones = np.ones_like(s)
	# # s = (s - np.min(s) * ones) / (np.max(s) - np.min(s))
	# # scipy.misc.toimage(s);

	# print(np.max(s1))
	# print(np.min(s1))
	mer=cv2.merge([R,G,B])
	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img4
	d=psnr(mer,imgo)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=450)

	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
        # img2 = numpy.asarray(Image.open(argv[2]))
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	d=ssim(i1,i2)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing image in gui with coordinates x=180 and y=400
	label4.place(x=600,y=400)
	

def inver():
	# image = cv2.imread('/home/shubham/Desktop/ip_assign2/Blurry1_2.jpg')
	image=cv2.imread(path)
	
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	
	(B,G,R)=cv2.split(image);
	
	# kernel1 = cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png')
	kernel1=cv2.imread(path2);
	
	(B2,G2,R2)=cv2.split(kernel1);
	
	kb = np.zeros_like(B);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	



	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	

	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	

	kr=np.divide(fr, Hr)
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	
	R=np.abs(np.fft.ifft2(kr))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))
	
	mer=cv2.merge([R, G, B])
	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img4
	d=psnr(imgo,mer)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=450)

	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
        # img2 = numpy.asarray(Image.open(argv[2]))
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing image in gui with coordinates x=180 and y=400
	label4.place(x=600,y=400)
	

def weiner():
	# imag1=cv2.imread('/home/shubham/Desktop/ip_assign2/Blurry1_2.jpg')
	imag1=cv2.imread(path)
	# imag1=cv2.resize(imag1,(380,380), interpolation = cv2.INTER_AREA);
	
	i1=imag1
	(B1,G1,R1) = cv2.split(imag1);	
	# kernel1 = cv2.imread(path2)
	kernel1 = cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png')
	
	
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	
	
	(B2,G2,R2)=cv2.split(kernel1);
	# b1 = kernel1.copy()
	
	kb = np.zeros_like(B1);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G1);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R1);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	



	#getting value from slider 2 for K
	K =int(w3.get())
	# K=.1
	K1 = np.ones_like(kb)
	K=K*K1;
	f = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)

	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	



	# H2b=np.multiply(Hb,np.conj(Hb))
	# Denomb=np.multiply(Hb,H2b)+K
	# H3b=np.divide(H2b, Denomb)
	

	# H2r=np.multiply(Hr,np.conj(Hr))
	# Denomr=np.multiply(Hr,H2r)+K
	# H3r=np.divide(H2r, Denomr)

	# H2g=np.multiply(Hg,np.conj(Hg))
	# Denomg=np.multiply(Hg,H2g)+K
	
	# H3g=np.divide(H2g, Denomg)
	H3b = np.conj(Hb) / (np.abs(Hb) ** 2 + K)
	H3r = np.conj(Hr) / (np.abs(Hr) ** 2 + K)
	H3g = np.conj(Hg) / (np.abs(Hg) ** 2 + K)
	
	k=np.multiply(H3b,f)
	kb=np.multiply(H3b,fb)
	kg=np.multiply(H3g,fg)
	
	R=np.abs(np.fft.ifft2(k))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))	
	mer=cv2.merge([R, G, B])
	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img4
	d=psnr(imgo,mer)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=450)

	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
        # img2 = numpy.asarray(Image.open(argv[2]))
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing image in gui with coordinates x=180 and y=400
	label4.place(x=600,y=400)
	# plt.subplot(121),plt.imshow(mer.astype(np.uint8))
	# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122),plt.imshow(imag1.astype(np.uint8))
	# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	# plt.show()


def clsf():
	# imag1=cv2.imread('/home/shubham/Desktop/ip_assign2/Blurry1_2.jpg')
	imag1=cv2.imread(path)
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	
	
	(B1,G1,R1) = cv2.split(imag1);	
	kernel1 = cv2.imread(path2)
	(B2,G2,R2)=cv2.split(kernel1);
	
	kb = np.zeros_like(B1);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	kb = kb / (np.sum(kb))

	kg = np.zeros_like(G1);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	kg = kg / (np.sum(kg))
	
	kr = np.zeros_like(R1);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	kr = kr / (np.sum(kr))
	


	#getting value from slider 2 for K
	y =int(w2.get())
	
	p1=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	print(p1)
	p = np.zeros_like(B1);
	print(p)
	# p[0:p1.shape[0], 0:p1.shape[1] ]= p1
	p= np.pad(p1, [(0, B1.shape[0] - p1.shape[0]), (0, B1.shape[1] - p1.shape[1])], 'constant')
	# p = p / (np.sum(p))
	print(p)
	fp = np.fft.fft2(p)
	
	f = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)
	# H = np.fft.fft2(kn)
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	



	H2b=np.multiply(Hb,np.conj(Hb))
	P=np.multiply(fp,np.conj(fp))

	d1=y*P
	print(d1)
	denb=H2b+d1
	H3b=np.divide(np.conj(Hb), denb)
	

	H2g=np.multiply(Hg,np.conj(Hg))
	
	deng=H2g+d1
	H3g=np.divide(np.conj(Hg), deng)
	

	H2r=np.multiply(Hr,np.conj(Hr))
	denr=H2r+d1
	H3r=np.divide(np.conj(Hr), denr)
	








	k=np.multiply(H3r,f)
	kb=np.multiply(H3b,fb)
	kg=np.multiply(H3g,fg)
	k=np.fft.fftshift(k);
	kb=np.fft.fftshift(kb);
	kg=np.fft.fftshift(kg);
	R=np.abs(np.fft.ifft2(k))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))	
	mer=cv2.merge([R, G, B])

	
	


	img3 = scipy.misc.toimage(mer)
	#image is resized to fit in label
	img4=img3.resize((380,380));
	#image is convered into photo image
	image2=ImageTk.PhotoImage(img4)
	#label3 in gui is configured to get new value
	label3.configure(image=image2)
	#labek3 displays image with image=image2
	label3.image=image2
	#label3 placed in gui 
	label3.place(x=500,y=20)
	#defining a global image variable
	global imagesave
	#current image is saved in global variable imagesave
	imagesave=img4
	d=psnr(imgo,mer)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label7.place(x=760,y=450)
	
	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
        # img2 = numpy.asarray(Image.open(argv[2]))
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=180 and y=400
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing image in gui with coordinates x=180 and y=400
	label4.place(x=600,y=400)
	



def choose():
	#path is defined global
	global path
	#opening file using tkinter and it return image path
	path1=tkFileDialog.askopenfilename(filetypes=[("Image File",'.png'),("Image File",'.jpg')])

	path=path1
	#image is opened 
	ima=Image.open(path1)
	#image is resized
	ima1=ima.resize((380,380));
	#image is configured to photo image
	image2=ImageTk.PhotoImage(ima1)
	label.configure(image=image2)
	label.image=image2
	label.place(x=50,y=20)
	

def choosek():
	#path is defined global
	global path2
	#opening file using tkinter and it return image path
	path2=tkFileDialog.askopenfilename(filetypes=[("Image File",'.png'),("Image File",'.jpg')])
	# global kernel1
	
	ima=Image.open(path2)
	#image is resized
	ima1=ima.resize((380,380));
	#image is configured to photo image
	image2=ImageTk.PhotoImage(ima1)
	label3.configure(image=image2)
	label3.image=image2
	label3.place(x=500,y=20)
	text1="Kernel"

	# #displaying text "Modified image" in label2
	# label4=Label(root,text="Kernel")
	# #placing image in gui with coordinates x=180 and y=400
	# label4.place(x=600,y=400)





def saveas():
	#using tkinter dialog box to get directory to save file
	root.filename=tkFileDialog.asksaveasfilename(initialdir = "/",title = "Save as",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
	#saving file imagesave in directory
	imagesave.save(root.filename)
	
#creating a slider w1 to take input from user
w1=Scale(root,from_=100,to=300,resolution=10,orient=HORIZONTAL)
#placing the silder in gui with co-ordinate x=350 and y=520
w1.place(x=350,	y=510)
#setting initial value to1 
w1.set(170)
#creating a slider w2 to take input from user
w2=Scale(root,from_=.1,to=2,resolution=.1,orient=HORIZONTAL)
#placing the silder in gui with co-ordinate x=350 and y=560
w2.place(x=350,	y=590)
#setting initial value to1 
w2.set(1)


w3=Scale(root,from_=.1,to=2,resolution=0.1,orient=HORIZONTAL)
w3.place(x=350,	y=550)
w3.set(1)

w4=Scale(root,from_=1 ,to= 4,resolution=1,orient=HORIZONTAL)
w4.place(x=350,	y=630)
w4.set(1)


label4=Label(root,text=text1)
#placing image in gui with coordinates x=180 and y=400
label4.place(x=600,y=400)

label=Label(image=image)

image = PhotoImage(file='')
label3=Label(image=image)

#creating a button for log tranformation and on clicking it will invoke function logt()
clsfb=Button(root, text="Constrained LSF", command= clsf) 
#placing the button logbutton in gui
clsfb.place(x=200,y= 600);


chooseb=Button(root, text='Open File', command=choose)
chooseb.place(x=100,y=420)

choosekb=Button(root, text='Open kernel', command=choosek)
choosekb.place(x=200,y=420)




ob=Button(root, text="Orignal", command= orignal)
ob.place(x=200,y= 640);



tinverb=Button(root, text="Truncated", command= tinver)
tinverb.place(x=200,y= 520);

weinerb =Button(root, text="Weiner", command= weiner)
weinerb.place(x=200, y=560);

inverb=Button(root, text="Inverse", command= inver)
inverb.place(x=200, y=480);

saveb=Button(root, text="Save", command= saveas)
saveb.place(x=600, y=490);



mkib=Button(root, text="Make blurr", command=mkimbl)
mkib.place(x=600, y=620);


#root is main window and it get closed with user clicks close button
root.mainloop()


