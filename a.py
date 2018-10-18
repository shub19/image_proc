

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



#defining function for orignal image restoration where kernel not known
def orignal():#Code written by me
	#reading the image using opencv
	image=cv2.imread(path)
	
	#splitting image in 3 scale R,G and B
	(B,G,R)=cv2.split(image);
	#taking value from user to choose Kernel
	k1=int(w4.get())
	#kernel selection
	if k1 == 1:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/K_1.png');
	elif k1 == 2:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel2G.png');
	elif k1 == 3:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel3G.png');
	elif k1 == 4:
		kernel1=cv2.imread('/home/shubham/Desktop/ip_assign2/Kernel4G.png');



	#splitting kernel in 3 scale R,G and B
	(B2,G2,R2)=cv2.split(kernel1);
	#creating a matrix with zero value same as B matrix
	kb = np.zeros_like(B);
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing 
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as B matrix
	kg = np.zeros_like(G);
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing 
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as B matrix
	kr = np.zeros_like(R);
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing 
	kr = kr / (np.sum(kr))
	


	#performing 2D fft on R,G and B scale of image
	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	

	#performing 2D fft on R,G and B scale of kernel
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	
	#diving the fft of image in respective scale to fft of kernel in respective scale
	kr=np.divide(fr, Hr)
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	#shape of matrix is known
	M = kr.shape[0]
	N = kr.shape[1]
	# H1 is defined for size M,N and value 1 is assigned
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
	#performing fft shift
	kr = np.fft.fftshift(kr)
	kb = np.fft.fftshift(kb)
	kg = np.fft.fftshift(kg)

	H1 = H1 / np.max(H1)
	# print(np.max(H1))

	# performing the convolution
	kr = kr * H1
	kb = kb * H1
	kg = kg * H1
	
	# computing the magnitude of the inverse FFT

	R=np.abs(np.fft.ifft2(kr))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))
	
	#merging the R,G and B scale
	mer=cv2.merge([R, G, B])
	label7=Label(root,text=" ")
	#placing label in gui with coordinates x=760 and y=450
	label7.place(x=760,y=450)

	label8=Label(root,text=" ")
	#placing image in gui with coordinates x=760 and y=480
	label8.place(x=760,y=480)



	#matrix converted to image
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
	

	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing text in gui with coordinates x=600 and y=400
	label4.place(x=600,y=400)
	

def mkimbl():#Code written by me
	#reading the image using opencv
	imgo = cv2.imread(path)
		
	#splitting image in 3 scale R,G and B
	(B1,G1,R1)=cv2.split(imgo);
	#reading the kernel using opencv
	kernel1 = cv2.imread(path2)
	
	#splitting kernel in 3 scale R,G and B
	(B2,G2,R2)=cv2.split(kernel1);
	
	#creating a matrix with zero value same as B1 matrix
	kb = np.zeros_like(B1);
	
	#initializing B2 from topleft side
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing it
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as G1 matrix
	kg = np.zeros_like(G1);
	#initializing G2  to kg from topleft side
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing it
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as R1 matrix
	kr = np.zeros_like(R1);
	#initializing B2  to kr from topleft side
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing it
	kr = kr / (np.sum(kr))

	#performing 2 D fft of image
	fr = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)
	
	# fr = np.fft.fftshift(fr)
	# fb = np.fft.fftshift(fb)
	# fg = np.fft.fftshift(fg)

	#performing 2 D fft of kernel image
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	


	#matrix multiplication element wise
	f=np.multiply(fr,Hr)
	fb=np.multiply(fb,Hb)
	fg=np.multiply(fg,Hg)
	
	# fr = np.fft.fftshift(f)
	# fb = np.fft.fftshift(fb)
	# fg = np.fft.fftshift(fg)

	

	#Finding inverse fft
	R=np.abs(np.fft.ifft2(f))
	B=np.abs(np.fft.ifft2(fb))
	G=np.abs(np.fft.ifft2(fg))
	#merging R,G and B scale
	mer=cv2.merge([R, G, B])
	#mer array converted to image
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
	

def psnr(image1, image2):#Code help taken from internet but modified it
    mse1 = numpy.mean( (image1 - image2) ** 2 )#computing mean square error 
    if mse1 == 0:
    	return 100;
    maxp = 255.0#max pixel value
    return 20 * math.log10(maxp / math.sqrt(mse1))


def tinver():#Code wriiten by me
	
	#reading the image using opencv
	image=cv2.imread(path)
	#reading orignal image using opencv
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	#splitting image in 3 scale R,G and B
	(B,G,R)=cv2.split(image);
	#reading the kernel image using opencv
	kernel1 = cv2.imread(path2)
	#splitting kernel image in 3 scale R,G and B
	(B2,G2,R2)=cv2.split(kernel1);
	
	#creating a matrix with zero value same as B matrix
	kb = np.zeros_like(B);
	#initializing B2 to kb from topleft side
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing it
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as G matrix
	kg = np.zeros_like(G);
	#initializing G2 to kg from topleft side
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing it
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as R matrix
	kr = np.zeros_like(R);
	#initializing R2 to kg from topleft side
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing it
	kr = kr / (np.sum(kr))
	

	#performing 2 D fft of image
	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	
	#performing 2 D fft of kernel image
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	
	#dividing fft of image with fft of kernel
	kr=np.divide(fr, Hr)
	
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	








	#Shape of matrix is deined
	M1= kr.shape[0]
	N1= kr.shape[1]
	# H is defined and
	# values in H are initialized to 1
	H1 = numpy.ones((M1,N1))
	c1 = M1/2	#center1
	c2 = N1/2	#center2
	rcut =int(w1.get()) #cut off radius
	
	t1 = 1 # the order of BLPF
	t2 = 2*t1
	# defining the convolution function for BLPF
	for i in range(1,M1):
		for j in range(1,N1):
			r1 = (i-c1)**2 + (j-c2)**2
			# euclidean distance from origin is computed
			r = math.sqrt(r1)
			# eliminating high frequency using cut-off radius 
			if r > rcut:
				H1[i,j] = 1/(1 + (r/rcut)**t1)


	#performing fftshift
	kr = np.fft.fftshift(kr)
	kb = np.fft.fftshift(kb)
	kg = np.fft.fftshift(kg)

	#normalization of H1
	H1 = H1 / np.max(H1)
	# print(np.max(H1))

	# performing the convolution
	kr = kr * H1
	kb = kb * H1
	kg = kg * H1
	
	#converting to spatial domain by IFFT
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
	#merging R , G and B scale
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
	#Calling function psnr
	d=psnr(mer,imgo)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=450
	label7.place(x=760,y=450)
	#color image is conerted to gray scale image
	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	#saving image temporary
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
	#finding ssim of image
	d=ssim(i1,i2)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=480
	label7.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing text in gui with coordinates x=600 and y=400
	label4.place(x=600,y=400)
	

def inver():#Code written by me
	#reading the image using opencv
	image=cv2.imread(path)
	#reading the orignal image using opencv
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	#splitting image in 3 components B,G and R
	(B,G,R)=cv2.split(image);
	#reading the kernel image using opencv
	kernel1=cv2.imread(path2);
	#splitting kernel image in 3 components B,G and R
	(B2,G2,R2)=cv2.split(kernel1);
	#creating a matrix with zero value same as B matrix
	kb = np.zeros_like(B);
	#initializing B2 from topleft side
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing it
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as G matrix
	kg = np.zeros_like(G);
	#initializing G2 from topleft side
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing it
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as R matrix
	kr = np.zeros_like(R);
	#initializing R2 from topleft side
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing it
	kr = kr / (np.sum(kr))
	


	#performing 2 D fft of image
	fr = np.fft.fft2(R)
	fb = np.fft.fft2(B)
	fg = np.fft.fft2(G)
	
	#performing 2 D fft of kernel image
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	
	#dividing fft of image with fft of kernel
	kr=np.divide(fr, Hr)
	kb=np.divide(fb, Hb)
	kg=np.divide(fg, Hg)
	#Finding inverse fft 
	R=np.abs(np.fft.ifft2(kr))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))
	#merging R,G and B values
	mer=cv2.merge([R, G, B])
	#converting to image
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
	#finding psnr with respect to orignal image
	d=psnr(imgo,mer)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=450
	label7.place(x=760,y=450)
	#Converting image to grayscale
	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	#temporary saving the image
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	#Converting another image to grayscale
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
	#finding ssim
	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=480
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing text in gui with coordinates x=600 and y=400
	label4.place(x=600,y=400)
	

def weiner():#Code written by me
	#reading the image using opencv
	imag1=cv2.imread(path)
	# imag1=cv2.resize(imag1,(380,380), interpolation = cv2.INTER_AREA);
	
	i1=imag1
	#splitting image in 3 components B1,G1 and R1

	(B1,G1,R1) = cv2.split(imag1);	
	#reading the kernel image using opencv
	kernel1 = cv2.imread(path2)
	#reading the image using opencv
	
	#reading the image using opencv
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	
	#splitting image in 3 components B,G and R
	(B2,G2,R2)=cv2.split(kernel1);
	#creating a matrix with zero value same as B1 matrix
	kb = np.zeros_like(B1);
	#initializing B2 from topleft side
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing it
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as G1 matrix
	kg = np.zeros_like(G1);
	#initializing G2 from topleft side
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing it
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as R1 matrix
	kr = np.zeros_like(R1);
	#initializing R2 from topleft side
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing it
	kr = kr / (np.sum(kr))
	



	#getting value from slider 2 for K
	K =int(w3.get())
	# K=.1
	K1 = np.ones_like(kb)
	K=K*K1;
	#performing 2 D fft of image for every scale
	f = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)
	#performing 2 D fft of kernel image for every scale
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
	
	#appling weiner formula
	H3b = np.conj(Hb) / (np.abs(Hb) ** 2 + K)
	H3r = np.conj(Hr) / (np.abs(Hr) ** 2 + K)
	H3g = np.conj(Hg) / (np.abs(Hg) ** 2 + K)
	#multiplying matrix in frequency domain
	k=np.multiply(H3b,f)
	kb=np.multiply(H3b,fb)
	kg=np.multiply(H3g,fg)
	#Finding inverse fft of image
	R=np.abs(np.fft.ifft2(k))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))	
	#merging R,G and B scale
	mer=cv2.merge([R, G, B])
	#matrix converted to image
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
	#finding psnr
	d=psnr(imgo,mer)
	# print(d)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=450
	label7.place(x=760,y=450)

	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
        # img2 = numpy.asarray(Image.open(argv[2]))
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)

	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=480
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing text in gui with coordinates x=600 and y=400
	label4.place(x=600,y=400)


def clsf():#Code written by me
	#reading the image using opencv
	imag1=cv2.imread(path)
	#reading the orignal image using opencv
	imgo = cv2.imread('/home/shubham/Desktop/ip_assign2/gt.png')
	
	#splitting image in 3 components B1,G1 and R1
	(B1,G1,R1) = cv2.split(imag1);	
	#reading the kernel image using opencv
	kernel1 = cv2.imread(path2)
	#splitting kernel in 3 components B2,G2 and R2
	(B2,G2,R2)=cv2.split(kernel1);
	#creating a matrix with zero value same as B1 matrix
	kb = np.zeros_like(B1);
	#initializing B2 from topleft side
	kb[0:B2.shape[0], 0:B2.shape[1]] = B2
	#normalizing it
	kb = kb / (np.sum(kb))
	#creating a matrix with zero value same as G1matrix
	kg = np.zeros_like(G1);
	#initializing G2 from topleft side
	kg[0:G2.shape[0], 0:G2.shape[1]] = G2
	#normalizing it
	kg = kg / (np.sum(kg))
	#creating a matrix with zero value same as R1 matrix
	kr = np.zeros_like(R1);
	#initializing R2 from topleft side
	kr[0:R2.shape[0], 0:R2.shape[1]] = R2
	#normalizing it
	kr = kr / (np.sum(kr))
	


	#getting value from slider 4 for y
	y =int(w2.get())
	#the laplacian kernel
	p1=np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
	# print(p1)
	
	#creating matrix with zero value with size of B1
	p = np.zeros_like(B1);
	# print(p)

	#padding zereos to the size of image
	p= np.pad(p1, [(0, B1.shape[0] - p1.shape[0]), (0, B1.shape[1] - p1.shape[1])], 'constant')
	# p = p / (np.sum(p))
	print(p)
	#performing 2 D fft of laplacian kernel
	fp = np.fft.fft2(p)
	#performing 2 D fft of image
	f = np.fft.fft2(R1)
	fb = np.fft.fft2(B1)
	fg = np.fft.fft2(G1)
	# H = np.fft.fft2(kn)
	#performing 2 D fft of kernel image
	Hb = np.fft.fft2(kb)
	Hr = np.fft.fft2(kr)
	Hg = np.fft.fft2(kg)
	


	#Applying constrained least square formula
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
	
	#multiplication in frequency domain
	k=np.multiply(H3r,f)
	kb=np.multiply(H3b,fb)
	kg=np.multiply(H3g,fg)
	#performing fft shift
	k=np.fft.fftshift(k);
	kb=np.fft.fftshift(kb);
	kg=np.fft.fftshift(kg);
	#Finding inverse fft
	R=np.abs(np.fft.ifft2(k))
	B=np.abs(np.fft.ifft2(kb))
	G=np.abs(np.fft.ifft2(kg))	
	#merging R,G and B values
	mer=cv2.merge([R, G, B])
	#converting to image
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
	#finding psnr of two image
	d=psnr(imgo,mer)
	label7=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=450
	label7.place(x=760,y=450)
	#Converting image to grayscale
	i1 = cv2.cvtColor(imgo, cv2.COLOR_BGR2GRAY)
	
	
	#temporary saving the image
	img3.save('/home/shubham/Downloads/test.png')
	i2=cv2.imread('/home/shubham/Downloads/test.png')
	#Converting another image to grayscale
	i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
	#finding ssim
	d=ssim(i1,i2)
	# print(d)
	label8=Label(root,text=d)
	#placing image in gui with coordinates x=760 and y=480
	label8.place(x=760,y=480)


	text1="Restored Image"
	label4=Label(root,text=text1)
	#placing text in gui with coordinates x=600 and y=400
	label4.place(x=600,y=400)
	


#code written by me
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
	
#Code written by me
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
	




#Code written by me
def saveas():
	#using tkinter dialog box to get directory to save file
	root.filename=tkFileDialog.asksaveasfilename(initialdir = "/",title = "Save as",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
	#saving file imagesave in directory
	imagesave.save(root.filename)
	
#creating a slider w1 to take input from user
w1=Scale(root,from_=100,to=300,resolution=10,orient=HORIZONTAL)
#placing the silder in gui with co-ordinate x=350 and y=510
w1.place(x=350,	y=510)
#setting initial value to 170  
w1.set(170)
#creating a slider w2 to take input from user
w2=Scale(root,from_=.1,to=2,resolution=.1,orient=HORIZONTAL)
#placing the silder in gui with co-ordinate x=350 and y=590
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


