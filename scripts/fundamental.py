import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

def read_img(img1_path, img2_path):
	I1 = cv2.imread(img1_path)
	I2 = cv2.imread(img2_path)
	im1 = cv2.cvtColor(I1, cv2.COLOR_RGB2GRAY)
	im2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)

	return I1, I2, im1, im2



def drawMatches(img1, kp1, img2, kp2, matches):
	rows1 = img1.shape[0]
	cols1 = img1.shape[1]
	rows2 = img2.shape[0]
	cols2 = img2.shape[1]

	out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
	out[:rows1,:cols1] = np.dstack([img1, img1, img1])
	out[:rows2,cols1:] = np.dstack([img2, img2, img2])

	match_out = []
	for mat in matches:

		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt
		match_list = []
		match_list.append((x1,y1))
		match_list.append((x2,y2))
		match_out.append(match_list)
		# print x1, y1
		# print x2, y2
		# print "---------------------------"

		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


	# print match_out
	return out, match_out

def matching(img1, img2):
	orb = cv2.ORB()

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	# print len(matches)
	# print kp1[matches[0].trainIdx].pt

	img3, match_out = drawMatches(img1,kp1,img2,kp2,matches[:50])
	# plt.imshow(img3)
	# plt.show()
	return match_out



def ransac(matches):
	max_count = 0
	t = []
	for times in range(0,300):
		lst = random.sample(matches,15)
		# print lst
		x = lst[0][0][0]
		y = lst[0][0][1]
		x_p = lst[0][1][0]
		y_p = lst[0][1][1]
		# print y
		r = np.mat([[int(x), y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p],
					[0, 0, 0, x, y, 1, -y_p * x, -y_p * y, -y_p]], dtype = int)
		A = r
		for i in range(1,15):
			x = lst[i][0][0]
			y = lst[i][0][1]
			x_p = lst[i][1][0]
			y_p = lst[i][1][1]
			r = np.mat([[x, y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p],
						[0, 0, 0, x, y, 1, -y_p * x, -y_p * y, -y_p]], dtype = int)
			A = np.vstack((A, r))
		# print "====================="	
		# print A
		# print "====================="
	
		v,w = np.linalg.eig(np.transpose(A) * A)
		v = abs(np.real(v))
		w = np.real(w)
		# print w
		# print "-----"
		t_ransac = np.reshape(w[:,np.argmin(v)],(3,3))
		# print t_ransac

		count = 0
		for j in matches:

			k = t_ransac * np.mat([[j[0][0]],[j[0][1]],[1]])
			
			d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[j[1][0]],[j[1][1]],[1]])
			# print np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]), np.mat([[j[1][0]],[j[1][1]],[1]])
			print sum(np.multiply(d,d))
			if sum(np.multiply(d,d)) < 300:
				count += 1

		if count > max_count:
			max_count = count
			t = t_ransac
	print t
	return t


def plot(matches, t, im1, im2):

	match_temp = []
	for m in matches:

		k = t * np.mat([[m[0][0]],[m[0][1]],[1]])
		d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[m[0][0]],[m[0][1]],[1]])
		# print sum(np.multiply(d,d))
		if sum(np.multiply(d,d)) < 70000:
			match_temp.append(m)
	I = np.concatenate((im1,im2),axis=1)
	for line in match_temp:

		pt1 = (int(line[0][0]), int(line[0][1]))
		pt2 = (int(line[1][0]) + im1.shape[1], int(line[1][1]))
		cv2.line(I,pt1,pt2,(255,255,0),2)
		
	plt.figure()
	plt.imshow(I)
	plt.show()

	A = np.mat(np.zeros((1,9)))

	for i in match_temp:
		x = i[0][0]
		y = i[0][1]
		x_p = i[1][0]
		y_p = i[1][1]
		
		r = np.mat([[x * x_p, x * y_p, x, y * x_p, y * y_p, y, x_p, y_p, 1]])
		A = np.vstack((A,r))

	A = A[1:A.shape[0],:]
	v,w = np.linalg.eig(np.transpose(A) * A)
	v = abs(np.real(v))
	w = np.real(w)
	f = np.reshape(w[:,np.argmin(v)],(3,3))

	lst = random.sample(match_temp,4)
	I = np.concatenate((im1,im2),axis=1)
	for i in lst:
		ul = i[0][0]
		vl = i[0][1]
		ur = i[1][0]
		vr = i[1][1]
		k = np.mat([[ul,vl,1]]) * f
		a = k[0,0]
		b = k[0,1]
		c = k[0,2]
		color = (random.randrange(250),random.randrange(250),random.randrange(250))
		cv2.circle(I,(int(ul),int(vl)),10,color,2)

		cv2.circle(I,(int(ur)+im1.shape[1],int(vr)),10,color,2)
		x1 = -c / a
		y1 = 0
		x2 = 0
		y2 = -c / b
		if x1 < 0:
			if (-c - b * im1.shape[1]) / a < im1.shape[0]:
				x1 = (-c - b * im1.shape[1]) / a
				y1 = im1.shape[1]
			else:
				x1  = im1.shape[0]
				y1  = (-c - a * x1) / b
		if x1 > im1.shape[0]:
			x1  = im1.shape[0]
			y1  = (-c - a * x1) / b
			if (-c - b * im1.shape[1]) / a > 0:
				x2 = (-c - b * im1.shape[1]) / a
				y2 = im1.shape[1]
		if y2 < 0:
			x2 = im1.shape[0]
			y2 = (-c - a * x2) / b
		if y2 > im1.shape[1]:
			x2 = (-c - b * im1.shape[1]) / a
			y2 = im1.shape[1]
		
		
		cv2.line(I,(int(y1 + im1.shape[1]),int(x1)),(int(y2 + im1.shape[1]),int(x2)),color,2)
	

	plt.figure()
	plt.imshow(I)
	plt.show()
		

# time.sleep(1000)
if __name__ == "__main__":
	I1, I2, im1, im2 = read_img('/home/lh/cv_ws/src/hw3/data/hopkins1.JPG', 
								'/home/lh/cv_ws/src/hw3/data/hopkins2.JPG')
	matches = matching(im1, im2)
	t = ransac(matches)
	plot(matches, t, I1, I2)