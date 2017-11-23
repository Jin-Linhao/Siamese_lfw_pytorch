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

	for mat in matches:

		img1_idx = mat.queryIdx
		img2_idx = mat.trainIdx

		(x1,y1) = kp1[img1_idx].pt
		(x2,y2) = kp2[img2_idx].pt

		cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
		cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

		cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255,0,0), 1)


	# Show the image
	# cv2.imshow('Matched Features', out)
	# cv2.waitKey(0)
	# cv2.destroyWindow('Matched Features')

	# Also return the image if you'd like a copy
	return out

def matching(img1, img2):
	orb = cv2.ORB()

	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

	matches = bf.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)

	print len(matches)
	print kp1[matches[0].trainIdx].pt
	# print matches[0].queryIdx, matches[1].queryIdx
	# print matches[0].distance, matches[1].distance
	img3 = drawMatches(img1,kp1,img2,kp2,matches[:20])
	# plt.imshow(img3)
	# plt.show()
	matches2 = []

	for i in xrange(20):
		match_list = []
		match_list.append(kp1[matches[i].trainIdx].pt)
		match_list.append(kp2[matches[i].trainIdx].pt)
		matches2.append(match_list)
	return matches2



def ransac(matches):
	max_count = 0
	t = []
	for times in range(0,1000):
		lst = random.sample(matches,10)
		# print lst
		x = lst[0][0][0]
		y = lst[0][0][1]
		x_p = lst[0][1][0]
		y_p = lst[0][1][1]
		# print y
		r = np.mat([[int(x), y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p],
					[0, 0, 0, x, y, 1, -y_p * x, -y_p * y, -y_p]], dtype = int)
		A = r
		for i in range(1,10):
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
		t_ransac = np.reshape(w[:,np.argmin(v)],(3,3))
		print t_ransac

		count = 0
		for j in matches:

			k = t_ransac * np.mat([[j[0][0]],[j[0][1]],[1]])
			
			d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[j[1][0]],[j[1][1]],[1]])
			# print np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]), np.mat([[j[1][0]],[j[1][1]],[1]])
			print sum(np.multiply(d,d))
			if sum(np.multiply(d,d)) < 30000:
				count += 1

		if count > max_count:
			max_count = count
			t = t_ransac
		print t, "===="
		return t
# [[  7.57712355e-04  -4.31160028e-03   9.57841400e-01]
#  [  2.30290235e-03  -4.53604330e-03   2.87211884e-01]
#  [  1.45246156e-06   1.23889116e-06  -2.03533008e-03]]
def plot(matches, t, im1, im2):

	match_temp = []
	for m in matches:
		k = t * np.mat([[m[0][0]],[m[0][1]],[1]])
		d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[m[0][0]],[m[0][1]],[1]])
		print sum(np.multiply(d,d))
		if sum(np.multiply(d,d)) < 50000:
			match_temp.append(m)

	A = np.mat(np.zeros((1,9)))
	print match_temp

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

	lst = random.sample(match_temp,5)
	I = np.concatenate((im1,im2),axis=1)
	for i in lst:
		ul = i[0][0]
		vl = i[0][1]
		k = np.mat([[ul,vl,1]]) * f
		a = k[0,0]
		b = k[0,1]
		c = k[0,2]
		color = (random.randrange(250),random.randrange(250),random.randrange(250))
		cv2.circle(I,(int(vl),int(ul)),10,color,2)
		
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