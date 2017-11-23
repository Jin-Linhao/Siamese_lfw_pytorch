import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

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

def matching():
	img1 = cv2.imread('/home/lh/cv_ws/src/hw3/data/hopkins1.JPG',0)
	img2 = cv2.imread('/home/lh/cv_ws/src/hw3/data/hopkins2.JPG',0)

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
	for times in range(0,300):
		lst = random.sample(matches,5)
		# print lst
		x = lst[0][0][0]
		y = lst[0][0][1]
		x_p = lst[0][1][0]
		y_p = lst[0][1][1]
		# print y
		r = np.mat([[int(x), y, 1, 0, 0, 0, -x_p * x, -x_p * y, -x_p],
					[0, 0, 0, x, y, 1, -y_p * x, -y_p * y, -y_p]], dtype = int)
		A = r
		for i in range(1,5):
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
		# print t_ransac

		count = 0
		for j in matches:

			k = t_ransac * np.mat([[j[0][0]],[j[0][1]],[1]])
			
			d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[j[1][0]],[j[1][1]],[1]])
			# print np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]), np.mat([[j[1][0]],[j[1][1]],[1]])
			# print sum(np.multiply(d,d))
			if sum(np.multiply(d,d)) < 100:
				count += 1
		t = []
		if count > max_count:
			max_count = count
			t = t_ransac
		print t, "===="
		return t

def plot(matches, t):
	match_temp = []
	for m in matches:
	    k = t * np.mat([[m[0][0]],[m[0][1]],[1]])
	    d = np.mat([[(k[0,0] / k[2,0])],[k[1,0] / k[2,0]],[1]]) - np.mat([[m[0][0]],[m[0][1]],[1]])
	    print sum(np.multiply(d,d))
	    if sum(np.multiply(d,d)) < 100:
	        match_temp.append(m)

	A = np.mat(np.zeros((1,9)))
	print match_temp

	# for i in match_temp:
	#     x = i[0][0][0]
	#     y = i[0][0][1]
	#     x_p = i[0][1][0]
	#     y_p = i[0][1][1]
		
	#     r = np.mat([[x * x_p, x * y_p, x, y * x_p, y * y_p, y, x_p, y_p, 1]])
	#     A = np.vstack((A,r))

	# A = A[1:A.shape[0],:]
	# v,w = np.linalg.eig(np.transpose(A) * A)
	# v = abs(np.real(v))
	# w = np.real(w)
	# f = np.reshape(w[:,np.argmin(v)],(3,3))

	# lst = random.sample(match_temp,8)
	# I = np.concatenate((I1,I2),axis=1)
	# for i in lst:
	#     ul = location1[i[0],0]
	#     vl = location1[i[0],1]
	#     k = np.mat([[ul,vl,1]]) * f
	#     a = k[0,0]
	#     b = k[0,1]
	#     c = k[0,2]
	#     color = (random.randrange(250),random.randrange(250),random.randrange(250))
	#     cv2.circle(I,(vl,ul),10,color,2)
		
	#     x1 = -c / a
	#     y1 = 0
	#     x2 = 0
	#     y2 = -c / b
	#     if x1 < 0:
	#         if (-c - b * im1.shape[1]) / a < im1.shape[0]:
	#             x1 = (-c - b * im1.shape[1]) / a
	#             y1 = im1.shape[1]
	#         else:
	#             x1  = im1.shape[0]
	#             y1  = (-c - a * x1) / b
	#     if x1 > im1.shape[0]:
	#         x1  = im1.shape[0]
	#         y1  = (-c - a * x1) / b
	#         if (-c - b * im1.shape[1]) / a > 0:
	#             x2 = (-c - b * im1.shape[1]) / a
	#             y2 = im1.shape[1]
	#     if y2 < 0:
	#         x2 = im1.shape[0]
	#         y2 = (-c - a * x2) / b
	#     if y2 > im1.shape[1]:
	#         x2 = (-c - b * im1.shape[1]) / a
	#         y2 = im1.shape[1]
		
	#     cv2.line(I,(int(y1 + im1.shape[1]),int(x1)),(int(y2 + im1.shape[1]),int(x2)),color,2)
	# plt.figure()
	# plt.imshow(I)
	# plt.show()
		

# time.sleep(1000)
if __name__ == "__main__":
	matches = matching()
	t = ransac(matches)
	plot(matches, t)