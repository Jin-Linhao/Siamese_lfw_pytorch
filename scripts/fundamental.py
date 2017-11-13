import numpy as np
import cv2
import matplotlib.pyplot as plt


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