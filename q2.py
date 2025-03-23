from imutils import face_utils
import dlib
import cv2
import numpy as np
import random


# def drawTriangles(img, triangles):
#     imgDraw = img.copy()
#     for i in range(len(triangles)):
#         t = triangles[i]
#         v = np.array([[int(t[1]), int(t[0])], [int(t[3]), int(t[2])], [int(t[5]), int(t[4])]])
#         imgDraw = cv2.polylines(imgDraw, [v.reshape((-1, 1, 2))], True, (0, 0, 255))
#     return imgDraw
#
#
# def drawPoints(img, points):
#     imgDraw = img.copy()
#     for i in range(len(points)):
#         imgDraw = cv2.circle(imgDraw, (points[i][1], points[i][0]), 2, (0, 255, 0), -1)
#     return imgDraw


def findTriangles(triangles1, points1, points2):
    triangles2 = []
    for i in range(triangles1.shape[0]):
        p1 = np.array([triangles1[i][0], triangles1[i][1]], dtype=int)
        p2 = np.array([triangles1[i][2], triangles1[i][3]], dtype=int)
        p3 = np.array([triangles1[i][4], triangles1[i][5]], dtype=int)
        n1 = np.where(np.all(points1 == p1, axis=1))[0][0]
        n2 = np.where(np.all(points1 == p2, axis=1))[0][0]
        n3 = np.where(np.all(points1 == p3, axis=1))[0][0]
        triangles2.append([points2[n1][0], points2[n1][1],
                           points2[n2][0], points2[n2][1],
                           points2[n3][0], points2[n3][1]])
    return triangles2


def iterate(img1, img2, triangles1, triangles2, triangles3, a):
    imgAns = np.zeros(img1.shape)
    for i in range(len(triangles1)):
        t1 = np.float32([[triangles1[i][1], triangles1[i][0]], [triangles1[i][3], triangles1[i][2]],
                         [triangles1[i][5], triangles1[i][4]]])
        t2 = np.float32([[triangles2[i][1], triangles2[i][0]], [triangles2[i][3], triangles2[i][2]],
                         [triangles2[i][5], triangles2[i][4]]])
        t3 = np.float32([[triangles3[i][1], triangles3[i][0]], [triangles3[i][3], triangles3[i][2]],
                         [triangles3[i][5], triangles3[i][4]]])
        matrix1 = cv2.getAffineTransform(t1, t3)
        matrix2 = cv2.getAffineTransform(t2, t3)
        imgWarped1, imgWarped2 = img1.copy(), img2.copy()
        imgWarped1 = cv2.warpAffine(imgWarped1, matrix1, (img1.shape[1], img1.shape[0]))
        imgWarped2 = cv2.warpAffine(imgWarped2, matrix2, (img1.shape[1], img1.shape[0]))
        mask = np.zeros((img1.shape[0], img1.shape[1]))
        mask = cv2.fillPoly(mask, [np.array([[triangles3[i][1], triangles3[i][0]], [triangles3[i][3], triangles3[i][2]],
                                             [triangles3[i][5], triangles3[i][4]]], dtype=int)], color=255)
        maski, maskj = np.where(mask == 255)
        imgAns[maski, maskj, :] = a * imgWarped2[maski, maskj, :] + (1 - a) * imgWarped1[maski, maskj, :]
    return imgAns


def morph(img1, img2):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    rect1 = detector(img1, 0)[0]
    rect2 = detector(img2, 0)[0]
    points1 = predictor(img1, rect1)
    points2 = predictor(img2, rect2)
    points1 = face_utils.shape_to_np(points1)
    points2 = face_utils.shape_to_np(points2)
    points1[:, [0, 1]] = points1[:, [1, 0]]
    points2[:, [0, 1]] = points2[:, [1, 0]]
    morePoints = np.array(
        [[0, 0], [height, 0], [0, width], [height, width], [0, int(width / 2)], [height, int(width / 2)],
         [int(height / 2), 0], [int(height / 2), width]])
    # manual points
    morePoints1=[[641,89],[627,150],[608,217],[415,223],[368,209],[309,207],[239,209],[150,209],[82,237],[31,290],[19,370],[31,453],[62,504],[128,543],[204,557],[286,566],[340,573],[399,551],[604,549],[627,594],[646,650],[152,283],[138,440],[156,369]]
    morePoints2=[[621,123],[569,168],[531,233],[420,233],[369,205],[301,199],[250,204],[188,198],[126,218],[73,267],[36,357],[47,458],[79,516],[125,559],[209,575],[287,580],[356,585],[410,550],[524,555],[561,615],[613,674],[205,266],[188,419],[207,363]]
    points1 = np.vstack((points1, morePoints))
    points1 = np.vstack((points1, morePoints1))
    points2 = np.vstack((points2, morePoints))
    points2 = np.vstack((points2, morePoints2))
    rect = (0, 0, height + 1, width + 1)
    subdiv = cv2.Subdiv2D(rect)
    for i in range(len(points1)):
        subdiv.insert((points1[i][0], points1[i][1]))
    triangles1 = subdiv.getTriangleList()
    triangles2 = findTriangles(triangles1, points1, points2)
    imagesList = []
    imagesList.append(img1)
    k = 50
    v = (points2 - points1)
    for i in range(k):
        print("\r", str(int(i / (k - 1) * 100)), "%", end="")
        a = i / (k - 1)
        points3 = v * a + points1
        triangles3 = findTriangles(triangles1, points1, points3)
        imagesList.append(iterate(img1, img2, triangles1, triangles2, triangles3, a))
    imagesList.append(img2)
    return imagesList

# def morphAll(imgList):
#     imagesList=[]
#     for i in range(len(imgList)-1):
#         imgMorph=morph(imgList[i],imgList[i+1])
#         for j in range(len(imgMorph)):
#             imagesList.append(imgMorph[j])
#     return imagesList

I1 = cv2.imread("1.jpg")
I2 = cv2.imread("2.jpg")
height, width, channels = I2.shape
imagesList = morph(I1,I2)
out = cv2.VideoWriter("res2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
for i in range(len(imagesList)):
    out.write(np.array(imagesList[i], dtype='uint8'))
for i in range(len(imagesList)):
    out.write(np.array(imagesList[len(imagesList) - i - 1], dtype='uint8'))
out.release()
