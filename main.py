import cv2
import numpy as np
import math
import os

img_base_path = "images/"
scaling_factor = 0.25
visualize = True
save = False

def main():
    images = os.listdir("images")
    for file in images:
        if file == ".gitignore": 
            continue

        original_img = cv2.imread(img_base_path + file)
        org_max_y, org_max_x, _ = original_img.shape
        target_height = org_max_y
        
        img = preprocessing(original_img)
        
        kp = fastCornerDetection(img)
        corners = cv2.KeyPoint.convert(kp)

        paper_corners = findPaperCorners(corners, img)
        paper_corners = resetScaling(paper_corners)

        M, _ = cv2.findHomography(np.array(list(paper_corners)), np.array([[0,0], [int(target_height / math.sqrt(2)), 0], [0, target_height], [int(target_height / math.sqrt(2)), target_height]]))
        img_warped = cv2.warpPerspective(original_img, M, (int(target_height / math.sqrt(2)), target_height))

        combination = cv2.hconcat([original_img, img_warped])
        combination = resizeImage(combination, factor=0.4)

        if visualize:
            cv2.imshow("WARPED DOCUMENT", combination)
            cv2.waitKey(0)
        
        if save:
            combination = cv2.hconcat([original_img, img_warped])
            cv2.imwrite(img_base_path + "combined_" + file, combination)
    return


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resizeImage(img)
    img = cv2.GaussianBlur(img, (3,3), 0)
    return img


def resizeImage(img, factor=scaling_factor):    
    w = int(img.shape[1] * factor)
    h = int(img.shape[0] * factor)
    dims = (w, h)
    return cv2.resize(img, dims, interpolation=cv2.INTER_AREA)


def fastCornerDetection(img):
    fast = cv2.FastFeatureDetector_create()
    kp = fast.detect(img, None)

    if visualize:
        img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
        cv2.imshow("FAST CORNER DETECTION", img)
        cv2.waitKey(1)
    return kp 


def findPaperCorners(detections, img):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    (max_y, max_x) = img.shape

    tl = tr = bl = br = None
    tl_sum = tr_sum = bl_sum = br_sum = float('inf')

    for d in detections:
        d_x = int(d[0])
        d_y = int(d[1])

        # tl
        sum = d_x + d_y
        if (sum < tl_sum): tl, tl_sum = [int(i) for i in d], sum

        # tr
        sum = (max_x - d_x) + d_y
        if (sum < tr_sum): tr, tr_sum = [int(i) for i in d], sum

        # bl
        sum = d_x + (max_y - d_y)
        if (sum < bl_sum): bl, bl_sum = [int(i) for i in d], sum

        # br
        sum = (max_x - d_x) + (max_y - d_y)
        if (sum < br_sum): br, br_sum = [int(i) for i in d], sum

        if visualize:
            canvas = cimg.copy()
            canvas = cv2.circle(canvas, (d_x, d_y), radius=3, color=(0,0,255), thickness=-1)
            canvas = cv2.circle(canvas, (tl[0], tl[1]), radius=3, color=(0,255,0), thickness=-1)
            canvas = cv2.circle(canvas, (tr[0], tr[1]), radius=3, color=(0,255,0), thickness=-1)
            canvas = cv2.circle(canvas, (bl[0], bl[1]), radius=3, color=(0,255,0), thickness=-1)
            canvas = cv2.circle(canvas, (br[0], br[1]), radius=3, color=(0,255,0), thickness=-1)
            cv2.imshow("FIND PAPER CORNERS", canvas)
            cv2.waitKey(20)

    return (tl, tr, bl, br)


def resetScaling(paper_corners):
    for corner in paper_corners:
        corner[0] = corner[0] / scaling_factor
        corner[1] = corner[1] / scaling_factor
    return paper_corners


if __name__ == "__main__":
    main()