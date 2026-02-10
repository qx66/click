import cv2
import subprocess
import numpy as np
import time
def example1():
    img1 = cv2.imread("assets/img/pk/win.png")
    img2 = cv2.imread("assets/img/pk/win-next.png")
    
    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)
    good_matches = [m for m in matches if m.distance < 50]
    print(good_matches)
    print(len(good_matches))
    score = len(good_matches) / min(len(kp1), len(kp2))
    print(score)

    score = len(good_matches) / ((len(kp1) + len(kp2)) / 2)
    print(score)

def main():
    while True:
        currentImgByte = adbScreenCap()
        if currentImgByte == None:
            return
        currentImg = readByByte(currentImgByte)

        # main0
        s1,s2,s3= compareWinMain0Page(currentImg)
        print(f"Pk页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1 >= 0.5:
            adbClick(1816, 747)
            print("匹配上PK主页面，自动点击\"个人赛\"")
            continue

        # main
        s1,s2,s3= compareWinMainPage(currentImg)
        print(f"Pk页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1 >= 0.5:
            adbClick(1360, 971)
            print("匹配上PK页面，自动点击\"友谊赛\"")
            continue


        # 
        s1,s2,s3= compareWinPage(currentImg)
        print(f"匹配上获胜/失败结果页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1>=0.45:
            adbClick(1103, 972)
            print("匹配上获胜/失败结果页面，自动点击\"下一页\"")
            continue

        s1,s2,s3= compareWinNextPage(currentImg)
        print(f"匹配上获胜/失败结果页面的下一个页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1 >= 0.45:
            adbClick(1360, 970)
            print("匹配上获胜/失败结果页面的下一个页面，自动点击\"开启新的决斗\"")
            continue

        # 战斗页面 - 1: 1139，40 -> 2: x,y ->3: x,y - 考虑到战斗页面匹配比例比较差，暂时放在0.28上
        s1,s2,s3= compareWinFightPage(currentImg)
        print(f"战斗页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1 >= 0.28:
            adbClick(1220,34)
            print("匹配上战斗页面，自动点击\"设置\"")
            time.sleep(0.5)
            
            adbClick(1890,950)
            print("匹配上战斗页面，自动点击\"返回城镇\"")
            time.sleep(0.5)

            adbClick(1384,693)
            print("匹配上战斗页面，自动点击\"确定返回\"")
            continue


        # 失败页面和成功页面很相似，会被成功页面拦截，所以，打算在成功页面中模糊化点击，尽可能后的点击
        s1,s2,s3=compareResultPage(currentImg)
        print(f"挑战结果页面: s1: {s1}, s2: {s2}, s3: {s3}")
        if s1 >= 0.45:
            adbClick(1166, 972)
            print("挑战结果页面，自动点击\"下一页\"")
            continue

        print("unknow page, Please intervene manually.")

        # 考虑到adb screenshot 本身就比较慢，所以截图的时候sleep 0.5秒
        time.sleep(0.5)
def readByFile(filePath):
    return cv2.imread(filePath)

# buf为image byte
def readByByte(png_bytes):
    buf = np.frombuffer(png_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# 比较获胜后页面
def compareWinPage(currentImg):
    srcImg = cv2.imread("../assets/img/pk/win.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)

# 对比获胜后，下一场的进入页面
def compareWinNextPage(currentImg):
    srcImg = cv2.imread("../assets/img/pk/win-next.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)


# 对比战斗页面
def compareWinFightPage(currentImg):
    srcImg = cv2.imread("../assets/img/pk/fight.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)

# 对比Pk页面
def compareWinMainPage(currentImg):
    srcImg = cv2.imread("../assets/img/pk/main.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)

# 对比Pk主页面
def compareWinMain0Page(currentImg):
    srcImg = cv2.imread("../assets/img/pk/main0.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)

# 如果全程没有人操作，进入平局页面
def compareResultPage(currentImg):
    srcImg = cv2.imread("../assets/img/pk/result.png", cv2.IMREAD_GRAYSCALE)
    return score(srcImg, currentImg)

def adbScreenCap():
    cmd = [
        "adb", "exec-out", "screencap", "-p"
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return result.stdout  # PNG bytes

def adbClick(x, y):
    cmd = [
        "adb", "shell", "input", "tap", str(x), str(y)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True
    )
    return result.stdout

def newOrb():
    return cv2.ORB_create(nfeatures=2000)

def newBf():
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# G / max(K1, K2) —— “最大集合约束” - 在“信息最多的那张图”里，有多少比例能被匹配？ - 范围: [0, 1]，通常最低
# 返回相似度
def similarity(kp1, kp2, good_matches):
    similarity = len(good_matches) / max(len(kp1), len(kp2))
    return similarity

# G / min(K1, K2) —— “小图覆盖率”
def minScore(kp1, kp2, good_matches):
    return len(good_matches) / min(len(kp1), len(kp2))

# G / ((K1 + K2) / 2) —— “整体重合度”  - 范围: (0, 1]，通常偏小
def nornalScore(kp1, kp2, good_matches):
    return len(good_matches) / ((len(kp1) + len(kp2)) / 2)

def score(srcImg, currentImg):
    orb = newOrb()
    bf = newBf()

    kp1, des1 = orb.detectAndCompute(srcImg, None)
    kp2, des2 = orb.detectAndCompute(currentImg, None)

    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = [m for m in matches if m.distance < 50]

    s1 = similarity(kp1, kp2, good_matches)
    s2 = minScore(kp1, kp2, good_matches)
    s3 = nornalScore(kp1, kp2, good_matches)
    return s1, s2, s3

if __name__ == "__main__":

    main()
