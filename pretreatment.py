import cv2
from PIL import Image
import numpy as np
import os


def getbinaryimage(filedir,img_name):
    img_name = filedir + '/' + img_name
    img = cv2.imread(img_name)
    grayimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #灰值化
    binaryimg = cv2.adaptiveThreshold(grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 1) #二值化
    return binaryimg


def denoise(binaryimg,img_name,N,M): # 点降噪 3x3领域框
    if (binaryimg.ndim!=2):
        print("图像二值化失败")
        return

    img_height = binaryimg.shape[0]
    img_width = binaryimg.shape[1]
    for k in range(M):
        for j in range(img_width):
            for i in range(img_height):
                if (binaryimg[i,j]==255):#第i行j列像素点为白点
                    if (i==0):  #第一行
                        if (j==0):   #左上角点
                            r_pixel=int(binaryimg[i,j+1])
                            d_pixel=int(binaryimg[i+1,j])
                            rd_pixel=int(binaryimg[i+1,j+1])
                            binaryimg[i,j]=255-int((3 - (r_pixel + d_pixel +
                                                     rd_pixel)/255) / 2) * 255
                        elif (j==(img_width-1)):    #右上角点
                            l_pixel = int(binaryimg[i, j - 1])
                            d_pixel = int(binaryimg[i + 1, j])
                            ld_pixel = int(binaryimg[i + 1, j - 1])
                            # 相邻 3个点中，大于等于2个点为白，则为白
                            binaryimg[i, j] = 255-int((3 - (l_pixel + d_pixel
                                                   + ld_pixel) / 255) / 2) * 255
                        else:#第一行中间列
                            r_pixel = int(binaryimg[i, j + 1])
                            d_pixel = int(binaryimg[i + 1, j])
                            rd_pixel = int(binaryimg[i + 1, j + 1])
                            l_pixel = int(binaryimg[i, j - 1])
                            ld_pixel = int(binaryimg[i + 1, j - 1])
                            # 相邻 5个点中，大于等于3个点为白，则为白
                            binaryimg[i, j] = 255-int((5 - (l_pixel + d_pixel + ld_pixel
                                                   + r_pixel + rd_pixel) / 255) / 3) * 255
                    elif (i==(img_height-1)):    #最后一行
                        if (j==0):  #左下角点
                            r_pixel = int(binaryimg[i, j + 1])
                            u_pixel = int(binaryimg[i - 1, j])
                            ru_pixel = int(binaryimg[i - 1, j + 1])
                            binaryimg[i, j] = 255-int((3 - (r_pixel + u_pixel +
                                                        ru_pixel) / 255) / 2) * 255
                        elif (j==(img_width-1)):    #右下角点
                            l_pixel = int(binaryimg[i, j - 1])
                            u_pixel = int(binaryimg[i - 1, j])
                            lu_pixel = int(binaryimg[i - 1, j - 1])
                            binaryimg[i, j] = 255-int((3 - (l_pixel + u_pixel
                                                   + lu_pixel) / 255) / 2) * 255
                        else:   #最后一行中间列
                            r_pixel = int(binaryimg[i, j + 1])
                            u_pixel = int(binaryimg[i - 1, j])
                            ru_pixel = int(binaryimg[i - 1, j + 1])
                            l_pixel = int(binaryimg[i, j - 1])
                            lu_pixel = int(binaryimg[i - 1, j - 1])
                            # 相邻 5个点中，大于等于3个点为白，则为白
                            binaryimg[i, j] = 255-int((5 - (l_pixel + u_pixel + lu_pixel
                                                   + r_pixel + ru_pixel) / 255) / 3) * 255
                    else:   #中间行
                        if (j==0):  #第一列中间行
                            r_pixel = int(binaryimg[i, j + 1])
                            d_pixel = int(binaryimg[i + 1, j])
                            rd_pixel = int(binaryimg[i + 1, j + 1])
                            u_pixel = int(binaryimg[i - 1, j])
                            ru_pixel = int(binaryimg[i - 1, j + 1])
                            # 相邻 5个点中，大于等于3个点为白，则为白
                            binaryimg[i, j] = 255-int((5 - (r_pixel+d_pixel+rd_pixel+
                                                        u_pixel+ru_pixel)/255) / 3) * 255
                        elif (j==(img_width-1)):    #最后一列中间行
                            l_pixel = int(binaryimg[i, j - 1])
                            d_pixel = int(binaryimg[i + 1, j])
                            ld_pixel = int(binaryimg[i + 1, j - 1])
                            u_pixel = int(binaryimg[i - 1, j])
                            lu_pixel = int(binaryimg[i - 1, j - 1])
                            # 相邻 5个点中，大于等于3个点为白，则为白
                            binaryimg[i, j] = 255-int((5 - (l_pixel + d_pixel + ld_pixel
                                                   + u_pixel + lu_pixel) / 255) / 3) * 255
                        else:   #中间部分
                            l_pixel = int(binaryimg[i, j - 1])
                            r_pixel = int(binaryimg[i, j + 1])
                            d_pixel = int(binaryimg[i + 1, j])
                            u_pixel = int(binaryimg[i - 1, j])
                            ld_pixel = int(binaryimg[i + 1, j - 1])
                            rd_pixel = int(binaryimg[i + 1, j + 1])
                            lu_pixel = int(binaryimg[i - 1, j - 1])
                            ru_pixel = int(binaryimg[i - 1, j + 1])
                            # 相邻 8个点中，大于N个点为白，则为白
                            num_white =(l_pixel+r_pixel+d_pixel+
                                        u_pixel+ld_pixel+rd_pixel+
                                        lu_pixel+ru_pixel)/255
                            if (num_white >= N):
                                binaryimg[i,j]=255
                            else:
                                binaryimg[i,j]=0
    #kernel = np.ones((2, 2), np.uint8)
    #dilation = cv2.dilate(binaryimg, kernel, iterations=1)
    #cv2.imwrite(image_output_path + img_name, binaryimg)
    '''for i in range(18):
        print(binaryimg[i])'''
    return binaryimg


def split_number(binaryimg,img_name):     #考虑到验证码是按行顺序填写的，几乎不可能出现两个数字在同一列出现且不黏连
    img_height = binaryimg.shape[0]
    img_width = binaryimg.shape[1]

    column_sum = [0]*img_width  # 每一列的像素和
    for j in range(img_width):
        for i in range(img_height):
            if (binaryimg[i,j]==255):#如果是白点
                column_sum[j]=column_sum[j]+1
        #if (column_sum[j]<2):   #阈值判定
            #column_sum[j]=0
    splited_num=[]  #保存被分割的各个部分起始列和末尾列，要求有至少连续4列有像素值
    i=0
    while(i<img_width-3):
        session_start=0
        session_end=0
        if ((column_sum[i]*column_sum[i+1]*column_sum[i+2])!=0):
            session_start=i
            j=i+3
            while(j<img_width and column_sum[j]!=0):
                j=j+1
            session_end=j-1
            i=j
            splited_num.append((session_start,session_end))
        elif ((column_sum[i]*column_sum[i+1])!=0 and (column_sum[i]+column_sum[i+1])>=10):
            session_start=i
            session_end=i+1
            splited_num.append((session_start, session_end))
        else:
            i=i+1

    #二次分割，舍弃掉总像素很低的部分

    tmp = []
    for i in range(len(splited_num)):
        sum=0
        j=splited_num[i][0]
        while(j<=splited_num[i][1]):
            sum=sum+column_sum[j]
            j=j+1
        if (sum < 10):  #总像素太少，极有可能是未完全消除的噪音
            tmp.append(i)   #不能在循环中直接删除列表元素
    for i in range(len(tmp)):
        print(i)
        del splited_num[tmp[-1]]
        del tmp[-1]

    #切割黏连的字符,若某一部分的宽度过宽进行对半切割
    joint=[]
    for i in range(len(splited_num)):
        if (splited_num[i][1]-splited_num[i][0]>13):
            joint.append(i)

    for i in range(len(joint)):
        j=joint[i]+i
        tmp_left=splited_num[j][0]
        tmp_right=splited_num[j][1]
        mid = int((tmp_left+tmp_right)/2)
        del splited_num[j]
        splited_num.insert(j,(tmp_left,mid))
        splited_num.insert(j+1,(mid,tmp_right))

    #对于每一块区域，再求出最上点和最下点，截出矩形
    for i in range(len(splited_num)):
        c_left=splited_num[i][0]
        c_right=splited_num[i][1]
        r_top=-1
        r_bottom=-1
        for j in range(img_height):
            for k in range(c_left,c_right+1):
                if (r_top==-1 and binaryimg[j,k]==255 ):
                    r_top=j
                    break
        for j in range(img_height-1,-1,-1):
            for k in range(c_left,c_right+1):
                if (r_bottom==-1 and binaryimg[j,k]==255 ):
                    r_bottom=j
                    break
        section = binaryimg[r_top:r_bottom+1,c_left:c_right+1]
        if (os.path.exists("./result/"+img_name)==False):
            os.makedirs("./result/"+img_name)
        cv2.imwrite("./result/"+img_name+"/"+str(i)+".jpg",section)
    return len(splited_num)





def irrotionated(image_input_path,img_name,size,image_output_path):  #去旋转(单张图片处理)
    img = cv2.imread(image_input_path+img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height = img.shape[0]
    img_width = img.shape[1]
    row_sum = [0 for x in range(0, img_height+1)]
    for k in range(img_height):
        for j in range(img_width):
            if (img[k, j] == 255):
                row_sum[k] = row_sum[k] + 1

    row_splited_num = []  # 保存被分割的各个部分起始列和末尾列，要求有中间间断不能超过1行
    j = 0
    while (j < img_height):
        session_start = 0
        session_end = 0
        if (row_sum[j] != 0):
            session_start = j
            k = j
            while (k < img_height and (row_sum[k] + row_sum[k + 1]) != 0):
                k = k + 1
            session_end = k
            j = k
            if (session_end - session_start >= 4):
                row_splited_num.append((session_start, session_end))
        else:
            j = j + 1
    if (len(row_splited_num) != 1):
        print(row_splited_num)
        if ((row_splited_num[0][1] - row_splited_num[0][0]) <
                (row_splited_num[1][1] - row_splited_num[1][0])):
            del row_splited_num[0]
        else:
            del row_splited_num[1]
    section = img[row_splited_num[0][0]:row_splited_num[0][1],
              0: img_width]
    column_sum = [0 for x in range(0, img_width)]
    for j in range(img_width):
        for k in range(row_splited_num[0][1] - row_splited_num[0][0]):
            if (section[k, j] == 255):
                column_sum[j] = column_sum[j] + 1
    for j in range(img_width):
        if (column_sum[j] != 0):
            c_left = j
            break
    for j in range(img_width - 1, 0, -1):
        if (column_sum[j] != 0):
            c_right = j
            break
    section = img[row_splited_num[0][0]:row_splited_num[0][1],
              c_left: c_right]
    img = section
    img_height = img.shape[0]
    img_width = img.shape[1]

    addwidth=int((size-img_width)/2)
    addheight=int((size-img_height)/2)
    a = cv2.copyMakeBorder(img, addheight, addheight, addwidth, addwidth,
                                cv2.BORDER_CONSTANT, value=0)#向外扩充成size*size
    
    #a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)  # 灰值化
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(a, kernel, iterations=1)  #先对图像进行膨胀，提高轮廓识别成功率
    image, contours, hierarchy = cv2.findContours(dilation,3,2)
    cnt=contours[0]
    rect = cv2.minAreaRect(cnt)  # 最小外接矩形
    degree=rect[2]  #范围是-90°到0°，当为-30°时，可能是左偏30°或右偏60°，一般而言，偏移不会超过45°
    width = rect[1][0]
    height = rect[1][1]
    if (degree > -45 and degree<-0.0):
        if (width> height):
            degree = 90 + degree
        else:
            degree = 360 + degree
    elif(degree<=-45):
        if (width < height):
            degree = 360 + degree
        else:
            degree = 90 + degree  # 右偏取正，左偏取负
    M = cv2.getRotationMatrix2D(rect[0], degree, 1)#旋转矩阵
    dst = cv2.warpAffine(a, M, (size, size))


    if (os.path.exists(image_output_path)==False):
        os.makedirs(image_output_path)
    
    cv2.imwrite(image_output_path + img_name, dst)
