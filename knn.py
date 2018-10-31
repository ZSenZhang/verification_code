import pretreatment 
import os
import numpy as np
from PIL import Image

 
def binaryzation(img):
	img_arr = np.array(img, 'i') # 28px * 28px 灰度图像
	img_normlization = np.round(img_arr/255) # 对灰度值进行归一化
	img_arr2 = np.reshape(img_normlization, (1,-1)) # 1 * 784 矩阵
	return img_arr2

 
def load_data(train_data_path):
	files = os.listdir(train_data_path)
	train_img_num = len(files)
 
	img_mat = np.empty((train_img_num, 1, 28, 28), dtype = "float32")
 
	train_data = np.empty((train_img_num, 28 * 28), dtype = "float32")
	train_label = np.empty((train_img_num), dtype = "uint8")
 
	print ("loading train data...")
	for i in range(train_img_num):
		file = files[i]
		file_path = os.path.join(train_data_path, file)
		img_mat[i] = Image.open(file_path)
		train_data[i] = binaryzation(img_mat[i])
		train_label[i] = int(file.split('.')[0])
 
	return train_data, train_label
 
def KNN(test_vec, train_data, train_label, k):
	train_data_size = train_data.shape[0]
	dif_mat = np.tile(test_vec, (train_data_size, 1)) - train_data
	sqr_dif_mat = dif_mat ** 2
	sqr_dis = sqr_dif_mat.sum(axis = 1)
 
	sorted_idx = sqr_dis.argsort()
 
	class_cnt = {}
	maxx = 0
	best_class = 0
	for i in range(k):
		tmp_class = train_label[sorted_idx[i]]
		tmp_cnt = class_cnt.get(tmp_class, 0) + 1
		class_cnt[tmp_class] = tmp_cnt
		if(tmp_cnt > maxx):
			maxx = tmp_cnt
			best_class = tmp_class
	return best_class
 
def knn_test(test_img_path,img_name,K,train_data, train_label):
	for i in range(len(img_name)):
		if img_name[i] == '.':
			break
	img_name_tmp = img_name[0:i]

	N = 2 #去噪参数
	M = 2 #去噪次数
	
	print("pretreatment for test")
	binaryimg = pretreatment.getbinaryimage(test_img_path, img_name)
	binaryimg = pretreatment.denoise(binaryimg, img_name, N, M)
	number_num = pretreatment.split_number(binaryimg,img_name_tmp)
	files = os.listdir("./result/"+img_name_tmp)
	sort_path = "./result/"+img_name_tmp+"/sorting/"
	for file in files:
		pretreatment.irrotionated("./result/"+img_name_tmp+"/",file,28,sort_path)

	print("testing...")

	files = os.listdir(sort_path)
 
	test_img = np.empty((1, 1, 28, 28), dtype = "float32")
	test_data = np.empty((1, 28 * 28), dtype = "float32")
	test_result = np.empty((1, len(files)), dtype = "int")

	i = 0
	files.sort()
	for file in files:
		test_img = Image.open(sort_path+file)
		test_data = binaryzation(test_img)
		test_result[0][i] = KNN(test_data, train_data, train_label, K)
		i += 1

	return test_result

if __name__=="__main__":
        test_img_path = "./class1/"                     #要测试的数据集
        train_imgs_path = "D:/闫溪芮/作业/mnist_data"   #训练数据集
        files = os.listdir(test_img_path)
        test_result = np.empty(len(files), dtype = "int")
        test_total_num = len(files)
        error = 0
        K = 5 # KNN参数
        train_data, train_label = load_data(train_imgs_path)
        
        for img_name in files:
                test_result = knn_test(test_img_path,img_name,K,train_data, train_label)
                print(test_result)
                for i in range(len(test_result)):
                        if test_result[0][i] != int(img_name[i]):
                                error += 1
        print(error/test_total_result)   #错误率
