import importlib
import UtilsManager
from PIL import Image
import time


suanfamodule="FaceGenerate" #修改成你的算法文件夹名称
mymodulename="teamworker"+"."+suanfamodule+"."+"TaskWorkerImpl"
mymodule=importlib.import_module(mymodulename)
taskworkerimpl=mymodule.TaskWorkerImpler("test")
taskworkerimpl.setDevice("cuda:0")
initmodelparams=dict()
#根据需要换成你的模型初始化入参
# path='./weight/vgg19-dcbb9e9d.pth'
initmodelparams["face_generate_weight"]='./teamworker/FaceGenerate/face-generate-1024x1024.pt'

taskworkerimpl.initImpler(initmodelparams)
#根据需要换成你的检测图片
# filename1="D:/1pic/12.jpg"#推荐使用像素不小于100W
# orgdata1=Image.open(filename1)
# base64str1=UtilsManager.image_to_base64(orgdata1, 0)
# base64data=[]
# base64data.append(base64str1)

#time_start=time.time()

myresponse=taskworkerimpl.predictPic()
'''
if len(myresponse)<1:
   print("no hair")
else:   
   cvimg=UtilsManager.base64_to_image(myresponse[0]["outimg"],1)
   cv2.namedWindow("test",cv2.WINDOW_GUI_NORMAL)
   cv2.imshow("test",cvimg)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
'''
#time_end=time.time()
#print('totally cost：', time_end-time_start)

print(myresponse)

