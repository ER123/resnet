#coding:utf-8
import cv2
import PIL
from PIL import Image
import tensorflow as tf 
image = cv2.imread("G:\\07_FRGC\\07202463\\07202463_202463468.jpg")
print("image:",image.shape())
string = image.tostring()

img = tf.decode_raw(string, tf.uint8)
img = tf.reshape(img, [150, 150, 3])  #reshape为128*128的3通道图片
img = tf.cast(img, tf.float32)# * (1. / 255) - 0.5 #在流中抛出img张量

with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init) 
	coord=tf.train.Coordinator()
	threads= tf.train.start_queue_runners(coord=coord)
	img = 
	example = sess.run(img)#在会话中取出image和label
	img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
	img.save("C:/Users/salaslyrin/Desktop/ResNet/_MYRESNET/a.jpg")#存下图片
	img.show()
	coord.request_stop()
	coord.join(threads)