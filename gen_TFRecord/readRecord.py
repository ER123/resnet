#coding:utf-8
import tensorflow as tf 
import base64
from PIL import Image

def read_and_decode(filename): # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.float32),
                                           'image' : tf.FixedLenFeature([], tf.string),
                                       })#将image数据和label取出来
 
    img = tf.decode_raw(features['image'], tf.uint8)
    img = tf.reshape(img, [96, 96, 3])  #reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量
    label = tf.cast(features['label'], tf.float32) #在流中抛出label张量
    print("label:",label)
    return img, label

filename_queue = tf.train.string_input_producer(["G:/train.tfrecord"]) #读入流中
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)   #返回文件名和文件
features = tf.parse_single_example(serialized_example,
                                   features={
                                       'image' : tf.FixedLenFeature([], tf.string),
                                       'label': tf.FixedLenFeature([256], tf.float32),
                                   })  #取出包含image和label的feature对象
print("features:",features)

f = open("G:/ready_for_tfrecord/labels.txt",'w')

image = tf.decode_raw(features['image'], tf.uint8)
image = tf.reshape(image, [96, 96, 3])
label = tf.cast(features['label'], tf.float32)
print("image:",image)
print("label:",label)
with tf.Session() as sess: #开始一个会话
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord=tf.train.Coordinator()
    threads= tf.train.start_queue_runners(coord=coord)
    for i in range(0,10):
        example, l = sess.run([image,label])#在会话中取出image和label
        img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
        img.save("G:/ready_for_tfrecord/"+str(i)+'.jpg')#存下图片
        #print(l.dtype)
        for num in l:
          #print("num: ",num)
          f.write(str(num)+"\n")
        f.write("\n")
        #img.show()
        #print("l:",l)
    coord.request_stop()
    coord.join(threads)