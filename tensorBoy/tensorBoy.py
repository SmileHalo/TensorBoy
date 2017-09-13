from ImageHandle.imageHandler import ImageHandler as ih
import tensorflow as tf
import random

imageSize = 400

x = tf.placeholder("float", [None, imageSize],name='inputX') #定义占位符 X为输入结构是 宽度400 的输入流
W = tf.Variable(tf.zeros([imageSize,10]),name='weight')  #定义运算变量 W为400*10的矩阵
b = tf.Variable(tf.zeros([10]),'bias')
y = tf.nn.softmax(tf.matmul(x,W) + b,name='y') #定义激活函数 y=x*w+b
y_ = tf.placeholder("float", [None,10],'inputY_') #定义正确的输出值 占位符 类型为float 宽度为10

tf.summary.histogram("weights", W)
tf.summary.histogram("biases", b)
tf.summary.histogram('activations',y)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y),name='crossEntropy') #计算交叉熵的函数 这里用于计算 正确值与估计量的偏差
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #定义使用梯度下降算法，学习速率为0.1 目标为最小化交叉熵
tf.summary.scalar('lost',cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
tf.summary.scalar("accuracy", accuracy)

init = tf.initialize_all_variables() #定义初始化所有变量
Images = ih.get_trainning_images()

def tensorFlowTrainning():
    sess = tf.Session() #创建session 运行初始化函数
    summ = tf.summary.merge_all()
    writer = tf.summary.FileWriter(r'./tmp/mnist_logs', sess.graph)
    writer.add_graph(sess.graph)
    sess.run(init)
    for i in range(50):
        tempInputList = random.sample(Images,50)#读取数据集
        inX, inY = zip(*tempInputList) #数据集随机切片
        sess.run(train_step, feed_dict={x: inX, y_: inY})
        [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: inX, y_: inY})
        writer.add_summary(s, i)
        #print ('Trainning Progress:{now} of 1000 accuracy={acc}'.format(now=i+1,acc=sess.run(accuracy, feed_dict={x: inX, y_: inY})))
    print('Trainning is finished!')
    saver = tf.train.Saver()
    saver.save(sess, r".\net\sess.ckpt")
    print('Saved!')

def tensorFlowTest():
    session = tf.Session()
    session.run(init)
    saver = tf.train.Saver()
    saver.restore(session, r".\net\sess.ckpt") #读取
    get_result = y
    rawImages = ih.read_images(50,80)
    images = ih.Image_binaryzation(rawImages)
    images = crop_images(images)
    images = ih.resize_images(images)
    images = ih.getdata(images)
    images = ih.normalalizeImages(images)
    names=[]
    for eachImg in images :
        testedResult = session.run(get_result,feed_dict={x:[eachImg]})
        names.append(max(enumerate(testedResult[0].tolist()),key=lambda x:x[1])[0])
    index=0
    for img in rawImages:
        img.save(r'.\testOutput\{name}.png'.format(name=''.join(map(str,(names[index*4:(index+1)*4])))))
        index+=1
    print('test finished')
    
def crop_images(images):
     cropedImages = []
     for image in images:
         imgs = ih.crop_Image(ih,image=image)
         for img in imgs:
            cropedImages.append(img)
     return cropedImages

def gen_testDatas():
    images = ih.read_images(50,100)
    images = ih.Image_binaryzation(images)
    images = crop_images(images)
    images = ih.resize_images(images)
    for image in images:
        image.save(r'.\testOutput\{name}.png'.format(name=random.randint(0,99999)))
def main():
   tensorFlowTrainning()
   #tensorFlowTest()
    #gen_testDatas()
if __name__ == "__main__":
   main()
   input()