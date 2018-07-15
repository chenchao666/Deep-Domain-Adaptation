from alexnet import *
from DataLoader import *
from Utils import *
os.environ['CUDA_VISIBLE_DEVICES']='1'



class Train():
    def __init__(self,class_num,batch_size,iters,learning_rate,keep_prob,param):
        self.ClassNum=class_num
        self.BatchSize=batch_size
        self.Iters=iters
        self.LearningRate=learning_rate
        self.KeepProb=keep_prob
        self.target_loss_param=param[0]
        self.domain_loss_param=param[1]
        Data=DataLoader("office31",source="Amazon",target="Webcam")
        self.SourceData,self.SourceLabel=Data.LoadSource()
        self.TargetData,self.TestData, self.TestLabel,self.LapMatrix=Data.LoadTarget()

        #######################################################################################
        self.source_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 227,227,3],name="source_image")
        self.source_label = tf.placeholder(tf.float32, shape=[self.BatchSize, self.ClassNum],name="source_label")
        self.target_image = tf.placeholder(tf.float32, shape=[self.BatchSize, 227, 227, 3],name="target_image")
        self.Training_flag = tf.placeholder(tf.bool, shape=None,name="Training_flag")
        self.L=tf.placeholder(tf.float32,shape=[self.BatchSize, self.BatchSize])
        self.Landmarks_weights=tf.placeholder(tf.float32,shape=self.BatchSize)

       ##################################################################################################

    def TrainNet(self):
        self.source_model=AlexNet(self.source_image, self.ClassNum, self.KeepProb,  "model/", reuse=False)
        self.target_model=AlexNet(self.target_image, self.ClassNum, self.KeepProb,  "model/", reuse=True)
        self.CalLoss()
        self.LandmarksSelecttion()
        self.source_prediction = tf.argmax(self.source_model.softmax_output, 1)
        self.target_prediction = tf.argmax(self.target_model.softmax_output, 1)
        var_all=tf.trainable_variables()
        var0=[var for var in var_all if 'fc7' in var.name]
        var1=[var for var in var_all if 'adapt' in var.name]
        var2=[var for var in var_all if 'fc8' in var.name]
        self.train_OP0 = tf.train.AdamOptimizer(learning_rate=0.0 * self.LearningRate).minimize(self.loss,var_list=var0)
        self.train_OP1 = tf.train.AdamOptimizer(learning_rate= self.LearningRate).minimize(self.loss,var_list=var1)
        self.train_OP2 = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(self.loss,var_list=var2)
        self.solver=tf.group(self.train_OP0,self.train_OP1,self.train_OP2)
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            self.source_model.weights_initial(sess)
            self.target_model.weights_initial(sess)
            true_num = 0.0
            for step in range(self.Iters):
                # self.SourceData,self.SourceLabel=shuffle(self.SourceData,self.SourceLabel)
                i= step % int(self.SourceData.shape[0]/self.BatchSize)
                j= step % int(self.TargetData.shape[0]/self.BatchSize)
                source_batch_x = self.SourceData[i * self.BatchSize: (i + 1) * self.BatchSize, :]
                source_batch_y = self.SourceLabel[i * self.BatchSize: (i + 1) * self.BatchSize, :]
                target_batch_x = self.TargetData[j * self.BatchSize: (j + 1) * self.BatchSize, :]
                LapMatrix=self.LapMatrix[j * self.BatchSize: (j + 1) * self.BatchSize,j * self.BatchSize: (j + 1) * self.BatchSize]
                total_loss, source_loss, target_loss, domain_loss, source_prediction,_= sess.run(
                    fetches=[self.loss, self.source_loss, self.target_loss, self.domain_loss, self.source_prediction, self.solver],
                    feed_dict={self.source_image: source_batch_x, self.source_label: source_batch_y,self.target_image: target_batch_x,
                               self.Training_flag: True, self.L:LapMatrix})


                true_label = argmax(source_batch_y, 1)
                true_num = true_num + sum(true_label == source_prediction)
                if step % 20 ==0:
                    print "Iters-{} ### TotalLoss={} ### SourceLoss={} ### TargetLoss={} ### domain_loss={}".format(step, total_loss, source_loss, target_loss, domain_loss)
                    train_accuracy = true_num / (20 * self.BatchSize)
                    true_num = 0.0
                    print " ########## train_accuracy={} ###########".format(train_accuracy)
                    self.Test(sess)






    def CalLoss(self):
        self.source_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.source_label, logits=self.source_model.softmax_output)
        self.source_loss = tf.reduce_mean(self.source_cross_entropy)
        self.CalTargetLoss(method="Manifold")
        self.CalDomainLoss(method="KMMD")
        self.loss = self.source_loss+ self.target_loss_param * self.target_loss+ self.domain_loss_param * self.domain_loss


    def CalDomainLoss(self,method):
        if method=="MMD":
            Xs=self.source_model.fc8
            Xt=self.target_model.fc8
            diff=tf.reduce_mean(Xs, 0, keep_dims=False) - tf.reduce_mean(Xt, 0, keep_dims=False)
            self.domain_loss=tf.reduce_sum(tf.multiply(diff,diff))


        elif method=="KMMD":
            Xs=self.source_model.adapt_layer
            Xt=self.target_model.adapt_layer
            self.domain_loss=tf.maximum(1e-4,KMMD(Xs,Xt))


        elif method=="CORAL":
            Xs = self.source_model.fc8
            Xt = self.target_model.fc8
            d=int(Xs.shape[1])
            Xms = Xs - tf.reduce_mean(Xs, 0, keep_dims=True)
            Xcs = tf.matmul(tf.transpose(Xms), Xms) / self.BatchSize
            Xmt = Xt - tf.reduce_mean(Xt, 0, keep_dims=True)
            Xct = tf.matmul(tf.transpose(Xmt), Xmt) / self.BatchSize
            self.domain_loss = tf.reduce_sum(tf.multiply((Xcs - Xct), (Xcs - Xct)))
            self.domain_loss=self.domain_loss / (4.0*d*d)



    def CalTargetLoss(self,method):
        if method=="Entropy":
            trg_softmax=self.target_model.softmax_output
            self.target_loss=-tf.reduce_mean(tf.reduce_sum(trg_softmax * tf.log(trg_softmax), axis=1))


        elif method=="Manifold":
            Xt = self.target_model.adapt_layer
            Mat0=tf.matmul(tf.matrix_transpose(Xt),self.L)
            Mat=tf.matmul(Mat0,Xt)
            self.target_loss=tf.trace(Mat)/(31*self.BatchSize**2)



    def LandmarksSelecttion(self):
        pass




    def Test(self,sess):
        true_num=0.0
        num=int(self.TargetData.shape[0]/self.BatchSize)
        total_num=num*self.BatchSize
        for i in range (num):
            # self.TestData, self.TestLabel = shuffle(self.TestData, self.TestLabel)
            k = i % int(self.TestData.shape[0] / self.BatchSize)
            target_batch_x = self.TestData[k * self.BatchSize: (k + 1) * self.BatchSize, :]
            target_batch_y= self.TestLabel[k * self.BatchSize: (k + 1) * self.BatchSize, :]
            prediction=sess.run(fetches=self.target_prediction, feed_dict={self.target_image:target_batch_x, self.Training_flag: False})
            true_label = argmax(target_batch_y, 1)
            true_num+=sum(true_label==prediction)
        accuracy=true_num / total_num
        print "###########  Test Accuracy={} ##########".format(accuracy)






def main():
    target_loss_param = 0.01 #MMD0.005 0.001, #KMMD 0
    domain_loss_param = 0.1 #MMD0.01   0.01  #KMMD 0.01
    param=[target_loss_param, domain_loss_param]
    Runer=Train(class_num=31,batch_size=256,iters=2000,learning_rate=0.0001,keep_prob=1.0,param=param)
    Runer.TrainNet()



if __name__=="__main__":
    main()