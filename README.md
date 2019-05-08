# ID_vs_Spot_verification
依赖环境
    python 3.6
    Pytorch-gpu 1.0
    

人证比对系列文章
《Large-scale Bisample Learning on ID vs. Spot Face Recognition》
《DocFace: Matching ID Document Photos to Selfies》
《DocFace+: ID Document to Selfie Matching》

Prepare：

train datset： CASIA weface, MSRA, NJN_train(private, number 600000)
test dataset: LFW, NJN_test(private, 8000)

1、First stage

   Based on pretrained model trained on CASIA weface dataset, then fix the parameters of feature and update the parameters of classifier while training on MSRA dataset. Finally, we get the pretrained model by updating all parameters.
  
    Phase one, update parameters of classifiers.
        batch size = 384,
        epochs = 20
        augment: RandomHorizontalFlip
        lr = 0.01， lr decreased by factor 10  [31000,82000,120000]

    Phase two, update parameters of the entire network.
        batch size = 384
        epochs = 14
        augment: RandomHorizontalFlip
        lr:convolutional layers:0.01, classifier layers:0.1，lr decreased by factor 10 [51000,110000,150000] 
        
2、Second stage
    
   Use the model
	 



3、Third stage




Reference:
1. Large-scale Bisample Learning on ID vs. Spot Face Recognition
2. DocFace+: ID Document to Selfie Matching
3.《Improved Deep Metric Learning with Multi-class N-pair Loss Objective》


