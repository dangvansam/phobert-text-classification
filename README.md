#transformers  
`
pip install transformers
`
#vncorenlp  
`
pip install vncorenlp
`
`
mkdir -p vncorenlp/models/wordsegmenter  
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar  
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab  
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr  
mv VnCoreNLP-1.1.1.jar vncorenlp/   
mv vi-vocab vncorenlp/models/wordsegmenter/  
mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/  
`
#Training
train model with tensorflow keras  
![alt text](train_classifier_keras.JPG)
train model with transformers(RobertaForSequenceClassification) pytorch  
![alt text](train_transformers_classifier_pytorch.JPG)