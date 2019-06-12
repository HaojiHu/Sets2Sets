# Sets2Sets

This is our implementation for the paper: 

Haoji Hu and Xiangnan He (2019). Sets2Sets: Learning from Sequential Sets with Neural Networks. In the 25th ACM SIGKDD Conference on KnowledgeDiscovery and Data Mining (KDD ’19), Anchorage, AK, USA

**Please cite our paper if you use our codes. Thanks!** 
Author: Haoji Hu

## Environment Settings
We use pytorch to implement our method. 
- Torch version:  '1.0.1'
- Python version: '3.6.8'

## A quick start to run the codes with Ta-Feng data set.

Training:
```
python Sets2Sets.py ./data/TaFang_history.csv ./data/TaFang_future.csv 1 2 1 
```
The above command will train our model based on 4 folds of the Ta-Feng data set. The left three numbers are the model id, the number of subsequent sets in the training instances, and the flag for mode. You can use any float number for the model id. Our example data can only support the number of subsequent sets no more than 3, which is the same as the results reported in our paper. The flag is set to 1 for training mode and 0 for test mode. The models learned from different epoches are saved under the folder './models/'. We use a default number of max epoches 20. We observe that our method usually achieves the best performance around 6-8 epoches accross different data sets. You can stop the training step if you get the model you need. 

Test:
```
python Sets2Sets.py ./data/TaFang_history.csv ./data/TaFang_future.csv 1 2 0 8 
```
The above command will test the learned model on the left 1 fold data. We just need to change the mode flag from 1 to 0 and specify  which model (here we use the model learned at epoch 8). The results will be printed out. 



### Preprocess the Dunnhumby data set

If we want to try our method on Dunnhumby data set, please visit [the offical website.](https://www.dunnhumby.com/careers/engineering/sourcefiles) View the 'Let's Get Sort-of-Real'. Download the the data for randomly selected sample of 50,000 customers. We provide our script to transfer their data into the formate our method needs. After extracting all the files in the zip file and put them under a folder (e.g. ./dunnhumby_50k/), please remember to delete a file named time.csv which is not needed in our method. Then, put our script and the folder './dunnhumby_50k/' at the same level. Run our script by following command:
```
python Dunnhumby_data_preprocessing.py ./dunnhumby_50k/ past.csv future.csv
```
The data will be generated under the current folder. You can just replace the two files (TaFang_history.csv and TaFang_future.csv) with these two generated files to apply our method on Dunnhumby data set as before. 


Last Update Date: June 12, 2019
