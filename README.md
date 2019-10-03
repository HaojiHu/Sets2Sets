# Sets2Sets

This is our implementation for the paper: 

Haoji Hu and Xiangnan He (2019). Sets2Sets: Learning from Sequential Sets with Neural Networks. In the 25th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD ’19), Anchorage, AK, USA

**Please cite our paper if you use our codes. Thanks!** 

Author: Haoji Hu

## Environment Settings
We use pytorch to implement our method. 
- Torch version:  '1.0.1'
- Python version: '3.6.8'

## A quick start to run the codes with Ta-Feng data set.

Training:
```
python Sets2Sets.py ./data/TaFang_history.csv ./data/TaFang_future.csv TaFang 2 1 
```
The above command will train our model based on 4 folds of the Ta-Feng data set. The three parameters in the command tail are the model name, the number of subsequent sets in the training instances, and the flag for mode. You can use any float number for the model id. Our example data can only support the number of subsequent sets no more than 3, which is the same as the results reported in our paper. Note that our method can handle variable length of subsequent sets due to the RNN. We fix this for experimental goal. The flag is set to 1 for training mode and 0 for test mode. The models learned from different epochs are saved under the folder './models/' (Our code will create this folder). We use a default number of max epochs 20 for demonstration. You can change this if you need more epochs. 

Test:
```
python Sets2Sets.py ./data/TaFang_history.csv ./data/TaFang_future.csv TaFang 2 0 
```
The above command will test the learned model on the left 1 fold data. We just need to change the mode flag from 1 to 0. The test  performance of the model giving best performance on the validation set will be printed out.



### Preprocess the Dunnhumby data set

If you want to try our method on Dunnhumby data set, please visit [the offical website.](https://www.dunnhumby.com/careers/engineering/sourcefiles) View the 'Let's Get Sort-of-Real'. Download the the data for randomly selected sample of 50,000 customers. We provide our script to transfer their data into the formate our method needs. After extracting all the files in the zip file and put them under a folder (e.g. ./dunnhumby_50k/), please remember to delete a file named time.csv which is not needed in our method. Then, put our script and the folder './dunnhumby_50k/' at the same level. Run our script by following command:
```
python Dunnhumby_data_preprocessing.py ./dunnhumby_50k/ past.csv future.csv
```
The data will be generated under the current folder. You can just replace the two files (TaFang_history.csv and TaFang_future.csv) with these two generated files to apply our method on Dunnhumby data set as before. As there are more customers in this data set, we can achieve good performance with 6 epochs.


Last Update Date: Oct. 1, 2019
