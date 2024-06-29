# ReadMe
1、Data preparation: you should download the Radio3DMapSeer from the data set website "https://ieee-dataport.org/documents/dataset-pathloss-and-toa-radio-maps-localization-application", and find the code folder to extract the examples and put them in the correct location. 


2、Install the latest version of the pytorch deep learning framework and some necessary functions.


3、We provide two versions available, one is the baseline model, REM-NET, in which input features include buildings and transmitters carrying height information. The other is REM-NET+, which uses a radio propagation model. Compared to the former, REM-NET+ has an additional input feature, a free propagation map in the .png folder.


4、Note that when using different models, please correctly modify the number of features returned by the loader.py file, the input channel of the model, the initial number of features propagated forward, and the number of data loaded in the for loop.


5、If you want to retrain, just select a different model and run the file train.py


6、If you want to test, select a different model directly and run the file test.py (tensor layer).


7、If you want to view the radio environment map to evaluate the quality, select a different model directly and run the file save_image.py. The prediction results are saved in the image_result folder.


8、If you have any problem in the code, please email us. e-mail: chenqi\_7oou@stu.ynu.edu.cn;
