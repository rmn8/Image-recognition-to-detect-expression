# Image-recognition-to-detect-expression
Using CNN,the different expressions of faces are detected and classified



Software/Package Requirments
Python 2.7
Keras
Pandas
Numpy

For this project,I used Keros package with Tensorflow as background to construct a convolutional neural network with two hidden layers with dropout to prevent overfitting.The data is in the form of images of various actors obtained from IMFD database.The script gives a moderate accuracy of 60% .





                  HAPPINESS  NEUTRAL  SADNESS  SURPRISE  DISGUST  ANGER  FEAR       (True)

    HAPPINESS        664      318        7         0       18      0     0
    
    NEUTRAL           74     1151        7         0        8      0     0
    
    SADNESS           63      230      127         1        9      1     0
    
    SURPRISE          30      167        3        17       12      1     0
    
    DISGUST           34      130        5         0      241      0     0
    
    ANGER             75      223       10         0       24     35     0
    
    FEAR              12       41        5         0        5      0    16

    (predicted)

