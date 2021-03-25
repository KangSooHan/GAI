Video Captioning Based on Both Egocentric and Exocentric Views of Robot Vision for Human-Robot Interaction
---

Robot vision data can be thought of as first-person videos.
![alt tag](https://github.com/KangSooHan/GAI/images/gai.png)
There are three possible situations in one egocentric video.
  1. Global
    - The global explains the overall situation including detailed information such as place, light,weather
  2. Action
    - The action explains what the subject, i.e.I, is doing.
  3. Interaction
    - The interaction explains the interacting situation or behavior between the subject, i.e. me,and others

Global Action Interaction(GAI)

## Environment
  - Python3.6
  - Tensorflow 1.5.0
  
## How To Run
  1. Download UTEgocentric Dataset
     [Dataset](http://vision.cs.utexas.edu/projects/egocentric_data/UT_Egocentric_Dataset.html)
     [Preprocess Dataset] (https://drive.google.com/file/d/1IlX_WosLWfqRnIGIobI9gipZ8EGOJIUz/view?usp=sharing)
     
  2. Extract Video Features
     ```bash
     $ python extract_RGB_feature.py
     ```
     
  3. Train model
     ```bash
     $ python train.py
     ```
     
  4. Test model
     ```bash
     $ python test.py
     ```


## References
  S2VT model by [chenxinpeng](https://github.com/chenxinpeng/S2VT)
  
  Vgg model by [AlonDaks/tsa-kaggle](https://github.com/AlonDaks/tsa-kaggle)
  
  attention by [chenxinpeng] (https://github.com/AdrianHsu/S2VT-seq2seq-video-captioning-attention)
 
  Dataset by [UTEgocentric] (http://vision.cs.utexas.edu/projects/egocentric/storydriven.html)

