# StatML_Final_Project
ECE 6254 Final Project Spring 2019

## TODO
- [x] Define Project Task
- [x] Data EDA
- [x] Write Dataloader for PyTorch
- [ ] Possible Approaches:
  - [x] KNN Classifier (Advait)
  - [x] CNN Based
    - [x] Shufflenet (Advait)
    - [x] SimpleCNN (Advait)
    - [x] ResNet (Advait)
    - [x] AlexNet (Sanmathi)
  - [x] PCA +KNN
  - [x] Logistic Regression
  - [x] SVM
  - [x] Ensemble (Advait)
- [x] Implement Grid Search for Classification (can't use sklearn GridSearchCV :/ ) (Advait)
- [ ] Implement Visualization and Graphs for various classifiers (Advait, Sanmathi, Pranjali)
- [x] Implement face Detectors (Advait)
  - [x] State of the Art Detector (face_recogition library) 
- [x] Face Tracking with Unsupervised Learning (Advait)
  - [x] DBSCAN
  - [x] Mean Shift
- [ ] Complete Implementation (Advait)
 - [x] Video Input from Camera 
 - [x] Face Detection 
 - [x] Classification 
 - [x] Clustering
- [x] Proposal (Due March 27)
  - [x] Project Summary
  - [x] Project Decription
  - [x] List of Tasks
- [ ] Poster
## Useful Links:
* Dataset: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
* Project Idea: cs231n.stanford.edu/reports/2017/pdfs/221.pdf
* Face Detection using Clustering: http://openaccess.thecvf.com/content_iccv_2017/html/Jin_End-To-End_Face_Detection_ICCV_2017_paper.html
* 2 Stream CNN for Face Detection: cs231n.stanford.edu/reports/2017/pdfs/222.pdf
* Tutorial for OpenCV Face Detection: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
* Facial expression recognition using Open CV tutorial https://medium.com/@hinasharma19se/facial-expressions-recognition-b022318d842a
* DBSCAN Explained: https://medium.com/@elutins/dbscan-what-is-it-when-to-use-it-how-to-use-it-8bd506293818
* Clustering paper: openaccess.thecvf.com/.../Lin_Deep_Density_Clustering_CVPR_2018_paper.pdf
* t-SNE: http://mlexplained.com/2018/09/14/paper-dissected-visualizing-data-using-t-sne-explained/
* More t-SNE: https://distill.pub/2016/misread-tsne/
## References
### Proposal
* Paper defining 6 basic expressions: http://gunfreezone.net/wp-content/uploads/2008/12/Universal-Facial-Expressions-of-Emotions1-2015_03_12-21_10_38-UTC.pdf
* Review of FER: https://arxiv.org/pdf/1804.08348
* Robust model for FER: openaccess.thecvf.com/content.../Kuo_A_Compact_Deep_CVPR_2018_paper.pdf

## Main Dependencies
* face_recognition: https://pypi.org/project/face_recognition/
* PyTorch, TorchVision
* OpenCV
* Scikit Learn
