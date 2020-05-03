# Automatic Speech Recognition for Regional Indian Languages

## INTRODUCTION

The aim of this project is to implement automatic speech recognition algorithms using Hidden Markov Models (HMMs) for regional Indian languages. We have self-recorded Tamil digits, Telugu digits and words, and English continuous speech. We have also used external datasets for Hindi continuous speech and English digits. We have implemented HMM based systems using hmmlearn (Python library) and HTK (toolkit). We have also implemented a Deep Neural Network (DNN) based system to draw comparison and have presented our analysis. Following are the working implementations:
* [hmmlearn for Tamil and Telugu Digits, and Telugu Words Recognition](### hmmlearn-for-tamil-and-telugu-digits,-and-telugu-words-recognition)
* [HTK for Hindi Continuous Speech Recognition](### HTK-for-Hindi-Continuous-Speech-Recognition)
* [Deep Neural Network (DNN) for Tamil and Telugu Digits Recognition](### Deep-Neural-Network-(DNN)-for-Tamil-and-Telugu-Digits-Recognition)

## LITERATURE REVIEW
Automatic Speech Recognition (ASR) is a well researched field. The utilization of HMMs for ASR is studied well in [The Application of Hidden Markov Models in Speech Recognition](https://mi.eng.cam.ac.uk/~mjfg/mjfg_NOW.pdf). The paper presents the core architecture of a HMM-based Large Vocabulary Continuous Speech Recognition (LVCSR) system and then describes ways to achieve state-of-the-art performance. There is also a recent seminar report on [Hidden Markov Model and Speech Recognition](https://www.cse.iitb.ac.in/~nirav06/i/HMM_Report.pdf) which explains the Forward algorithm, the Viterbi algorithm and the Baum-Welch algorithm in the context of speech recognition and HMMs concisely.  

In the past few years, there has been significant work on developing speech recognition systems using HMMs for regional Indian languages. [Syllable Based Continuous Speech Recognition for Tamil Language](http://technicaljournalsonline.com/ijeat/VOL%20VII/IJAET%20VOL%20VII%20ISSUE%20I%20JANUARY%20MARCH%202016/20167101.pdf) uses MFCC feature vectors and an acoustic HMM model to develop a recognition system for Tamil. We have used to similar methodology to develop a recognition system for Telugu words using [HTK](http://htk.eng.cam.ac.uk/ftp/software/htkbook-3.5.alpha-1.pdf), a toolkit for building HMMs. [Grapheme Gaussian Model and Prosodic Syllable Based Tamil Speech Recognition System](https://www.researchgate.net/publication/269329025_Grapheme_Gaussian_model_and_prosodic_syllable_based_Tamil_speech_recognition_system) built upon this system and produced an accuracy of 77% on a dataset of 20 Tamil words, with 2 speakers and 2 utterances each. However, the implementation of this was beyond the scope of this project. [HTK Based Speech Recognition Systems for Indian Regional languages: A Review](https://www.irjet.net/archives/V3/i7/IRJET-V3I7115.pdf) presents well the summaries and best obtained accuracies of HTK based speech recognition systems developed for 13 regional languages including Tamil, Telugu, Hindi and English.

[Automatic Speech Recognition Systems for Regional Languages in India](https://www.researchgate.net/publication/338790124_Automatic_Speech_Recognition_Systems_for_Regional_Languages_in_India) argues that Deep Neural Networks (DNNs) must be more efficent and accurate for speech recognition. 



## DATASETS

### Tamil Digits

The Tamil digits dataset consists of audio files recorded in ‘.wav’ format. Each file contains the utterance of one Tamil digit from 0-9. The length of each file is approximately 1 second. A total of 230-250 samples are present with each digit having around 13-15 samples. The dataset can be accessed [here](https://drive.google.com/file/d/1S2JTQHnG5QLgcG8X3DIIopU5Z0oeglpM/view?usp=sharing). 

The digit-label-utterance mapping is given in the following table.

Digit | Label | Utterance
----- | ----- | ---------
Zero | 0 | Poojyam
One | 1 | Onnu
Two | 2 | Rendu
Three | 3 | Munnu
Four | 4 | Naalu
Five | 5 | Anju
Six | 6 | Aaru
Seven | 7 | Yezhu
Eight| 8 | Yettu
Nine | 9 | Ombodu

### Telugu Digits

The Telugu digits dataset consists of audio files recorded in ‘.wav’ format. Each file contains the utterance of one Telugu digit from 1-10. A total of ~60 samples are present with each digit having around 6-7 samples. The dataset can be accessed [here](). 

The digit-label-utterance mapping is given in the following table.

Digit | Label | Utterance
----- | ----- | ---------
One | 1 | Okati
Two | 2 | Rendu
Three | 3 | Mudu
Four | 4 | Nalugu
Five | 5 | Aidu
Six | 6 | Aaru
Seven | 7 | Edu
Eight| 8 | Enimidi
Nine | 9 | Tommidi
Ten | 10 | Padi

### Telugu Words

The Telugu words dataset consists of audio files recorded in ‘.wav’ format. Each file contains the utterance of one Telugu word. A total of 80 samples are present with each word having 4 samples. The dataset can be accessed [here](). 

The Telugu-English word mapping is given in the following table.

Word | Meaning 
----- | ----- 
abbayi | boy
amma | mother
ammayi | girl
andarum | all
batuku | everyone
bojanum | meal
chudama | check
cinnema | movie
dhairyam | courage
kalisi | together
kannu | eye
kodatanu | beats
konchum | slightly
manum | 
meeru | you
nanna | father
nenu | I
pinni | aunt
sonthum | 
yevaru | who

### English (Indian Accent) Continuous Speech

#### Externally Obtained Datasets:
##### Hindi Continuous Speech
  
The dataset consists of 150 sentences in Hindi with 7 different speakers for each. It can be accessed [here](https://drive.google.com/open?id=1DHZkTDgRsG3X9YRAzsWXBdYUXj3sE2DY).
    
##### English Digits
    
The dataset can be accessed [here](https://github.com/Ralireza/spoken-digit-recognition/tree/master/spoken_digit).


## IMPLEMENTATIONS

### HMMLEARN IMPLEMENTATION FOR TAMIL AND TELUGU DIGITS RECOGNITION

### Dependencies

* Python (version 2.7. *)
* hmmlearn
* python_speech_features

### HTK IMPLEMENTATION FOR HINDI CONTINUOUS SPEECH AND TELUGU WORDS RECOGNITION
 
### HTK Installation (Linux)

* Follow the installation steps mentioned here: https://github.com/conbitin/htk3.5-install 

### Speech Recognition

* Download the Hindi dataset mentioned above.
* Clone this repository: https://github.com/KunalDhawan/ASR-System-for-Hindi-Language/tree/master/HTK

### DEEP NEURAL NETWORK FOR TAMIL AND TELUGU DIGITS RECOGNITION

### Dependencies
* Numpy 
* Pandas 
* Librosa
* Pytorch 
* Sklearn

In order to compare the performance of the HMM model on the tamil digits dataset, we train a modern deep learning architecture for the same dataset and observe the performance and compare it with the previous model. 

The deep learning model that has been chosen is a Long Short-Term Memory (LSTM) model. LSTM are a special member of the Recurrent Neural Network (RNN) family and have the ability to model the data based on previous data. A non-recurrent Neural Network does not have any memory whereas an RNN has a limited memory and they tend to perform badly on data that has long term temporal dependency on the previous data. LSTM also has the ability to decide how much information to use in its memory as they have input gates, forget gates and output gates. 

The LSTM architecture and the other hyper-parameters and functions used are given below:

* Architecture

LSTM(
  (rnn): LSTM(input = 81, hidden_neurons = 10, num_layers=2, dropout=0.1)
  (fc): Sequential(Linear(in_features=10, out_features=10, bias=True))
  (output) : Softmax(input = 10 , output = 1)
)
* Hyper-Parameters
  * Learning Rate = 0.01
  * Loss function used = MSE (Mean Squared Error) Loss
  * Optimizer used = Adam Optimizer

### Training Results

We trained the model on 220 samples by shuffling the samples. The model was trained for 100 epochs and used batch gradient descent on a batch size of 20 samples. The results are as follows :

Total number of test samples = 220

Correct predictions = 192

Accuracy = 87.27272727272727%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_1.png)

#### Loss Plot for Training:

![Loss Plot for Training](/images/loss_plot.png)

The trained model along with the weights can be accessed [here](https://drive.google.com/file/d/12TkL3GNNogYDo4VXYAvGymli3S1RFHMf/view?usp=sharing).

### Testing Results

After training the model, we test on a few unseen samples to see the performance of the model. 

Total number of test samples = 20
 
Correct predictions = 13
 
Accuracy = 65.0%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_2.png)

## REFERENCES

