# Automatic Speech Recognition for Regional Indian Languages

## INTRODUCTION

The aim of this project is to implement automatic speech recognition algorithms using Hidden Markov Models (HMMs) for regional Indian languages. We have self-recorded Tamil digits, Telugu digits and words, and English continuous speech. We have also used external datasets for Hindi continuous speech and English digits. We have implemented HMM based systems using hmmlearn (Python library) and HTK (toolkit). We have also implemented a Deep Neural Network (DNN) based system to draw comparison and have presented our analysis. Following is the list of implementations:
* [hmmlearn for Tamil, Telugu and English Digits, and Telugu Words Recognition](#hmmlearn-for-tamil-telugu-and-english-digits-and-telugu-words-recognition)
* [HTK for Hindi Continuous Speech and Telugu Words Recognition](#htk-for-hindi-continuous-speech-and-telugu-words-recognition)
* [Deep Neural Network (DNN) for Tamil and Telugu Digits Recognition](#deep-neural-network-dnn-for-tamil-and-telugu-digits-recognition)

## LITERATURE REVIEW
Automatic Speech Recognition (ASR) is a well researched field. The utilization of HMMs for ASR is studied well in [The Application of Hidden Markov Models in Speech Recognition](/References/The_Application_of_Hidden_Markov_Models_in_Speech_Recognition.pdf). The paper presents the core architecture of a HMM-based Large Vocabulary Continuous Speech Recognition (LVCSR) system and then describes ways to achieve state-of-the-art performance. There is also a recent seminar report on [Hidden Markov Model and Speech Recognition](/References/HMM_Report.pdf) which explains the Forward algorithm, the Viterbi algorithm and the Baum-Welch algorithm in the context of speech recognition and HMMs concisely.  

In the past few years, there has been significant work on developing speech recognition systems using HMMs for regional Indian languages. [Syllable Based Continuous Speech Recognition for Tamil Language](/References/Syllable_Based_Recognition_for_Tamil.pdf) uses MFCC feature vectors and an acoustic HMM model to develop a recognition system for Tamil. We have used a similar methodology to develop a recognition system for Telugu words using [HTK](/References/htkbook.pdf), a toolkit for building HMMs. [Grapheme Gaussian Model and Prosodic Syllable Based Tamil Speech Recognition System](/References/Grapheme_Gaussian_Model_and_Prosodic_Syllable_Based_Tamil.pdf) builds upon this system and produces an accuracy of 77% on a dataset of 20 Tamil words, with 2 speakers and 2 utterances each. However, the implementation of this was beyond the scope of this project. [HTK Based Speech Recognition Systems for Indian Regional languages: A Review](/References/HTK_Based_Speech_Recognition_Systems_for_Indian_Regional_languages.pdf) presents well the summaries and best obtained accuracies of HTK based speech recognition systems developed for 13 regional languages including Tamil, Telugu, Hindi and English.

[Automatic Speech Recognition Systems for Regional Languages in India](/References/ASRs_for_regional_languages.pdf) argues that Deep Neural Networks (DNNs) must be more efficent and accurate for speech recognition. 

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
:-----: | :-----: | :---------:
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
:-----: | :-----: 
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
manum | us
meeru | you
nanna | father
nenu | I
pinni | aunt
sonthum | ourselves
yevaru | who

### English (Indian Accent) Continuous Speech

Three speakers recorded data for English continuous speech in an Indian accent.

[Speaker 1](https://drive.google.com/drive/u/1/folders/1yXDQKNBNo4SL6NR2xdRu7Ac2lDRrrIyt) (Male, 20yrs):
File | Duration
:-----: | :-----:
rec1.wav | 6:49
rec2.wav | 7:04
rec3.wav | 13:34
rec4.wav | 4:07
rec5.wav | 7:53

[Speaker 2](https://drive.google.com/drive/u/1/folders/1E1XdjohsZ5X3L9MTh_ombaE_0Gq4T09A) (Male, 21yrs):
File | Duration
:-----: | :-----:
rec1.wav | 18:14
rec2.wav | 30:59

[Speaker 3](https://drive.google.com/drive/u/1/folders/1fDZzaBPO0TB-cTZO9VsEUdWU4JSAblAY) (Male, 21 yrs):
File | Duration
:-----: | :-----:
rec1.wav | 8:36
rec2.wav | 10:29
rec3.wav | 11:25
rec4.wav | 9:54

### Externally Obtained Datasets:
* **Hindi Continuous Speech:**

  The dataset consists of 150 sentences in Hindi with 7 different speakers for each. It can be accessed [here](https://drive.google.com/open?id=1DHZkTDgRsG3X9YRAzsWXBdYUXj3sE2DY).
    
* **English Digits:**

  The dataset can be accessed [here](https://github.com/Ralireza/spoken-digit-recognition/tree/master/spoken_digit).

## IMPLEMENTATIONS

### HMMLEARN FOR TAMIL, TELUGU AND ENGLISH DIGITS, AND TELUGU WORDS RECOGNITION

### Dependencies

* Python (version 2.7. *)
* hmmlearn
* python_speech_features

### Tamil Digits

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

Training Results:

![Fig](/images/tam_train.png)
**Accuracy:** 61.48%

Testing Results:

![Fig](/images/tam_test_1.png)
**Accuracy:** 60.97%

Summary:

![Fig](/images/tam_test_2.png)

### Telugu Digits

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

Training Results:

![Fig](/images/telugu_dig_train.png)
**Accuracy:** 50.24%

Testing Results:

![Fig](/images/telugu_dig_test.png)
**Accuracy:** 58.06%

### Telugu Words

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

Training Results:

![Fig](/images/telugu_words_train.png)
**Accuracy:** 65%

Testing Results:

![Fig](/images/telugu_words_test_1.png)
**Accuracy:** 60%

Summary:
![Fig](/images/telugu_words_test_2.png)

### English Digits

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

* **Entire Dataset**

  Training Results:

  ![Fig](/images/eng_f_train.png)
  **Accuracy:** 96.75%

  Testing Results:

  ![Fig](/images/eng_f_test.png)
  **Accuracy:** 94%
  
* **Limited Dataset**

  We used 20% of the original test data and took 15 samples per digit.

  Training Results:

  ![Fig](/images/eng_l_train.png)
  **Accuracy:** 60%

  Testing Results:

  ![Fig](/images/eng_l_test.png)
  **Accuracy:** 60% 

### HTK FOR HINDI CONTINUOUS SPEECH AND TELUGU WORDS RECOGNITION
 
### HTK Installation (Linux)

* Follow the installation steps mentioned here: https://github.com/conbitin/htk3.5-install 

### Hindi Continuous Speech

* Create a fork of this repository: https://github.com/KunalDhawan/ASR-System-for-Hindi-Language/tree/master/HTK and clone it.
* Download the Hindi dataset mentioned [above](#externally-obtained-datasets).
* Store the downloaded 'data' directory in the HTK folder of the cloned repository.
* Follow the steps mentioned [here](https://kunal-dhawan.weebly.com/asr-system-in-hindi-language-from-scratch.html).

![Fig](/images/htk-hindi.jpeg)

### Telugu Words

**Note**: Our forked repository can be found [here](https://github.com/verma-bhavya/telugu_asr_htk/tree/master/HTK).

* Upload the data in ./data dir in the corresponding train and test directories. 
* Prepare a transliteration file and a lexicon file (phone level) (hindiSentences150.txt and lexicon.txt respectively in the original repository) for all the words present in the speech samples and put in ./doc and ./lm respectively.
* Now go to scripts_ph_pl_py folder and edit the HTK_home variable in master.sh with the absolute path of your HTK dir. 
* Also give read_write permissions to all the files present here -> chmod a+rx *.sh *.pl *.py.
* Now cd into the parent directory and run the following commands to:
  1. Generate env var and mfcc features
  2. Write the transcription  
  3. Initialize PDF of each phone model
  4. Fit the data
  5. Evaluate the output

>> scripts_sh_pl_py/master.sh HCOPY   
>> scripts_sh_pl_py/master.sh LEXICON   
>> scripts_sh_pl_py/master.sh HCOMPV  
>> scripts_sh_pl_py/master.sh HEREST          
>> scripts_sh_pl_py/master.sh ALIGN  
>> scripts_sh_pl_py/master.sh HVITE_MONO 

![Fig](/images/htk-telugu.jpeg)

### DEEP NEURAL NETWORK (DNN) FOR TAMIL AND TELUGU DIGITS RECOGNITION

### Dependencies
* Numpy 
* Pandas 
* Librosa
* Pytorch 
* Sklearn

### Tamil Digits

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

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

**Training Results**

We trained the model on 220 samples by shuffling the samples. The model was trained for 100 epochs and used batch gradient descent on a batch size of 20 samples. The results are as follows :

* Total number of test samples = 220
* Correct predictions = 192
* Accuracy = 87.27272727272727%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_1.png)

#### Loss Plot for Training:

![Loss Plot for Training](/images/loss_plot.png)

**Testing Results**

After training the model, we test on a few unseen samples to see the performance of the model. 

* Total number of test samples = 20
* Correct predictions = 13
* Accuracy = 65.0%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_2.png)

### Telugu Digits

**Note:** To reproduce the results, refer [this](#comparison) section for the script and weight file.

In order to compare the performance of the HMM model on the telugu digits dataset, we train a modern deep learning architecture for the same dataset and observe the performance and compare it with the previous model. 

The deep learning model that has been chosen is a Long Short-Term Memory (LSTM) model. LSTM are a special member of the Recurrent Neural Network (RNN) family and have the ability to model the data based on previous data. A non-recurrent Neural Network does not have any memory whereas an RNN has a limited memory and they tend to perform badly on data that has long term temporal dependency on the previous data. LSTM also has the ability to decide how much information to use in its memory as they have input gates, forget gates and output gates. 

The LSTM architecture and the other hyper-parameters and functions used are given below:

* Architecture

LSTM(
  (rnn): LSTM(81, 10, num_layers=2, dropout=0.1)
  (fc): Sequential(
    (0): Linear(in_features=10, out_features=10, bias=True))
)

* Hyper-Parameters

Learning Rate = 0.01
Loss function used = MSE (Mean Squared Error) Loss
Optimizer used = Adam Optimizer

**Training Results**

We trained the model on 50 samples by shuffling the samples. The model was trained for 20 epochs and used batch gradient descent on a batch size of 5 samples. The results are as follows :

* Total number of test samples = 50
* Correct predictions = 45
* Accuracy = 90.0%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_1_telugu.png)

#### Loss Plot for Training:

![Loss Plot for Training](/images/loss_plot_telugu.png)

**Testing Results**

After training the model, we test on a few unseen samples to see the performance of the model. 

* Total number of test samples = 16
* Correct predictions = 10
* Accuracy = 62.5%

#### Confusion Matrix:

![Confusion Matrix](/images/confusion_matrix_2_telugu.png)

## COMPARISON

DATASET | IMPLEMENTATION | TRAINING ACCURACY | TESTING ACCURACY | SCRIPT | WEIGHT FILE
:-------------: | :---------: | :---: | :---: | :--------------: | :-------------------------:
Tamil Digits | hmmlearn | 61.48% | 60.97% | [hmm_digits.py](/hmmlearn/hmm_digits.py)| [tamil_digits.pkl](/hmmlearn/tamil_digits.pkl)
Tamil Digits | DNN | 87.27% | 65% | [DNN_Tamil.ipynb](/DNN/DNN_Tamil.ipynb) | [link](https://drive.google.com/file/d/12TkL3GNNogYDo4VXYAvGymli3S1RFHMf/view?usp=sharing)
Telugu Digits | hmmlearn | 50.24% | 58.06% | [hmm_digits.py](/hmmlearn/hmm_digits.py) | [telugu_digits.pkl](/hmmlearn/telugu_digits.pkl)
Telugu Digits | DNN | 90% | 62.5% | [DNN_TELUGU.ipynb](/DNN/DNN_TELUGU.ipynb) | [link](https://drive.google.com/file/d/1-3g4bKk5_QeMQGhrKUyPxn7gc35z0fGS/view?usp=sharing)
English Digits (entire dataset) | hmmlearn | 96.75% | 94% | [hmm_digits.py](/hmmlearn/hmm_digits.py) | [english_digits.pkl](/hmmlearn/english_digits.pkl)
English Digits (limited dataset)| hmmlearn | 60% | 60% | [hmm_digits.py](/hmmlearn/hmm_digits.py) | [english_digits_limited.pkl](/hmmlearn/english_digits_limited.pkl)
Telugu Words | hmmlearn | 65% | 60% | [hmm_words.py](/hmmlearn/hmm_words.py) | [telugu_words.pkl](/hmmlearn/telugu_words.pkl)
Telugu Words | HTK:heavy_exclamation_mark: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign: | :heavy_minus_sign:
Hindi Continuous Speech| HTK | :heavy_minus_sign: | 67.35% | :heavy_minus_sign: | :heavy_minus_sign:

## CONCLUSION AND FUTURE WORK

* DNN outperforms the training accuracy of hmmlearn by a large margin.
* DNN outperforms the testing accuracy of hmmlearn by a smaller margin.
* Training and testing on the entire English dataset gave ~95% accuracy as opposed to 60% for limited dataset. Thus, more data will significantly improve the accuracies on our self-recorded regional language datasets. 
* The HTK implementation works successfully for continuous speech data. The next step would be to try regional language datasets for continuous speech.

### CONTRIBUTORS
* [Sharmistha Gupta](mailto:sharmistha16193@iiitd.ac.in)
* [Pavan Garimella](mailto:pavan17172@iiitd.ac.in)
* [Nakul Ramanathan](mailto:nakul16168@iiitd.ac.in)
* [Jai Mahajan](mailto:jai16154@iiitd.ac.in)
* [Bhavya Verma](mailto:bhavya17142@iiitd.ac.in)
* [Aditya Shidhaye](mailto:aditya17128@iiitd.ac.in)
* [Aditya Aggarwal](mailto:aditya16127@iiitd.ac.in)
