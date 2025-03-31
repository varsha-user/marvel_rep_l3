# marvel_rep_l3

## 1. DECISION TREES

Learnt about Decision Tress and how it works. Learnt how a tree consists of root node, parent node, decision node, leaf node, etc. Also learnt about pruning and splitting. Also learnt about ID3 algorithm and how it works. 
![WhatsApp Image 2025-01-03 at 13 50 32_3c86e7b6](https://github.com/user-attachments/assets/3e7e10d9-6751-43fa-a389-97232501f68e)
![WhatsApp Image 2025-01-03 at 13 50 56_6ddbc174](https://github.com/user-attachments/assets/58438e3f-45a9-4a94-8fde-51d257c4c8cf)
![WhatsApp Image 2025-01-03 at 13 51 13_8148b8e7](https://github.com/user-attachments/assets/60252e99-b28e-45d0-81fd-b3e31a22440f)

[Implementation of Decision tress on the Plays Tennis dataset](https://colab.research.google.com/drive/1o7HnzMapUYYitPdF0pvUFKtXqoDbDW_T?usp=sharing)

## 2. NAIVE BAYES

Learnt about what Naive Bayes is, how it works. Got a small recap of Bayes Theorem, from third sem, unit 5 maths 😭
Then, implemented it from scratch on a diabetes dataset. 
![WhatsApp Image 2025-03-02 at 22 55 02_dcf39bc0](https://github.com/user-attachments/assets/31f04c14-fb2b-443c-b798-7c9b779ad694)
![WhatsApp Image 2025-03-02 at 22 55 15_a6ee6c27](https://github.com/user-attachments/assets/52a9bdd3-0d06-44f7-8742-454c6aaf14c2)
[Implementation of Naive Bayes on the Diabetes dataset](https://colab.research.google.com/drive/1VHYBERsAy4Pb13ZZvaal8t0Pye8-V1P_?usp=sharing)

## 3. AND 4. ENSEMBLE TECHNIQUES AND RANDOM FORESTS, GRADIENT BOOST MACHINE(GBM) AND XGBOOST

Learnt about different ensemble techniques and understood how they help in improving model accuracies and help in better prediction and classification. 
I learnt about:
* Bagging
* Boosting
* Stacking
  
And I implemented it on the Titanic dataset. 

Under the implementation of boosting itself I have covered the implementation of both GBM and XGboost for the same Titanic dataset.
![image](https://github.com/user-attachments/assets/338b89ba-6090-44f1-9b71-121c23f95bf2)
![Screenshot 2025-03-23 195119](https://github.com/user-attachments/assets/d25b05b3-7143-4376-8a9d-ddc99c421b72)

[Implementation of Ensemble methods on the Titanic Dataset](https://colab.research.google.com/drive/1Q2qzXytnvE8rWrXQfxMWgixSbojk9jJ5?usp=sharing)

And Random forests,
[Implementation of Random Forests on the Iris Dataset](https://colab.research.google.com/drive/17G6KJZJ1i7NudinlD5Wfei9tgWkjSLc2?usp=sharing

## 5. HYPERPARAMTER TUNING

It is the process of predetermining the hyperparameters for an ML model. It is great because, it prevents overfitting and underfitting and makes the model simpler, more interpretable and more accurate ofc. But it is a bit computationally expensive and is has usage of high dimensional hyperparameter spaces. It involves different methods: 

* GridSearchCV: Every combination of hyperparameters is tried out and the one yielding the best results is chosen. This is computationally expensive, and exhaustive cuz of the several iterations.
  
* RandomizedSearchCV: It is better than GridSearchCV, since it does not need to go over and try every combination. It randomly starts with a combination of values of hyperparamets and repets this several ties and decides the best result one as the final combination to be used. This process is good since a greater range of values are covered, but we might not end up getting the best possible comination.
  
* Bayesian Optimization: It uses a probablistic model, based on the evaluation of the previous combinations results and uses something called as a surrogate function that is a probabilitis estimation of other objective functions like rmse/mse that are computationally expensive to calculate everytime. So, it iteratively finds this surrogate function's value and comes to the most accurate solution.
  
![image](https://github.com/user-attachments/assets/3f8f3753-5a95-4a00-8dd5-da6ac29db590)

I tried to implement this on the classic California Housing Dataset available on Google Colab, fo linear regression. I have included some text about my understanding of hyperparameter tuning for this particular ML model.
[Link for the same](https://colab.research.google.com/drive/1imXdMxPDuDBA9pfeKzuxo9S_KAAeuONG?usp=sharing)

## 6. K MEANS CLUSTERING

K-means clustering is an unsupervised learning ML algorithm, which forms clusters by initialising some random data points as centroids and assigning other data points to ach of these clusters. The 'k' here is the number of clusters. Once the clusters are formed, their mean value is assigned as the new centroid and this process is iterated till there is no more change in the new clusters are formed. Here, the clustering happens by finding out the Euclidian distance of the point from the centroid. It is computationally efficient and is ideal for use cases like: *1) Fraud detection *2) Shopping store customer grouping and optimisation *3) Document classification
I also briefly went through alternative options for unsupervised learning ML algos and ways to improve K-means clustering.

I used the MNIST dataset available in Google Colab and imported it for implementing K-means clustering on that MNIST Dataset. 
[Implementing K-means clustering on MNIST Dataset](https://colab.research.google.com/drive/1woO4XHX0A_tkKc1rFbX80vJOVsxYYNzc?usp=sharing)

## 7. ANOMALY DETECTION

It is the process of identifying a deviation from the expected or normal behavior or pattern in the data. We collect and analyse data and then analyse it. We use the pyod library in python to detect the utliers. 
It could be classified into:
![image](https://github.com/user-attachments/assets/dfcab714-f264-4048-89eb-f06d8e37524f)
The algorthms used majorly are:
1* **_Local Outlier Factor (LOF) algorithm_**: This algorithm uses the local density of points in a dataset to identify anomalies.
2* **_Isolation Forest_**: This uses decision trees to identify anomalies, by isolating points that are difficult to reach in the decision tree. 
3* **_One-class Support Vector Machines (SVMs)_**: This separates the majority of the data from the anomalies.
4* **_Elliptic Envelope_**: This assumes that the data is normally distributed and it fits an ellipse around the data, and identifies points that fall outside of the ellipse as anomalies.


## 8. Generative AI Task Using GAN(Generative Adversial Networks)

I learnt about how GAN works, and how it used to generate images or music or any kind of data. It is an unsupervised model, turned supervised, since the training dataset doesn't consist of any labels, whereas the discriminator ends up labelling the generator's output as real or fake(generated). So, a GAN model uses neural networks concept and consists of a Generator and a Discriminator. The generator(G) uses the input information and generates some information (G(z)), which is the fake data. Then the discriminator(D), uses the real data(x) and as well as G(z) and labels the input it got as fake or real. So, if discriminator guesses it right, then the generator models needs to improve itself and if the opposite happens, then it's the other way round. It utilises a cost function, where we sum the individual probabilities of the discriminator's labelling score on the real data and the fake data. 
#### The discriminator wants: *High D(x) *Low D(G(x))
![image](https://github.com/user-attachments/assets/8ed690bc-4ecd-4f53-8186-de0c31a471aa)

For this task, I have, tried to generate Pokemon Images using DC-GAN(Deep Convolutional GAN), which is a GAN architecture. 
[Generating Pokemon images](https://colab.research.google.com/drive/1xICH046ZbYIWJ7ebnD1h6XEdmio0Pv_b?usp=sharing)

## 9. PDF QUERY USING LANGCHAIN

So, Langchain is an NLP framework, which is used to get info form an uploaded document based on user query. issues and breaking them down into smaller sub-tasks. I used  HuggingFaceEmbeddings which uses sentence-transformers to generate embeddings for the text chunks, and RetrievalQA Chain which combines retrieval with the LLM to provide answers to your questions.
I implemented this and tested by uploading a resume and a research paper.
[The implementation.](https://colab.research.google.com/drive/1gxJojJxFlcS4CQgX-B3ZefP6XQ10A0dv?usp=sharing)

## 10. TABLE ANALYSIS USING PADDLEOCR

PaddleOCR is an OCR(Optical Character Recognition) framework, which detects and parses texts from uploaded the uploaded image extract that data from the image. 
It works by creating bounding boxes around the recognised text from the image and then using algorithms like feature analysis(where each character's feature like number of lines/curver, ect is compared with the training dataset characters) or by comparing a found character with every character of the training dataset. It also sometimes uses it's dictionary, like for example: if a word say dog is written in such a way that the ocr model cnat particularly make out if the middle letter is an a or o, then it could just check the dictionary out and conclude the word being dog. And realized how Google DocAI, Google lens, Amazon Textract might all be working using a similar logic. 
I implemented this and tested by uploading a bunch of bills and invoices.
[The implementation.](https://colab.research.google.com/drive/1BzXc0RJQNU4LZlgL8yz5SaOK3qQEZPub?usp=sharing)






