**Introduction**

The report evaluates several modeling approaches for classifying fashion product images to support marketing-related information gathering. The project aims to reduce sample size and processing time while preserving high classification accuracy, thus optimizing costly and time-intensive models. Furthermore, the findings are expected to offer insights and recommendations for enhancing the company’s data infrastructure, tailored to the unique characteristics of the data in use.

The techniques employed in this development was Neural Networks (h20 library).

The models were developed using a training data set **“MNIST-fashion training set-49.csv” of 60000 observations**, which was randomly split into subsets of various sizes. Additionally, a testing data set **“MNIST-fashion testing set-49.csv”, containing 10000 observations** was used to evaluate model performance.

To balance these competing objectives, an overall scoring function is introduced to assess the quality of a classification:

**Points=0.15×𝐴+0.1×𝐵+0.75×𝐶**

where:
.**A is the proportion of the training data used in the model**. For example, if 30,000 out of 60,000 rows are used, then 𝐴=30,00060,000=0.5
.**B is the running time penalty**, defined as: 𝐵=min(1,𝑋60) where 𝑋 is the running time of the selected algorithm in seconds. This time includes both the model training on the training data and generating predictions on the testing data. If the running time is at least 1 minute (60 seconds), then 𝐵=1 , resulting in the full running time penalty.
.**C is the proportion of incorrect predictions** on the testing set. For example, if 1,000 out of 10,000 rows are misclassified, then 𝐶=1,00010,000=0.1

The objective is to develop a classification method that minimizes the Points score. Ideally, the algorithm will use the least amount of data, execute as quickly as possible, and accurately classify as many items in the testing set as possible.
