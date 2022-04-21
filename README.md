# SDE_Project
Objective : According to an IBM study, identifying and fixing defects after a system has been implemented is 5x more expensive than doing so during the design process. Software development may be a fast-paced, difficult, and perplexing process. Technical details may be kept in silos by certain developers, and technical details information may be lost over time. Structured processes such as CI/CD and testing frameworks are common industry practices; this project intends to build on those frameworks to help prevent the rise of recurring issues in software projects. 

Summary : A framework for applying neural net-based language models to software projects to support software developers in identifying errors and delivering better code. AWS was utilized to prepare the demonstration. The data was preprocessed locally because of the high cost of a GPU-based instance on AWS. It was transferred to an S3 bucket, where it inevitably found its way to an AWS GPU-based occurrence. On AWS, I utilized both the Nvidia Profound Learning AMI and the Nvidia Tensorflow docker holder. A g4dn.lxarge VM occurrence with a T4 GPU was utilized in this trial. For the VM occasion, 1TB of extra capacity space was moreover allocated. We used 15 Java projects from GitHub to create a public bug database form for this investigation.

Project Details : Predicting software defects is a crucial component of ensuring software quality. Deep Learning techniques can also be applied in this situation.To train the model with the data, we used Random Forest, Convolutional Neural Networks, SVM, Decision Tree, Naive Bayes, and Artificial Neural Networks.We blend the different results obtained from these strategies using Logistic Regression to obtain the final outcome.To conduct this comparison, we used a variety of open source datasets .Accuracy, F1 scores, and Areas under the Receiver Operating Characteristic curve are three extensively used measures for evaluation. Artificial Neural Networks were proven to outperform all existing dimensionality reduction strategies.  

Underlying tech used: Python, AWS, Code2Seq, Tensorflow
Common industry practices involve structured processes such as CI/CD, and testing framework this project aims to expand on the existing frameworks to help reduce the rise of recurring bugs in software projects.
