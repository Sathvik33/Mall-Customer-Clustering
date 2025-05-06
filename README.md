# Mall Customer Segmentation Using KMeans Clustering

This project demonstrates how to perform customer segmentation using the KMeans clustering algorithm on a dataset of mall customers. The goal is to group customers based on their income and spending score to uncover meaningful patterns in consumer behavior.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Steps Involved](#steps-involved)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)

## Introduction

In this project, we use customer data from a mall to segment customers into clusters using the KMeans algorithm. We perform the following tasks:

- Data preprocessing: Handling missing values, encoding categorical variables, and scaling numerical features.
- Visualization: Using PCA (Principal Component Analysis) to reduce the dataset to 2D and visualize the data.
- KMeans Clustering: Fitting the KMeans model to the data and predicting clusters.
- Elbow Method: Identifying the optimal number of clusters using the elbow method.
- Silhouette Score: Evaluating the clustering performance using silhouette scores.
- Cluster Visualization: Visualizing the clusters with color-coding on the PCA-reduced data.

## Dataset

The dataset used in this project is the **Mall Customers Dataset** with the following columns:

- **CustomerID**: Unique identifier for each customer.
- **Genre**: The gender of the customer.
- **Age**: The age of the customer.
- **Annual Income (k$)**: The annual income of the customer (in thousands of dollars).
- **Spending Score (1-100)**: A spending score assigned to each customer based on their spending behavior.

### Data Source:
This dataset can be found in the repository as `Mall_Customers.csv`.

## Installation

To run the project, you need to install the necessary dependencies. You can do so by running:

```bash
pip install -r requirements.txt
