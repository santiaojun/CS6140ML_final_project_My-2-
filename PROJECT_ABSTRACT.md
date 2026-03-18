# Project Abstract: Recommender Systems Comparison

## Project Title
Comparing Classical and Neural Models for Recommender Systems on the Amazon Dataset

## Project Goal
The goal of this project is to compare the performance of classical recommender systems with modern neural recommendation approaches. Specifically, we aim to develop a Two-Tower model and conduct a comparative analysis against traditional model baselines (such as Matrix Factorization) using the Amazon Review Data (2018).

## Project Motivation
Our primary motivation is to bridge the gap between theoretical machine learning and real-world industrial applications. By using real-world e-commerce data, we aim to demystify the "black box" behind the algorithms we interact with daily on platforms like YouTube and Amazon.

Technically, we seek to evaluate the performance leap from classical Collaborative Filtering to modern deep learning. While Matrix Factorization is effective, it struggles with the high dimensionality and sparsity of massive datasets. The Two-Tower model is motivated by its ability to decouple User and Item embeddings, enabling high-performance Approximate Nearest Neighbor (ANN) searches—a requirement for modern, low-latency industry systems.

## Related Work
* **Matrix Factorization (MF):** A classical CF technique representing users and items as low-dimensional latent vectors.
* **Deep Neural Networks for YouTube Recommendations (Covington et al., 2016):** A landmark two-stage architecture (candidate generation and ranking).
* **Neural Collaborative Filtering (NCF) (He et al., 2017):** A framework that replaces the linear inner product of MF with a Multi-Layer Perceptron (MLP) to capture non-linear interactions.

## Scope of Work
1. **Data Preprocessing:** Cleaning and sampling the Amazon Review Data (2018).
2. **Baseline Implementation:** Building a Matrix Factorization model.
3. **Neural Model Implementation:** Developing a Two-Tower architecture for retrieval.
4. **Evaluation:** Comparing models based on metrics such as Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG).

## Split of Work
* **Minjia Fang:** Data pipeline construction, exploratory data analysis (EDA), and baseline MF model implementation.
* **Yunhong Huang:** Neural Two-Tower model architecture design and training loop implementation.
* **Yiyi Yu:** Evaluation framework development, hyperparameter tuning, and comparative performance analysis.
