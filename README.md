Course: CSC3034 – Computational Intelligence
Assignment 2: Artificial Neural Network (ANN) on Real-World Datasets
Title: FLIGHT FARE PREDICTION USING MACHINE LEARNING

Group Members:
[Kiroshan Ram], [Myles Lim Wenn Liang], [Ngoi Yi Ming], [Chow De Xian], [William Foong Mun Kit]

Overview

This project implements a neural network–based regression model to predict flight ticket prices from structured flight data. The goal is to evaluate whether a neural network can effectively model the non-linear relationships present in flight pricing and how it compares to classical machine learning baselines.

Prediction Task

Target variable: price

Objective: Predict final flight price based on flight characteristics such as airline, route, timing, number of stops, duration, and days before departure.

Models Implemented

Linear Regression (baseline)

Random Forest Regressor (tree-based baseline)

Neural Network (PyTorch)

Fully connected feedforward network

Trained with Adam optimizer and MSE loss

Evaluated using MAE and RMSE

Data Setup

Cleaned flight dataset (Clean_Dataset.xlsx)

Subsampled to 100,000 records

Data split:
Training: 70%
Validation: 15%
Test: 15%
Categorical features encoded using one-hot encoding
Numerical features standardised

Results

The neural network significantly outperforms linear regression, demonstrating effective non-linear modelling. However, the Random Forest achieves the lowest error overall, which is consistent with known strengths of tree-based models on tabular data.

How to Run:
python Flight_Price_Prediction_NN.py


The script handles preprocessing, training, evaluation, and result visualisation in a single pipeline.

Dependencies:
Python ≥ 3.10
NumPy
Pandas
Scikit-learn
PyTorch
Matplotlib
