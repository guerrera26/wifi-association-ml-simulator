# Machine Learning Wi-Fi Association Simulator

Python simulation modeling wireless client association across multiple access points.  
The project evaluates whether a machine learning model can outperform traditional RSSI-based association policies.

## Overview

Wireless clients often remain connected to suboptimal access points even when a better one is available.  
This phenomenon, known as the **sticky client problem**, can reduce network efficiency.

This project simulates a wireless environment and trains a machine learning model to predict optimal client-to-access-point associations.

## Features

- Simulates multiple wireless access points and client devices
- Generates synthetic training data representing wireless conditions
- Trains a Random Forest classifier to predict optimal AP selection
- Compares ML association against traditional RSSI-based policies
- Evaluates throughput and sticky client behavior across simulations

## Results

- Generated **1,000,000 synthetic training samples**
- Evaluated **1000 simulated network scenarios**
- Eliminated sticky client events entirely (baseline average: 2.49 per simulation)
- Maintained comparable network throughput performance

## Technologies

Python  
NumPy  
pandas  
scikit-learn  

## How to Run

```bash
python simulator.py
