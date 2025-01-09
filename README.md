
## **Stroke Prediction using SVM, Decision Tree, and KNN**

### **Description**:
This project addresses the critical task of predicting stroke occurrences using machine learning models. Accurate stroke prediction is essential for early medical intervention and saving lives. The dataset used contains patient health data, and three machine learning models were implemented:  
- **Support Vector Machine (SVM)**  
- **Decision Tree**  
- **K-Nearest Neighbors (KNN)**  

The project evaluates each model’s performance using metrics like accuracy, precision, recall, and F1-score.

---

### **Contents**:
1. **Introduction**  
   - Overview of the problem and dataset used.  
   - Brief explanation of the machine learning algorithms applied.  

2. **Dataset and Methodology**  
   - **Dataset Source**: Kaggle stroke prediction dataset.  
   - **Preprocessing Steps**:  
     - Handling missing values.  
     - Encoding categorical variables.  
     - Standardizing numerical features.  
     - Splitting data into training and test sets.  

3. **Model Implementation**  
   - **Support Vector Machine (SVM)**:  
     - Used a linear kernel.  
     - Applied feature scaling and encoding.  
   - **Decision Tree**:  
     - Two models: one without constraints and one with hyperparameter tuning to reduce overfitting.  
     - Visualized the decision tree structure.  
   - **K-Nearest Neighbors (KNN)**:  
     - Used `GridSearchCV` for optimal hyperparameter tuning.  
     - Evaluated using confusion matrix and accuracy metrics.

4. **Model Evaluation and Results**  
   - Accuracy scores for each model.  
   - Confusion matrices to visualize model performance.  
   - Discussion of which model performed best and why.

---

### **Results**:
- The **Decision Tree** model showed the best performance with a training accuracy of **96%** and a testing accuracy of **94%**.
- **Support Vector Machine (SVM)** and **K-Nearest Neighbors (KNN)** also performed well, but the Decision Tree achieved the best balance between accuracy and misclassification rates.

---

### **Technologies Used**:
- **Python**  
- **Jupyter Notebook**  
- Libraries:  
  - `pandas`, `numpy` for data manipulation.  
  - `matplotlib`, `seaborn` for visualization.  
  - `sklearn` for machine learning models.

---

### **How to Run**:
1. Clone this repository:
   ```bash
   git clone https://github.com/LujainAloufi/Stroke-Prediction-ML.git
   ```
2. Open the Jupyter Notebook in any environment (e.g., Jupyter Lab, Jupyter Notebook, or Google Colab).
3. Run the cells sequentially to reproduce the results.

---

### **Contributors**:
This project was developed as part of the **CS364 Machine Learning** course at **Al Imam Mohammad ibn Saud Islamic University**.  

---

### **References**:
1. [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)  
2. [GitHub Repository Reference](https://github.com/jordanjzhao/data-science-stroke-prediction)

---

### **License**:
This project is licensed under the MIT License - feel free to modify and use it for educational purposes.
