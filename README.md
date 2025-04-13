# 🌸 Iris Classification using k-Nearest Neighbors (k-NN)

This project demonstrates a simple machine learning pipeline using the **Iris dataset** and the **k-Nearest Neighbors classifier** from scikit-learn. The Iris dataset is a classic dataset in machine learning, consisting of 150 flower samples with 4 features each.

---

## 📊 Dataset

**Iris Dataset** from `sklearn.datasets.load_iris()` contains:

- **Features**:  
  - Sepal length (cm)  
  - Sepal width (cm)  
  - Petal length (cm)  
  - Petal width (cm)

- **Target classes**:
  - `0`: Setosa  
  - `1`: Versicolor  
  - `2`: Virginica

---

## 🧠 ML Model

We use **k-Nearest Neighbors (k-NN)** with `k=1` to classify Iris flowers based on feature values.

---

## 📦 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy scikit-learn
```

---

## 🚀 How It Works

1. **Load the dataset** using `load_iris()`.
2. **Explore** the structure, target classes, and feature details.
3. **Split** the dataset into training and test sets using `train_test_split()`.
4. **Train** the `KNeighborsClassifier` on the training data.
5. **Predict** on a new sample and test data.
6. **Evaluate** the model using accuracy scores.

---

## 🧪 Sample Output

```text
Target names: ['setosa' 'versicolor' 'virginica']
Feature names: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Type of data: <class 'numpy.ndarray'>
Shape of data: (150, 4)

X_train shape: (112, 4)
X_test shape: (38, 4)

X_new.shape: (1, 4)
Prediction: [0]
Predicted target name: ['setosa']

Test set score (np.mean): 0.97
Test set score (knn.score): 0.97
```

---

## 📂 File Structure

```
iris_knn_classifier.py   # Main script for training and testing k-NN on Iris dataset
```

---

## 📈 Accuracy

The model achieves around **97% accuracy** on the test set using just 1 neighbor (`k=1`), showing that even simple models can perform well on well-structured data.

---

## 📌 Notes

- You can tweak `n_neighbors` for better performance.
- Try visualizing data using `matplotlib` or `seaborn` for deeper insights.
- Consider using cross-validation for more robust evaluation.

---

## 📄 License

This project is open-source and free to use under the MIT License.
