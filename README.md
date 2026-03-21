# Feedforward-Neural-Network
# 🔢 Handwritten Digit Classification (MNIST) using Neural Network

This project builds a simple neural network using TensorFlow to classify handwritten digits from the MNIST dataset.

---

## 📌 Features
- Uses MNIST dataset (0–9 digits)
- Neural Network with Dense layers
- Trained using TensorFlow & Keras
- Achieves high accuracy on test data

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- NumPy

---

## 📊 Dataset

MNIST dataset contains:
- 60,000 training images
- 10,000 testing images
- Images of size 28×28 pixels

---

## 🧠 Model Architecture

- Input Layer: 784 neurons (flattened image)
- Hidden Layer: 128 neurons (ReLU)
- Output Layer: 10 neurons (Softmax)

---

## ▶️ How to Run

```bash
python mnist_model.py
```

---

## 📈 Output

### Training Output (Example)

```
Epoch 1/5
accuracy: 0.91

Epoch 5/5
accuracy: 0.98
```

---

### Test Accuracy

```
Test Accuracy: ~97% - 98%
```

---

## 📷 Output Screenshot
<img width="1195" height="346" alt="image" src="https://github.com/user-attachments/assets/52a97515-db09-483c-9fd1-bb6780cbe472" />


![Output](output.png)

---

## 📊 Performance
- High accuracy (~97–98%)
- Fast training using Adam optimizer

---

## ⚠️ Challenges Faced
- Data preprocessing (reshaping & normalization)
- Choosing correct number of epochs
- Avoiding overfitting

---

## 🚀 Future Improvements
- Add more hidden layers (Deep Learning)
- Use CNN for better accuracy
- Add dropout for regularization

---

## 📚 Conclusion
Neural networks can effectively classify handwritten digits with high accuracy using simple architectures.
