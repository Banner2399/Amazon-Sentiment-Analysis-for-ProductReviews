# 📌 Sentiment Analysis for Product Reviews 🛍️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat&logo=streamlit)
![SVM](https://img.shields.io/badge/Model-SVM-green?style=flat)
![Word2Vec](https://img.shields.io/badge/Embedding-Word2Vec-yellow?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

---

## 🚀 Project Overview
This project focuses on **Sentiment Analysis for Product Reviews** using **Support Vector Machine (SVM)** and **Word2Vec** for text embeddings. The model is deployed using **Streamlit**, providing an interactive interface to analyze customer sentiments. 📊💡

---

## 🛠️ Tech Stack
- **Programming Language:** Python 🐍
- **Machine Learning Model:** Support Vector Machine (SVM) 📈
- **Text Embeddings:** Word2Vec 🔤
- **Deployment:** Streamlit 🌐
- **Libraries Used:** Scikit-Learn, Gensim, Pandas, NumPy, Matplotlib, Streamlit, NLTK

---

## ⚙️ Installation
Follow these steps to set up and run the project locally:

```bash
# Clone the repository
git clone https://github.com/your-username/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Application
After installing the dependencies, run the following command:

```bash
streamlit run app.py
```

The Streamlit app will launch in your browser, where you can enter a product review and analyze its sentiment. 🎭

---

## 📂 Project Structure
```
📂 sentiment-analysis-product-reviews
│── 📄 app.py            # Streamlit application
│── 📂 models            # Saved trained models
│── 📂 data              # Dataset for training
│── 📄 requirements.txt  # Required Python packages
│── 📄 sentiment-analysis-product-review.ipynb  # Jupyter Notebook for analysis
│── 📄 README.md         # Project documentation
```

---

## 📊 Model Training & Evaluation
- The dataset consists of labeled product reviews.
- Text is preprocessed (tokenization, stopword removal, stemming, etc.).
- Word2Vec is used to generate word embeddings.
- SVM is trained on the embeddings for sentiment classification.
- The trained model is saved and used in the Streamlit app.

---

## 🎯 Features
✅ Sentiment classification (Positive, Negative, Neutral) 👍👎
✅ Interactive Streamlit UI 🎨
✅ Word2Vec embeddings for better feature representation 🔍
✅ Scalable & customizable 🛠️
✅ Jupyter Notebook included for detailed exploratory analysis 📒


---

## 🏆 Future Enhancements
- Implementing other NLP models (e.g., LSTMs, Transformers)
- Enhancing the UI with more visualization tools
- Deploying the model to cloud platforms

---

## 🤝 Contributing
Contributions are welcome! Feel free to fork this repository and submit a pull request. 😊

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify it as needed.

---

## 📬 Contact
For any questions or suggestions, feel free to reach out:
📧 Email: banner.cse1998@outlook.com
🔗 LinkedIn: [Your Profile](https://linkedin.com/in/banner2399)  

Happy Coding! 🚀🔥
