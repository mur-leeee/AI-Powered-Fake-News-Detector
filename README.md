# AI-Powered-Fake-News-Detector
AI-Powered-Fake-News-Detector
This is a hybrid disinformation detection system that combines classical machine learning with advanced generative AI reasoning. The system utilizes a two-tier pipeline to identify misinformation with high efficiency and deep contextual analysis.

System Architecture
Baseline Classifier: A Logistic Regression model using TF-IDF vectorization for rapid initial screening. This layer identifies potential misinformation based on linguistic patterns and word frequencies.

Reasoning Agent: A generative AI layer powered by the Gemma 4 (26B-MoE) architecture. This stage performs deep analysis to identify logical fallacies, emotional manipulation, and lack of credible sourcing in flagged content.

Project Structure
```text
fake-news-detector/
├── app/
│   └── app.py            # Streamlit web application with Gemma 4 integration
├── data/
│   ├── True.csv          # Dataset of verified news articles
│   └── Fake.csv          # Dataset of identified misinformation
├── models/               # Directory for serialized .pkl model files
├── src/
│   ├── train_model.py    # Script for data preprocessing and model training
│   └── predict.py        # Command-line interface for standalone inference
├── .gitignore            # Configuration to exclude secrets and local binaries
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```
Technical Specifications
Machine Learning: Scikit-learn (Logistic Regression), Pandas, Pickle

Natural Language Processing: NLTK (Tokenization, Stopword removal)

Generative AI: Google Generative AI SDK (Gemma 4 Architecture)

Frontend: Streamlit

Security: Implementation of Streamlit Secret-Management for API credential protection and a passcode-protected access gateway.

Installation and Usage
1. Environment Setup
Clone the repository and install the required dependencies:

Bash
git clone https://github.com/mur-leeee/AI-Powered-Fake-News-Detector.git
cd AI-Powered-Fake-News-Detector
pip install -r requirements.txt
2. Configuration
Create a .streamlit/secrets.toml file in the root directory to store your credentials:

Ini, TOML
GOOGLE_API_KEY = "YOUR_API_KEY"
APP_PASSWORD = "YOUR_CUSTOM_PASSWORD"
3. Execution
To launch the web interface:

Bash
streamlit run app/app.py
To run a prediction via the command line interface:

Bash
python src/predict.py
Future Scope
Integration of real-time web scraping for live fact-checking.

Expansion of the training dataset to include regional news cycles from 2026.

Fine-tuning Gemma 4 on specific disinformation categories such as financial fraud or political propaganda.
