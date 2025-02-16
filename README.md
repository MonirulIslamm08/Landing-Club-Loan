Lending Club Loan Defaulters Prediction

📌 Project Overview

This project aims to analyze and predict loan defaults using Lending Club loan data. We perform exploratory data analysis (EDA), feature engineering, and machine learning modeling to identify patterns and risk factors associated with loan repayment.

📂 Dataset Information

The dataset contains information about various loan applicants, including their employment details, loan amounts, interest rates, and repayment statuses. Some key features include:

loan_amnt: Loan amount requested

term: Loan duration (36 or 60 months)

int_rate: Interest rate

installment: Monthly installment amount

emp_length: Employment length in years

home_ownership: Homeownership status

annual_inc: Annual income

loan_status: Indicates if the loan was fully paid or defaulted

🔍 Exploratory Data Analysis (EDA)

Visualized loan status distribution using sns.countplot()

Identified high correlations between loan_amnt and installment

Analyzed the impact of employment length, home ownership, and annual income on loan defaults

Created heatmaps to understand feature correlations

🏗 Data Preprocessing

Handled missing values in mort_acc, emp_title, and emp_length

Converted categorical variables to numerical using pd.get_dummies()

Scaled numerical features using MinMaxScaler() to normalize data

🤖 Model Building

A deep learning model was developed using TensorFlow:

Input Layer: 78 features

Hidden Layers: Three dense layers with 39, 19 neurons, using ReLU activation and dropout regularization

Output Layer: 1 neuron with sigmoid activation

Loss Function: Binary Cross-Entropy

Optimizer: Adam

🎯 Model Performance

Trained for 25 epochs with batch size 256

Achieved 89% accuracy on the test set

Precision, recall, and F1-score were evaluated using classification_report()

Confusion matrix showed a strong classification ability

🔮 Prediction Example

A random loan applicant’s data was passed through the trained model to predict repayment probability.

🚀 Technologies Used

Python: pandas, NumPy, Matplotlib, Seaborn, Plotly

Machine Learning: Scikit-Learn

Deep Learning: TensorFlow, Keras

📝 How to Run the Project

Clone this repository:

git clone https://github.com/your-username/Lending-Club-Loan-Prediction.git

Install dependencies:

pip install -r requirements.txt

Run the Jupyter Notebook or script:

python loan_prediction.py

📌 Future Improvements

Implement more advanced models like Random Forest or XGBoost

Fine-tune hyperparameters for better accuracy

Deploy the model as a web application

📧 Contact

For any queries, reach out to [your email or GitHub profile].
