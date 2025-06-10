# Decision Tree Maker

Dive into the Titanic dataset and discover how attributes like age, gender, and class played a role in survival. This project leverages machine learning to shed light on the human side of the tragedy, one prediction at a time.

*June 2025*

## 🚢 Overview

This project is a hands-on approach to the classic [Kaggle Titanic challenge](https://www.kaggle.com/c/titanic).  
Using a Random Forest Classifier, the goal is to predict which passengers survived the Titanic disaster, based on features like passenger class, gender, and family relationships.

Along the way, I:
- **Cleaned and prepped the data:** Handled missing values, converted categorical data to numbers, and selected key features.
- **Built a Random Forest model:** Trained an ensemble of decision trees to make robust survival predictions.
- **Visualized the decision trees:** Saved each individual tree as an image and combined them into a single grid for easy viewing.

This project helped me learn the full machine learning workflow—from data cleaning to model interpretation—using `scikit-learn`, `matplotlib`, and `Pillow`.

## ✨ Features

- Data cleaning and preprocessing
- Feature engineering with one-hot encoding
- Random Forest Classifier for survival prediction
- Visualization of every decision tree in the forest
- Combined grid image of all trees
- Uses Python libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `Pillow`

## ⚙️ Installation

1. **Clone the repository:**
    git clone https://github.com/DvP05/Titanic_Survival_Prediction.git

2. **Chnage the directory:**
    cd titanic-survival-prediction

3. **Install dependencies:**
    pip install -r requirements.txt

*Main libraries:* pandas, numpy, scikit-learn, matplotlib, Pillow

## 🚀 Usage

1. Download the Titanic dataset from Kaggle and place `train.csv` and `test.csv` in your project folder.
2. Update the file paths in the code if needed.
3. Run the main script:
   python titanic_survival.py

4. The script will:
- Train a Random Forest on selected features (`Pclass`, `Sex`, `SibSp`, `Parch`)
- Save each decision tree as a PNG in the `decision_trees/` folder
- Combine all tree images into `combined_decision_trees.png`
- Display the combined image

## 📁 Project Structure

<pre>
titanic-survival-prediction/
├── titanic prediction.py
├── requirements.txt
├── train.csv
├── test.csv
├── decision_trees/
│ ├── decision_tree_1.png
│ └── ...
├── combined_decision_trees.png
└── README.md
</pre>


## 🤝 Contributing

I welcome all contributions—big or small! If you’d like to collaborate, please check the open issues or submit your ideas via pull requests.

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

_Exploring data, one project at a time_
