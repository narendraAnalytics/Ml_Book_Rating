
![Project Image](path/to/image.png)

# Book Rating Prediction Project

## Dataset Description
The dataset contains information about various books, including their titles, authors, publishers, descriptions, page counts, categories, average ratings, ratings count, and language. Additional features such as the published year and book age have been created for this project.

### Columns and Features:
- **title**: Title of the book
- **authors**: Authors of the book
- **publisher**: Publisher of the book
- **published_date**: Date when the book was published
- **description**: Description of the book
- **page_count**: Number of pages in the book
- **categories**: Categories of the book
- **average_rating**: Average rating of the book
- **ratings_count**: Number of ratings the book has received
- **language**: Language of the book
- **published_year**: Extracted year of publication
- **book_age**: Age of the book based on the current year
- **author_publisher_interaction**: Interaction feature combining authors and publishers

## Project Overview
The objective of this project is to predict the average rating of a book based on its features. The project involves the following steps:
- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Model Building
- Model Evaluation
- Creating a Streamlit Dashboard for prediction

## Project Folder Structure
The project is organized into the following directories and files:
```
book-rating-prediction/
├── data/
│   ├── raw/
│   │   └── books.csv
│   ├── processed/
│   │   └── df_cleaned_with_interaction.csv
├── notebooks/
│   ├── data_cleaning.ipynb
│   ├── feature_engineering.ipynb
│   ├── eda.ipynb
│   ├── model_building.ipynb
│   ├── model_evaluation.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_building.py
│   ├── model_evaluation.py
├── app/
│   ├── streamlit_app.py
├── README.md
├── requirements.txt
```

## Getting Started
1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter notebooks in the `notebooks` directory to reproduce the analysis.
4. Launch the Streamlit app by running `streamlit run app/streamlit_app.py`.
