# DATA-PIPELINE-DEVELOPMENT
COMPANY: CODTECH IT SOLUTIONS

NAME: P.Vishnu Prakash

INTERN ID: CT04DH1031

DOMAIN: DATA SCIENCE

DURATION: 4 WEEEKS

MENTOR: NEELA SANTOSH


The pipeline is structured into several key steps:

1. **Data Loading**: Using Pandas, the dataset is read from a CSV file into a DataFrame. This step ensures flexibility and compatibility with common data formats.

2. **Feature Separation**: Features (input variables) are separated into **numerical** and **categorical** columns based on data types. This allows appropriate preprocessing techniques to be applied to each type.

3. **ColumnTransformer** is used to apply the numerical and categorical transformations in parallel, preserving column alignment and ensuring compatibility with downstream models.

4. **Train-Test Splitting**: The dataset is split into training and testing sets using `train_test_split`, maintaining reproducibility through a fixed random seed.

5. **Transformation Application**: The preprocessing pipeline is fitted on the training data and applied to both training and test sets to ensure consistency.

This approach makes the preprocessing process reusable, scalable, and easier to maintain. By encapsulating logic in functions and leveraging Scikit-learn’s pipeline architecture, the code remains clean and adaptable for future enhancements like model integration or cross-validation.

This pipeline is ideal for real-world machine learning tasks and can be directly integrated into model training workflows.
