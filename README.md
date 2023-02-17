# Insurance Cost Prediction

This is a machine learning project that predicts the insurance cost for a patient based on their characteristics such as age, sex, bmi, number of children, smoking habits, and region.

The project consists of several stages:

        * Data cleaning and preprocessing
        * Train-test split
        * Data standardization
        * Model architecture design and hyperparameter tuning
        * Model training and evaluation
  
  
#### Dependencies

The following dependencies are required to run this project:

        * wandb    
        * sklearn
        * hydra
        * pandas
        * tensorflow
        * scipy

#### Usage

The main.py file consists of several functions that are called sequentially to perform the different stages of the project. These functions include:

data_cleaning: This function cleans the input data, transforms categorical data into numerical. It checks if the data is balanced and creates sample weights.

train_test_split: This function splits the data into training and testing sets.

data_standardization: This function standardizes the input data using the StandardScaler method from sklearn.

get_model: This function defines the model architecture and hyperparameters using the Sequential method from tensorflow.

get_lr_scheduler_earlystopping: This function defines the learning rate scheduler and early stopping using the ReduceLROnPlateau and EarlyStopping methods from tensorflow.

model_fit: This function trains the model using the fit method from tensorflow.

loss_comparison: This function compares the training and validation losses and plots them using matplotlib.

The main.py file also uses the hydra library to manage the configuration and parameters of the project. The configuration file is located in the conf directory and is named config.yaml. You can modify the configuration parameters to customize the project.

#### Contributing

If you find any issues or bugs in this project, please feel free to contribute by opening an issue or creating a pull request.