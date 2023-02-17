import wandb
import sklearn
import hydra
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score as r2
from sklearn.metrics import explained_variance_score as exp_var
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from scipy.stats import gaussian_kde
from wandb.keras import WandbMetricsLogger
from omegaconf import DictConfig, OmegaConf


def data_cleaning(df, ord_enc, sw):
    # Dropping nan columns
    df.dropna(axis=0, inplace=True)
    # Transforming categorical data into numerical
    enc = ord_enc
    df[['sex', 'smoker', 'region']] = enc.fit_transform(df[['sex', 'smoker', 'region']]).astype('int')
    df['charges'] = df['charges'].round(decimals=2)
    # Checking if the data is balanced
    print("Imbalanced data")
    plt.hist(df['charges'], bins=10)
    plt.show()
    # Creating sample weights using inverse of population density
    kde = gaussian_kde(df['charges'])
    density = kde.evaluate(df['charges'])
    density = density / density.sum()
    sample_weights = np.clip((np.reciprocal(density) / 10000) * 4.5, a_max=5, a_min=0)
    sample_weights.mean()
    plt.scatter(df["charges"], sample_weights)
    plt.show()
    if sw==True:
        return (df, sample_weights)
    elif sw==False:
        return df, None


def train_test_split(df):
    # Convert dataframe into numpy array
    insurance_dataset = df.to_numpy()
    # Split features and target
    X_features = insurance_dataset[:, 0:5]
    Y_target = insurance_dataset[:, [6]]
    # Split train and test set
    # TODO: Check if the split is balanced
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_features, Y_target,
                                                                                test_size=0.33, random_state=42)
    return (X_train, X_test, y_train, y_test)


def data_standardization(X_train, X_test, y_train, y_test, scaler):
    # Train data scaling
    X_scaler = scaler
    X_scaler.fit(X_train)
    print('X_Train mean ', X_scaler.mean_)
    print('X_Train variance ', X_scaler.var_)
    X_train = X_scaler.transform(X_train)

    y_scaler = scaler
    y_scaler.fit(y_train)
    print('y_Train mean ', y_scaler.mean_)
    print('y_Train variance ', y_scaler.var_)
    y_train = y_scaler.transform(y_train)

    # Test data scaling
    X_test = X_scaler.transform(X_test)
    y_test = y_scaler.transform(y_test)
    return (X_train, X_test, y_train, y_test, y_scaler)


def dropout(regression_model, dp):
    if dp[0] == 'normal':
        return regression_model.add(layers.Dropout(rate=dp[1]))
    elif dp[0] == 'gaussian':
        return regression_model.add(layers.GaussianDropout(rate=dp[1]))
    elif dp[0] == 'alpha':
        return regression_model.add(layers.AlphaDropout(rate=dp[1]))


def get_model(cfg):
    # Define architecture
    regression_model = Sequential()
    regression_model.add(layers.Dense(cfg.layer_0, activity_regularizer=regularizers.L1L2(l1=cfg.l1_0, l2=cfg.l2_0),
                                      activation=cfg.activation_0))
    regression_model.add(BatchNormalization())
    dropout(regression_model, cfg.dropout)
    regression_model.add(layers.Dense(cfg.layer_1, activity_regularizer=regularizers.L1L2(l1=cfg.l1_1, l2=cfg.l2_1),
                                      activation=cfg.activation_1))
    regression_model.add(BatchNormalization())
    dropout(regression_model, cfg.dropout)
    regression_model.add(layers.Dense(1))
    # Compile the model
    opti = hydra.utils.instantiate(cfg.optimizer, learning_rate=cfg.learning_rate)
    regression_model.compile(optimizer=opti, loss=cfg.loss)
    return (regression_model)


def get_lr_scheduler_earlystopping(cfg):
    # Reduce learning rate when validation_loss flattens
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=cfg.rlr_monitor, factor=cfg.rlr_factor, patience=cfg.rlr_patience, verbose=1,
        mode='auto', min_delta=0.0, cooldown=0, min_lr=0)
    # Stop if validation loss doesn't improve
    early_stop = callbacks.EarlyStopping(
        monitor=cfg.es_monitor, min_delta=cfg.es_min_delta, patience=cfg.es_patience, verbose=1,
        mode='auto', baseline=None, restore_best_weights=True)
    return reduce_lr, early_stop


def model_fit(regression_model, X_train, y_train, reduce_lr, early_stop, eps, sample_weights):
    # Fit the model
    history = regression_model.fit(X_train, y_train, epochs= eps, verbose=2,
                                   callbacks=[early_stop, reduce_lr, WandbMetricsLogger()],
                                   validation_split=0.2, sample_weight=sample_weights)
    return (history)


def loss_comparison(history):
    # History of the fitting
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.4])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()


def output(regression_model, y_scaler, X_test, y_test):
    # Predicting output
    y_predict = regression_model.predict(X_test)
    y_predict = y_scaler.inverse_transform(y_predict)
    y_test = y_scaler.inverse_transform(y_test)
    mean_abs_error = mae(y_test, y_predict)
    r2_score = r2(y_test, y_predict)
    variance = exp_var(y_test, y_predict)
    wandb.log({"mean_abs_error": mean_abs_error, "r2_score": r2_score, "variance": variance})
    print("Variance ----------------> ", variance)
    print('Mean absolute error -----> ', mean_abs_error)
    print("R2 score ----------------> ", r2_score)


@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="insurance", config=config_dict)
    # Importing data as a pandas dataframe
    df = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\Projects\Insurance\insurance.csv')
    # Cleaning data
    df, sample_weights = data_cleaning(df, hydra.utils.instantiate(cfg.ord_enc), cfg.sw)
    # Splitting data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(df)
    # Standardizing data
    X_train, X_test, y_train, y_test, y_scaler = data_standardization(X_train, X_test, y_train, y_test,
                                                                      hydra.utils.instantiate(cfg.scaler))
    # Model architechture
    regression_model = get_model(cfg)
    # Stop overfitting
    reduce_lr, early_stop = get_lr_scheduler_earlystopping(cfg)
    # Fit the model
    history = model_fit(regression_model, X_train, y_train, reduce_lr, early_stop, cfg.epochs, sample_weights)
    # Loss comparison
    loss_comparison(history)
    # Output
    output(regression_model, y_scaler, X_test, y_test)


if __name__ == "__main__":
    my_app()
