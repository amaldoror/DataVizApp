#!/usr/bin/env python
# coding: utf-8

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

import tensorflow as tf
from keras.src import Sequential
from keras.src.layers import Dense #, Flatten
from keras.src.losses import CategoricalCrossentropy
from keras.src.optimizers import Adam
from keras.src.activations import relu, softmax

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from time import time


def preprocess_data():
    mnist = fetch_openml('mnist_784')
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')

    # DataSet in Trainings- und Testdaten splitten im Verhältnis 80/20
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Skalieren der Daten auf den Wertebereich [0, 1]
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, x, y


def plot_sample_images(X, y, num=5):
    plt.figure(figsize=(15, 20))
    for i in range(num):
        img = X.iloc[i].values.reshape(28, 28)
        plt.subplot(1, 5, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {y[i]}")
        plt.axis('off')

    plt.show()


def train_lr(x_train_scaled, x_test_scaled, y_train, y_test):
    clf_lr = LogisticRegression(max_iter=1000, random_state=0)

    # Starte die Zeitmessung für das Training
    start_time_lr = time()

    # Trainieren des Modells
    clf_lr.fit(x_train_scaled, y_train)

    # Beende die Zeitmessung und berechne die Dauer
    training_time_lr = time() - start_time_lr

    # Evaluieren des Modells
    y_pred_lr = clf_lr.predict(x_test_scaled)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)

    return clf_lr, training_time_lr, accuracy_lr


def print_lr(clf_lr, training_time_lr, accuracy_lr):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Training Time:              {training_time_lr}
    Accuracy:                   {accuracy_lr:.4f}
    Intercept:                  {clf_lr.intercept_}
    Class Weight:               {clf_lr.class_weight}
    Classes:                    {clf_lr.classes_}
    Random State:               {clf_lr.random_state}
    C:                          {clf_lr.C}
    N Jobs:                     {clf_lr.n_jobs}
    Dual:                       {clf_lr.dual}
    Fit intercept:              {clf_lr.fit_intercept}
    Intercept Scaling:          {clf_lr.intercept_scaling}
    L1 Ratio:                   {clf_lr.l1_ratio}
    Max Iter:                   {clf_lr.max_iter}
    N Iter:                     {clf_lr.n_iter_}
    Penalty:                    {clf_lr.penalty}
    Solver:                     {clf_lr.solver}
    Tol:                        {clf_lr.tol}
    Verbose:                    {clf_lr.verbose}
    Warm Start:                 {clf_lr.warm_start}
    Coef:                       {clf_lr.coef_}
    """)


def train_dt(x_train_scaled, y_train, x_test_scaled, y_test):
    clf_dt = DecisionTreeClassifier(random_state=0)

    # Starte die Zeitmessung für das Training
    start_time_dt = time()

    # Trainieren des Modells
    clf_dt.fit(x_train_scaled, y_train)

    # Beende die Zeitmessung und berechne die Dauer
    training_time_dt = time() - start_time_dt

    cross_val_score(clf_dt, x_train_scaled, y_train, cv=10)

    # Evaluieren des Modells
    y_pred_dt = clf_dt.predict(x_test_scaled)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)

    return clf_dt, training_time_dt, accuracy_dt


def print_dt(clf_dt, training_time_dt, accuracy_dt):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Training Time:              {training_time_dt}
    Accuracy:                   {accuracy_dt:.4f}
    Criterion:                  {clf_dt.criterion}
    Tree:                       {clf_dt.tree_}
    CCP Alpha:                  {clf_dt.ccp_alpha}
    Class Weight:               {clf_dt.class_weight}
    Classes:                    {clf_dt.classes_}
    Max Depth:                  {clf_dt.max_depth}
    Max Features:               {clf_dt.max_features}
    Max Leaf Nodes:             {clf_dt.max_leaf_nodes}
    Min Samples Leaf:           {clf_dt.min_samples_leaf}
    Min Samples Split:          {clf_dt.min_samples_split}
    Min Impurity Decrease:      {clf_dt.min_impurity_decrease}
    Splitter:                   {clf_dt.splitter}
    Monotonic cst:              {clf_dt.monotonic_cst}
    Random State:               {clf_dt.random_state}
    """)


def train_rf(x_train_scaled, y_train, x_test_scaled, y_test):

    clf_rf = RandomForestClassifier(max_depth=2, random_state=0)

    # Starte die Zeitmessung für das Training
    start_time_rf = time()

    # Trainieren des Modells
    clf_rf.fit(x_train_scaled, y_train)

    # Beende die Zeitmessung und berechne die Dauer
    training_time_rf = time() - start_time_rf

    y_pred_rf = clf_rf.predict(x_test_scaled)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)

    return clf_rf, training_time_rf, accuracy_rf


def print_rf(clf_rf, training_time_rf, accuracy_rf):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Training Time:              {training_time_rf}
    Accuracy:                   {accuracy_rf:.4f}
    Criterion:                  {clf_rf.criterion}
    CCP Alpha:                  {clf_rf.ccp_alpha}
    Class Weight:               {clf_rf.class_weight}
    Classes:                    {clf_rf.classes_}
    Max Depth:                  {clf_rf.max_depth}
    Max Features:               {clf_rf.max_features}
    Max Leaf Nodes:             {clf_rf.max_leaf_nodes}
    Min Samples Leaf:           {clf_rf.min_samples_leaf}
    Min Samples Split:          {clf_rf.min_samples_split}
    Min Impurity Decrease:      {clf_rf.min_impurity_decrease}
    Monotonic cst:              {clf_rf.monotonic_cst}
    Random State:               {clf_rf.random_state}
    """)


def train_svm(x_train_scaled, y_train, x_test_scaled, y_test):
    clf_svm = svm.SVC()

    # Starte die Zeitmessung für das Training
    start_time_svm = time()

    # Trainieren des Modells
    clf_svm.fit(x_train_scaled, y_train)

    # Beende die Zeitmessung und berechne die Dauer
    training_time_svm = time() - start_time_svm

    y_pred_svm = clf_svm.predict(x_test_scaled)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)

    return clf_svm, training_time_svm, accuracy_svm


def print_svm(clf_svm, training_time_svm, accuracy_svm):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Training Time:              {training_time_svm}
    Accuracy:                   {accuracy_svm:.4f}
    Class Weight:               {clf_svm.class_weight}
    Classes:                    {clf_svm.classes_}
    Random State:               {clf_svm.random_state}
    Tol:                        {clf_svm.tol}
    C:                          {clf_svm.C}
    NU:                         {clf_svm.nu}
    Verbose:                    {clf_svm.verbose}
    Max Iter:                   {clf_svm.max_iter}
    Break Ties:                 {clf_svm.break_ties}
    Cache Size:                 {clf_svm.cache_size}
    Coef 0:                     {clf_svm.coef0}
    Degree:                     {clf_svm.degree}
    Epsilon:                    {clf_svm.epsilon}
    Gamma:                      {clf_svm.gamma}
    Kernel:                     {clf_svm.kernel}
    Probability:                {clf_svm.probability}
    Shrinking:                  {clf_svm.shrinking}
    Unused Param:               {clf_svm.unused_param}
    Decision Function Shape:    {clf_svm.decision_function_shape}
    """)


def train_snn(x_train_scaled, x_test_scaled, y_train, y_test):
    # Modell erstellen
    model = Sequential([
        # Flatten(input_shape=(28, 28)),  # Input Layer: Flatten der 28x28 Bilder zu 784-dimensionalen Vektoren
        Dense(128, activation=relu),  # Hidden Layer 1: 128 Neuronen, relu Aktivierungsfunktion
        Dense(64, activation=relu),  # Hidden Layer 2: 64 Neuronen, relu Aktivierungsfunktion
        Dense(10, activation=softmax)  # Output Layer: 10 Neuronen (Anzahl Klassen), softmax Aktivierungsfunktion
    ])

    # Anzahl der Klassen bestimmen
    num_classes = len(np.unique(y_train))

    # Kompilieren des Modells
    model.compile(optimizer=Adam(),
                  loss=CategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Modellzusammenfassung anzeigen
    model.summary()

    # Starte die Zeitmessung für das Training
    start_time_nn = time()

    # Modell trainieren
    model.fit(x_train_scaled, tf.keras.utils.to_categorical(
        y_train,
        num_classes=num_classes),
              epochs=10,
              validation_data=(x_test_scaled, tf.keras.utils.to_categorical(y_test,
                                                                            num_classes=num_classes)))

    # Beende die Zeitmessung und berechne die Dauer
    training_time_nn = time() - start_time_nn

    y_pred_nn = model.predict(x_test_scaled)

    # ValueError: Classification metrics can't handle a mix of multiclass and continuous-multioutput targets
    # accuracy_nn = accuracy_score(y_test, y_pred_nn)

    loss, accuracy_nn = model.evaluate(x_test_scaled, tf.keras.utils.to_categorical(y_test, num_classes=num_classes))

    return model, training_time_nn, accuracy_nn


def print_model_info(model, training_time, accuracy, additional_info=""):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Training Time:              {training_time}
    Accuracy:                   {accuracy:.4f}
    {additional_info}
    """)


def print_model(model, training_time_nn, accuracy_nn, loss):
    print(f"""
    Modellinformationen:
    ----------------------------------------------------------------------------------
    Name:                       {model.name}
    Training Time:              {training_time_nn}
    Accuracy:                   {accuracy_nn:.4f}
    Loss:                       {loss:.4f}
    Activity Regularizer:       {model.activity_regularizer}
    Autocast:                   {model.autocast}
    Compiled:                   {model.compiled}
    Stop Training:              {model.stop_training}
    Stop Evaluating:            {model.stop_evaluating}
    Stop Predicting:            {model.stop_predicting}
    Steps per Execution:        {model.steps_per_execution}
    Built:                      {model.built}
    Supports jit:               {model.supports_jit}
    Test Function:              {model.test_function}
    Train Function:             {model.train_function}
    Predict Function:           {model.predict_function}
    Optimizer:                  {model.optimizer}
    """)


def compare_models(model_infos):
    df_eval = pd.DataFrame(model_infos)

    fix, ax = plt.subplots(figsize=(3, 3))
    ax.axis('off')
    col_widths = [.5]*len(df_eval.columns)
    table = pd.plotting.table(ax,
                              df_eval,
                              loc='center',
                              cellLoc='left',
                              colWidths=col_widths,
                              rowLoc='center',
                              edges='open')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.5, 1.5)

    plt.suptitle('Model Comparison')
    plt.show()
