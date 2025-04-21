# Cyber-Threat-Detection-Using-Machine-Learning-
 import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Preprocess the data
def process_data(df):  
    if df.empty:
        st.error("Please upload a dataset with data.")
        return None, None

    imputer = SimpleImputer(strategy='constant', fill_value=0)
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    label_encoder = LabelEncoder()
    for col in df_imputed.select_dtypes(include=['object']).columns:
        df_imputed[col] = label_encoder.fit_transform(df_imputed[col])
    
    numerical_features = df_imputed.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df_imputed[numerical_features] = scaler.fit_transform(df_imputed[numerical_features])
    df_imputed[numerical_features] = df_imputed[numerical_features].round().astype(int) 

    X = df_imputed.drop('class', axis=1)
    y = df_imputed['class']
    return X, y

def fitness_function(params, X_train, y_train):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')

    return np.mean(scores)

# Adaptive tuning function
def adaptive_hyperparameter_tuning(model, X_train, y_train, X_val, y_val, max_depth, min_samples_split, tolerance=0.90):
    model = RandomForestClassifier(n_estimators=100, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)

    if val_score < tolerance:
        if max_depth > 1:
            max_depth -= 1
        if min_samples_split < 10:
            min_samples_split += 1

    return model, max_depth, min_samples_split

# Slime Mold Algorithm for Hyperparameter Optimization
def slime_mold_algorithm(X_train, y_train, X_val, y_val, population_size=1, max_iterations=1):
    population = np.random.rand(population_size, 3)
    population[:, 0] = np.clip(population[:, 0], 50, 150)  # n_estimators
    population[:, 1] = np.clip(population[:, 1], 1, 20)    # max_depth
    population[:, 2] = np.clip(population[:, 2], 2, 10)    # min_samples_split

    best_solution = None
    best_fitness = -np.inf

    for iteration in range(max_iterations):
        fitness_values = []

        for individual in population:
            n_estimators = int(individual[0])
            max_depth = int(individual[1])
            min_samples_split = int(individual[2])

            model, max_depth, min_samples_split = adaptive_hyperparameter_tuning(None, X_train, y_train, X_val, y_val, max_depth, min_samples_split)
            fitness = fitness_function([n_estimators, max_depth, min_samples_split], X_train, y_train)
            fitness_values.append(fitness)

        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > best_fitness:
            best_fitness = fitness_values[best_index]
            best_solution = population[best_index]

        for i in range(population_size):
            step = np.random.uniform(-0.1, 0.1, size=3)
            population[i] += step * (population[i] - best_solution)
            population[i] = np.clip(population[i], [50, 1, 2], [150, 20, 10])

    return best_solution, best_fitness


def flamingo_search_algorithm(X_train, y_train, population_size=1, max_iterations=1):
    num_features = X_train.shape[1]
    population = np.random.rand(population_size, num_features)
    model = RandomForestClassifier(random_state=42)
    best_solution = None
    best_fitness = -np.inf

    for iteration in range(max_iterations):
        fitness_values = []

        for individual in population:
            selected_features = np.where(individual > 0.5)[0]  
            print(f"Selected features: {selected_features}")  

           
            if len(selected_features) == 0:
                selected_features = [0]  
                fitness = -np.inf
            else:
                X_selected = X_train[:, selected_features]
                fitness = np.mean(cross_val_score(model, X_selected, y_train, cv=5, scoring='accuracy'))
      
            fitness_values.append(fitness)

        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > best_fitness:
            best_fitness = fitness_values[best_index]
            best_solution = population[best_index]

        for i in range(population_size):
            step = np.random.uniform(-0.1, 0.1, size=num_features)
            population[i] += step * (population[i] - best_solution)
            population[i] = np.clip(population[i], 0, 1)

    selected_features = np.where(best_solution > 0.5)[0]  # Final selected features 
    return selected_features

# Generate Adversarial Examples
def generate_adversarial_example(model, X, epsilon=0.01):
    X_adv = np.copy(X)
    perturbation = np.random.uniform(-epsilon, epsilon, size=X.shape)
    np.add(X_adv,perturbation)
    X_adv = np.clip(X_adv, X.min().min(), X.max().max())
    return X_adv

__name__ = "_main_"
# Streamlit Front-End
st.title("Cyber Threat Detection using Machine Learning")

st.sidebar.header("Upload Your Dataset")
file = st.sidebar.file_uploader("File Name", type=["csv"])

# Read the dataset
if file is not None:
    st.write("File uploaded successfully!")
    df= pd.read_csv(file)  
    st.write("Dataset Overview:")
    st.dataframe(df.head())
    # Preprocessing
    st.write("Dataset Shape:", df.shape)

else:
    st.write("No file uploaded yet.")

X, y = process_data(df)
n = st.number_input("Enter number of rows to display:", min_value=1, max_value=len(X), value=5, step=1)
st.write(f"Preprocessed Data:", pd.DataFrame(X).head(n))
if y is not None:
    class_distribution = y.value_counts()
    st.write("Class Distribution Before Adversarial Example Generation:")
    st.bar_chart(class_distribution)
    
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
    # Feature selection and hyperparameter optimization
    selected_features = flamingo_search_algorithm(X_train.values, y_train.values)
    X_train_selected = X_train.iloc[:, selected_features]  
    X_test_selected = X_test.iloc[:, selected_features]  
    st.write(f"Final selected features: {selected_features}") 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    best_params, best_fitness = slime_mold_algorithm(X_train_selected.values, y_train.values, X_test_selected.values, y_test.values)
    best_n_estimators, best_max_depth, best_min_samples_split = map(int, best_params)
    final_model = RandomForestClassifier(n_estimators=best_n_estimators, max_depth=best_max_depth, min_samples_split=best_min_samples_split, class_weight='balanced', random_state=42)
    # Generate adversarial examples
    X_train_adv = generate_adversarial_example(final_model, X_train_selected.values, epsilon=0.01)
    X_train_combined = np.vstack((X_train_selected.values, X_train_adv))
    y_train_combined = np.hstack((y_train.values, y_train.values))
    final_model.fit(X_train_combined, y_train_combined)
    # Generate adversarial test examples
    X_test_adv = generate_adversarial_example(final_model, X_test_selected.values, epsilon=0.01)
    X_test_combined = np.vstack((X_test_selected.values, X_test_adv))
    y_test_combined = np.hstack((y_test.values, y_test.values))
    # Model Evaluation
    y_pred = final_model.predict(X_test_combined)
    accuracy_scores.append(accuracy_score(y_test_combined, y_pred))
    precision_scores.append(precision_score(y_test_combined, y_pred))
    recall_scores.append(recall_score(y_test_combined, y_pred))
    f1_scores.append(f1_score(y_test_combined, y_pred))
    # Display results
    st.write(f"Accuracy: {np.mean(accuracy_scores):.4f}")
    st.write(f"Precision: {np.mean(precision_scores):.4f}")
    st.write(f"Recall: {np.mean(recall_scores):.4f}")
    st.write(f"F1 Score: {np.mean(f1_scores):.4f}")
    # Final classification output
    y_pred_labels = ["Normal" if pred == -1 else "Attack" for pred in y_pred]
    st.write("\nFinal Classification Output:")
    for i in range(min(n,len(y_pred))):  
        st.write(f"Data Sample {i + 1}: {y_pred_labels[i]}")

st.write("Hyperparameters Tuned for This Fold:")
st.write(f"- Number of Estimators (n_estimators): {best_n_estimators}")
st.write(f"- Maximum Depth (max_depth): {best_max_depth}")
st.write(f"- Minimum Samples Split (min_samples_split): {best_min_samples_split}")
 
class_distribution_after = pd.Series(y_train_combined).value_counts()
st.write("Class Distribution After Adversarial Example Generation:")
st.bar_chart(class_distribution_after)
