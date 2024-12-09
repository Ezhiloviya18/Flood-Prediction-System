from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('agg')

def load_and_preprocess_data(state):
    dfx = pd.read_csv("/Users/joel/Pictures/rainfall in india 1901-2015.csv")
    grouped_data = dfx.groupby('STATE')

    if state not in grouped_data.groups:
        raise ValueError(f"Data for state {state} not found in the dataset.")

    x = grouped_data.get_group(state)

    # Your existing data preprocessing code here...
    y1 = list(x["YEAR"])
    x1 = list(x["Oct-Dec"])
    z1 = list(x["OCT"])
    n1 = list(x["NOV"])
    w3 = list(x["SEP"])
    flood = []
    nov = []
    sub = []

    for i in range(0, len(x1)):
        if x1[i] > 580:
            flood.append('1')
        else:
            flood.append('0')

    for k in range(0, len(x1)):
        nov.append(n1[k] / 3)

    for k in range(0, len(x1)):
        sub.append(abs(w3[k] - z1[k]))

    x["flood"] = flood
    x["avgnov"] = nov
    x["sub"] = sub

    X = x.loc[:, ["Oct-Dec", "avgnov", "sub"]].values
    y1 = x.iloc[:, 19].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y1, random_state=42, test_size=0.25)

    return X_train, X_test, Y_train, Y_test

def train_and_predict(X_train, X_test, Y_train):
    # Initialize individual models
    knn_model = KNeighborsClassifier(n_neighbors=3)
    svm_model = SVC(probability=True)
    ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, activation='logistic',
                              learning_rate='adaptive', solver='sgd')  # Initialize the ANN

    # Initialize the ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[('knn', knn_model), ('svm', svm_model), ('ann', ann_model)],
                                      voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_train, Y_train)

    # Make predictions using the ensemble model
    y_pred = ensemble_model.predict(X_test)

    return y_pred

def generate_prediction_graph(Y_test, y_pred):
    # Your existing code for generating the prediction graph...
    print("\nAccuracy Score:%f" % (accuracy_score(Y_test, y_pred) * 100))
    print("ROC score:%f" % (roc_auc_score(Y_test, y_pred) * 100))
    mat = confusion_matrix(Y_test, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("True Class")
    plt.ylabel("Predicted Class")
    graph_file = 'static/graph.png'
    plt.savefig(graph_file)
    plt.close()

    return graph_file

def evaluate_model(Y_test, y_pred):
    # Your existing code for evaluating the model...
    accuracy = accuracy_score(Y_test, y_pred)
    print("Ensemble Model Accuracy: {:.2f}".format(accuracy))
    print("\nAccuracy Score:%f" % (accuracy_score(Y_test, y_pred) * 100))
    print("ROC score:%f" % (roc_auc_score(Y_test, y_pred) * 100))

    return accuracy

def evaluate(a,b,c,X_train,Y_train):
    knn_model = KNeighborsClassifier(n_neighbors=3)
    svm_model = SVC(probability=True)
    ann_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, activation='logistic',
                              learning_rate='adaptive', solver='sgd')  # Initialize the ANN

    # Initialize the ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(estimators=[('knn', knn_model), ('svm', svm_model), ('ann', ann_model)],
                                      voting='soft')

    # Train the ensemble model
    ensemble_model.fit(X_train, Y_train)
    return ensemble_model.predict([[a,b,c]])


def fpredict(state, input1, input2, input3):
    try:
        X_train, X_test, Y_train, Y_test = load_and_preprocess_data(state)

        # Train and predict
        y_pred = train_and_predict(X_train, X_test, Y_train)

        # Evaluate the model
        accuracy = evaluate_model(Y_test, y_pred)

        result=evaluate(input1,input2,input3,X_train,Y_train)

        # Generate prediction graph
        graph_file = generate_prediction_graph(Y_test, y_pred)

        return result,graph_file

    except Exception as e:
        return str(e)

