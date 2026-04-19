import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer

# load + clean
def load_data(path):
    df = pd.read_excel(path)
    df = df.select_dtypes(include=['number'])
    df = df.fillna(0)
    df["Target"] = (df.iloc[:, 0] > df.iloc[:, 0].mean()).astype(int)
    X = df.drop("Target", axis=1)
    y = df["Target"]
    return X, y

# a1
def build_model():
    base = [
        ('rf', RandomForestClassifier()),
        ('svm', SVC(probability=True))
    ]
    return StackingClassifier(estimators=base, final_estimator=LogisticRegression())

# a2
def build_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

# a3
def lime_explain(pipe, X_train, X_test):
    explainer = LimeTabularExplainer(X_train.values,
                                     feature_names=X_train.columns.tolist(),
                                     class_names=['0','1'],
                                     mode='classification')
    exp = explainer.explain_instance(
        X_test.iloc[0].values,
        lambda x: pipe.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    )
    return exp

# main
X, y = load_data("Clarity_Text_student_teacher_with_glove.xlsx")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipe = build_pipeline(build_model())
pipe.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, pipe.predict(X_test)))

exp = lime_explain(pipe, X_train, X_test)
exp.save_to_file("lime_output.html")