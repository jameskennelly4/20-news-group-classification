import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def main():
    st.write("""
    #  Multiclass classification for 20-news-group data
    Showing results for ML classifiers. Choose a classifier on the sidebar.
    """)

    data_set = create_data_frame()
    X, y, X_train, X_test, y_train, y_test = preprocessing(data_set)

    choose_model = st.sidebar.selectbox("Choose the ML Model",
                                        ["None", "MultinomialNB", "BernoulliNB", "ComplementNB", "K-Nearest Neighbors",
                                         "XGBoost","LinearSVC"])

    if st.checkbox("Show data frame head"):
        st.write(data_set.head())

    cross_validate = st.sidebar.selectbox("Cross validate model?",
                                        ["No", "Yes"])
    if cross_validate == "Yes":
        st.markdown('#')
        fold_value_input = st.selectbox("Select number of folds",
                                     ('5', '10'))

    if choose_model == "None":
        st.markdown('#')
        st.text("No model chosen.")
    elif choose_model == "K-Nearest Neighbors" and cross_validate == "No":
        k_value_input = display_k_value_options()
        accuracy_score, classification_report = classify_data_KNN(X_train, X_test, y_train, y_test, int(k_value_input))
    elif cross_validate == "Yes":
        if choose_model == "K-Nearest Neighbors":
            k_value_input = display_k_value_options()
            mean, std_dev = classify_data_cv_knn(X, y, int(fold_value_input), int(k_value_input), choose_model)
        else:
            mean, std_dev = classify_data_cv(X ,y, int(fold_value_input), choose_model)
    else:
        accuracy_score, classification_report = classify_data(X_train, X_test, y_train, y_test, choose_model)

    if choose_model != "None":
        if cross_validate == "Yes":
            st.markdown('#')
            st.write("Mean accuracy score:")
            st.write(mean)
            st.markdown('#')
            st.write("Standard deviation of accuracy score:")
            st.text(std_dev)
        else:
            st.markdown('#')
            st.write("Accuracy score:")
            st.write(accuracy_score)
            st.markdown('#')
            st.text("Model Report:\n" + classification_report)


def display_k_value_options():
    k_value_input = st.selectbox("Select number of neighbors",
                                 ('1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                                  '100'))
    return k_value_input


@st.cache
def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame(newsgroups_train.target_names)
    targets.columns = ['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv('20_newsgroup.csv')


@st.cache
def create_data_frame():
    twenty_newsgroup_to_csv()
    data_set = pd.read_csv('20_newsgroup.csv')
    return data_set


@st.cache(allow_output_mutation=True)
def preprocessing(data_set):
    count_vect = CountVectorizer(stop_words='english')

    X = data_set.iloc[:, 1]
    y = data_set.iloc[:, 3]

    X = count_vect.fit_transform(X.values.astype('U'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
    return X, y, X_train, X_test, y_train, y_test


@st.cache(allow_output_mutation=True)
def get_classifier(input_classifier):
    if input_classifier == "MultinomialNB":
        classifier = MultinomialNB()
    elif input_classifier == "ComplementNB":
        classifier = ComplementNB()
    elif input_classifier == "BernoulliNB":
        classifier = BernoulliNB()
    elif input_classifier == "GaussianNB":
        classifier = GaussianNB()
    elif input_classifier == "XGBoost":
        classifier = XGBClassifier()
    elif input_classifier == "LinearSVC":
        classifier = LinearSVC()
    return classifier


@st.cache(suppress_st_warning=True)
def classify_data(X_train, X_test, y_train, y_test, input_classifier):
    classifier = get_classifier(input_classifier)

    if input_classifier == "GaussianNB":
        classifier.fit(X_train.toarray(), y_train)
        y_predict = classifier.predict(X_test.toarray())
        classifier.score(X_test.toarray(), y_test)
    else:
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)

@st.cache(suppress_st_warning=True)
def classify_data(X_train, X_test, y_train, y_test, input_classifier):
    classifier = get_classifier(input_classifier)

    if input_classifier == "LinearSVC":
        classifier.fit(X_train.toarray(), y_train)
        y_predict = classifier.predict(X_test.toarray())
        classifier.score(X_test.toarray(), y_test)
    else:
        classifier.fit(X_train, y_train)
        y_predict = classifier.predict(X_test)
        classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


@st.cache
def classify_data_KNN(X_train, X_test, y_train, y_test, k_value):
    knc_classifier = KNeighborsClassifier(n_neighbors=k_value)
    knc_classifier.fit(X_train, y_train)
    y_predict = knc_classifier.predict(X_test)
    knc_classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def classify_data_cv(X ,y, fold_value, input_classifier):
    classifier = get_classifier(input_classifier)
    scores = cross_val_score(classifier, X, y, cv=fold_value)
    return scores.mean(), scores.std()


@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def classify_data_cv_knn(X ,y, fold_value, k_value, input_classifier):
    knc_classifier = KNeighborsClassifier(n_neighbors=k_value)
    scores = cross_val_score(knc_classifier, X, y, cv=fold_value)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    main()
