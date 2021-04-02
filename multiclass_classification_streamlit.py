import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def main():
    st.write("""
    #  Multiclass classification for 20-news-group data
    Showing results for ML classifiers. Choose a classifier on the sidebar.
    """)
    data_set = create_data_frame()

    if st.checkbox('Show data frame head'):
        st.subheader("Showing data frame head")
        st.write(data_set.head())

    choose_model = st.sidebar.selectbox("Choose the ML Model",
                                        ["NONE", "BernoulliNB", "ComplementNB", "MultinomialNB", "K-Nearest Neighbors"])

    X_train, X_test, y_train, y_test = preprocessing(data_set)

    if choose_model == "MultinomialNB":
        accuracy_score, classification_report = classify_data_MultinomialNB(X_train, X_test, y_train, y_test)
    elif choose_model == "ComplementNB":
        accuracy_score, classification_report = classify_data_ComplementNB(X_train, X_test, y_train, y_test)
    elif choose_model == "BernoulliNB":
        accuracy_score, classification_report = classify_data_BernoulliNB(X_train, X_test, y_train, y_test)
    elif choose_model == "GaussianNB":
        accuracy_score, classification_report = classify_data_GaussianNB(X_train, X_test, y_train, y_test)
    elif choose_model == "K-Nearest Neighbors":
        k_value_input = st.selectbox("Number of Neighbors",
                              ('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','100'))
        accuracy_score, classification_report = classify_data_KNN(X_train, X_test, y_train, y_test, int(k_value_input))

    st.write(accuracy_score)
    st.write(classification_report)


def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame(newsgroups_train.target_names)
    targets.columns = ['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv('20_newsgroup.csv')


def create_data_frame():
    twenty_newsgroup_to_csv()
    data_set = pd.read_csv('20_newsgroup.csv')
    return data_set


def preprocessing(data_set):
    count_vect = CountVectorizer(stop_words='english')

    X = data_set.iloc[:, 1]
    y = data_set.iloc[:, 2]

    X = count_vect.fit_transform(X.values.astype('U'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)
    return X_train, X_test, y_train, y_test


def classify_data_MultinomialNB(X_train, X_test, y_train, y_test):
    mnb_classifier = MultinomialNB()
    mnb_classifier.fit(X_train, y_train)
    y_predict = mnb_classifier.predict(X_test)
    mnb_classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


def classify_data_ComplementNB(X_train, X_test, y_train, y_test):
    cnb_classifier = ComplementNB()
    cnb_classifier.fit(X_train, y_train)
    y_predict = cnb_classifier.predict(X_test)
    cnb_classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


def classify_data_BernoulliNB(X_train, X_test, y_train, y_test):
    bnb_classifier = BernoulliNB()
    bnb_classifier.fit(X_train, y_train)
    y_predict = bnb_classifier.predict(X_test)
    bnb_classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


def classify_data_GaussianNB(X_train, X_test, y_train, y_test):
    gnb_classifier = GaussianNB()
    gnb_classifier.fit(X_train.toarray(), y_train)
    y_predict = gnb_classifier.predict(X_test.toarray())
    gnb_classifier.score(X_test.toarray(), y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


def classify_data_KNN(X_train, X_test, y_train, y_test, k_value):
    knc_classifier = KNeighborsClassifier(n_neighbors=k_value)
    knc_classifier.fit(X_train, y_train)
    y_predict = knc_classifier.predict(X_test)
    knc_classifier.score(X_test, y_test)

    return accuracy_score(y_test, y_predict), classification_report(y_test, y_predict)


if __name__ == "__main__":
    main()
