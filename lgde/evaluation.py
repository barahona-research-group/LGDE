"""Code for dictionary evaluation."""

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support


def evaluate_prediction(kw_list, y_true, document_term_matrix, with_report=True):
    """Evaluate list of keywords through performance of simple classifier.

    Parameters:
    -----------
    kw_list : list[str]
        List of keywords to use for prediction of document class.

    y_true : array of shape (n_documents, 1)
        True document classes (0 or 1).

    document_term_matrix : pandas DataFrame of shape (n_documents, n_words)
        Document term matrix with column indices corresponding to all words considered.
        All keywords in kw_list should correspond to columns in matrix.

    with_report : bool
        Print classification report if True.as_integer_ratio

    Returns:
    --------
    precision : float
        Macro precision score.

    recall : float
        Macro recall score.

    f1 : float
        Macro F1 score.
    --------

    """
    # predict document classes through simple keyword matching
    y_pred = np.sum(np.asarray(document_term_matrix[kw_list]), axis=1) > 0

    if with_report:
        # print classification report
        print(classification_report(y_true, y_pred, digits=3, labels=np.unique(y_pred)))

    # compute macro scores
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=np.unique(y_pred)
    )

    return precision, recall, f1
