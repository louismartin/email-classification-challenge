import numpy as np


def top_emails(Y_pred, recipients_map, top=10):
    """
    Computes the emails with top scores from numerical matrix.
    Args:
        Y_pred (ndarray): Prediciton matrix (floats)
                          shape = (n_samples, n_emails)
        recipients_map (ndarray): List of emails strings in the same order
                                   as axis 1 of Y_pred
    Returns:
        predictions (ndarray): Top predicted emails (strings)
                               shape = n_samples, top
    """
    # Get top indexes
    if len(Y_pred.shape) > 1 and top < Y_pred.shape[1]:
        best_pred_idx = np.argpartition(-Y_pred, top, axis=1)[:, :top]
        sorted_ids = np.argsort(
            Y_pred[np.arange(Y_pred.shape[0])[:, None], best_pred_idx]
            )[:, ::-1]
        sorted_idx = best_pred_idx[
            np.arange(best_pred_idx.shape[0])[:, None],
            sorted_ids
            ]
    elif len(Y_pred.shape) > 1 and top >= Y_pred.shape[1]:
        sorted_idx = np.argsort(-Y_pred, axis=1)
    else:
        sorted_idx = np.zeros((len(Y_pred), 1)).astype(int)

    # Map these indices to emails
    recipients_array = np.array(recipients_map)
    predictions = np.apply_along_axis(
        recipients_array.__getitem__, 1, sorted_idx)
    return predictions


def get_precision(prediction, ground_truth):
    '''Computes the precision at 10 (or len(prediction)).
        Arguments:
            - ground_truth (str or list): the true recipients.
            - prediction (str or list): the predicted recipients.
        Output:
            - float: the precision
    '''
    # If we got strings in input, convert to list
    if type(ground_truth) == str:
        ground_truth = ground_truth.split()
    if type(prediction) == str:
        prediction = prediction.split()
    # First remove None values
    ground_truth = [rec for rec in ground_truth if rec is not None]
    n_truth = len(ground_truth)
    n_pred = len(prediction)
    # we identify which predictions are correct
    pred_truth = [int(rec in ground_truth) for rec in prediction]
    truth_ids = [i for i, b in enumerate(pred_truth) if b]
    # we compute the precision at each rank
    precision_at_rank = np.cumsum(pred_truth) / (np.arange(n_pred)+1)
    precision = np.sum(precision_at_rank[truth_ids]) / np.min([n_truth, n_pred])
    return precision


def evaluate(pred_recipients, true_recipients):
    """
    Mean Average Precision @10 on all predictions
    Args:
        pred_recipients (2D list): List of lists of 10 strings (predictions)
        pred_recipients (2D list): List of lists of strings (true values)
    Returns:
        score (float): The Mean Average Precision @10
    """
    assert len(pred_recipients) == len(true_recipients)
    score = 0
    for pred, true in zip(pred_recipients, true_recipients):
        score += get_precision(pred, true)
    score /= len(pred_recipients)
    return score


def test_get_precision():
    # Test 1
    true_recipients = ["true"]
    pred_recipients = ["false"]*10
    # First prediction is true
    pred_recipients[0] = "true"
    score = get_precision(pred_recipients, true_recipients)
    assert score == 1

    # Test 2
    true_recipients = ["true"]
    pred_recipients = ["false"]*10
    # Seventh prediction is true
    pred_recipients[6] = "true"
    score = get_precision(pred_recipients, true_recipients)
    assert score == (1/7)/min(1, 10)

    # Test 3
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # First and third predictions are true
    pred_recipients[0] = "true1"
    pred_recipients[2] = "true2"
    score = get_precision(pred_recipients, true_recipients)
    assert score == (1/1 + 2/3)/min(3, 10)

    # Test 4
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # First and second predictions are true
    pred_recipients[0] = "true1"
    pred_recipients[1] = "true3"
    score = get_precision(pred_recipients, true_recipients)
    assert score == (1/1 + 2/2)/min(3, 10)

    # Test 5
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # Ninth and tenth predictions are true
    pred_recipients[8] = "true3"
    pred_recipients[9] = "true2"
    score = get_precision(pred_recipients, true_recipients)
    assert score == (1/9 + 2/10)/min(3, 10)
test_get_precision()
