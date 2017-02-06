def evaluate_one(pred_recipients, true_recipients):
    """
    Mean Average Precision @10 on one prediction
    Args:
        pred_recipients (1D list): List of 10 strings being the predictions
        true_recipients (1D list): List of strings being the true values
    Returns:
        score (float): The Mean Average Precision @10
    """
    assert len(pred_recipients) == 10
    pred = [int(rec in true_recipients) for rec in pred_recipients]
    score = 0
    for i in range(len(pred)):
        if pred[i]:
            j = i+1
            score += sum(pred[:j])/j
    score /= min(len(true_recipients), len(pred_recipients))
    return score


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
        score += evaluate_one(pred_recipients, true_recipients)
    score /= len(pred_recipients)
    return score


def test_evaluate_one():
    # Test 1
    true_recipients = ["true"]
    pred_recipients = ["false"]*10
    # First prediction is true
    pred_recipients[0] = "true"
    score = evaluate_one(pred_recipients, true_recipients)
    assert score == 1

    # Test 2
    true_recipients = ["true"]
    pred_recipients = ["false"]*10
    # Seventh prediction is true
    pred_recipients[6] = "true"
    score = evaluate_one(pred_recipients, true_recipients)
    assert score == (1/7)/min(1, 10)

    # Test 3
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # First and third predictions are true
    pred_recipients[0] = "true1"
    pred_recipients[2] = "true2"
    score = evaluate_one(pred_recipients, true_recipients)
    assert score == (1/1 + 2/3)/min(3, 10)

    # Test 4
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # First and second predictions are true
    pred_recipients[0] = "true1"
    pred_recipients[1] = "true3"
    score = evaluate_one(pred_recipients, true_recipients)
    assert score == (1/1 + 2/2)/min(3, 10)

    # Test 5
    true_recipients = ["true1", "true2", "true3"]
    pred_recipients = ["false"]*10
    # Ninth and tenth predictions are true
    pred_recipients[8] = "true3"
    pred_recipients[9] = "true2"
    score = evaluate_one(pred_recipients, true_recipients)
    assert score == (1/9 + 2/10)/min(3, 10)
test_evaluate_one()
