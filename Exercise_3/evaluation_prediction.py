def evaluation(predictions, labels):
    if (len(predictions) == len(labels)):
        different_elements = 0
        for i in range(0, len(labels)):
            if predictions[i] != labels[i]:
                different_elements += 1
        return 1 - (different_elements / len(predictions))
