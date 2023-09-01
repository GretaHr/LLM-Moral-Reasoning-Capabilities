from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, mean_absolute_error
import tensorflow as tf

# scenarios ordered by ascending human.response

y_gold = [0.145161290, 0.174603175, 0.250000000, 0.255813954, 0.270833333, 0.274193548, 0.418604651, 0.472972973, 0.675675676, 0.702702703, 0.703333333, 0.779069767, 0.791666667, 0.840909091, 0.866666667, 0.883720930]

y_yes_no = [0, 0.125, 0.6, 0, 0.571428571, 0, 0, 0.6, 1, 1, 0.428571429, 0, 0.857142857, 1, 0.142857143, 0.75]
# exclude @[0] 0, only 1 decisive run

y_eth_com = [0, 0, 0.1, 0.125, 0, 0, 0.111111111, 0.1, 0, 0.222222222, 0.1, 0, 0.111111111, 0, 0, 0.555555556]

y_step_vic = [0, 0, 0, 0, 0, 0, 0.1, 0, 0, 0.5, 0, 0, 0.833333333, 0.6, 0, 1]
# exclude @[9] 0.5, only 2 decisive runs

y_step_fal = [0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 0, 0, 0, 0, 0, 0]

cases = [y_yes_no, y_eth_com, y_step_vic, y_step_fal]

# probability to 0/1 label
def to_binary(results):
    return [1 if r > 0.5 else 0 for r in results]


# f1 score
def f1(y_true, y_pred):
    return f1_score(to_binary(y_true), to_binary(y_pred), zero_division=1.0)

# accuracy
def acc(y_true, y_pred):
    return accuracy_score(to_binary(y_true), to_binary(y_pred))

# conservativity
def cons(y_true, y_pred):
    cm = confusion_matrix(to_binary(y_true), to_binary(y_pred))
    return cm[1][0]/(cm[1][0]+cm[0][1])
    
# mean absolute error
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# binary cross entropy
def bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(to_binary(y_true), y_pred).numpy()


def evaluate():

    for y_pred in cases:
        y_true = y_gold

        # exclude first scenario
        if y_pred == y_yes_no:
            y_pred = y_yes_no[1:]
            y_true = y_gold[1:]

        # exclude tenth scenario
        elif y_pred == y_step_vic:
            y_pred = y_step_vic[:9] + y_step_vic[10:]
            y_true = y_gold[:9] + y_gold[10:]

        scores = [f1(y_true, y_pred), acc(y_true, y_pred), cons(y_true, y_pred), mae(y_true, y_pred), bce(y_true, y_pred)]
        print("\t".join([str(s) for s in scores]))


if __name__ == '__main__':
    evaluate()
