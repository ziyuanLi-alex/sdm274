import numpy as np

class Evaluation():

    def calculate_accuracy(self, y_actual, y_pred):

        correct_predictions = 0
        total_predictions = len(y_actual) # 其实y_actual和y_pred的长度是一样的

        for i in range(total_predictions):
            if y_actual[i] == y_pred[i]:
                correct_predictions += 1

        return correct_predictions / total_predictions

    def calculate_recall(self, y_actual, y_pred, positive_label=1):
        true_positive = 0
        false_negative = 0

        for i in range(len(y_actual)):
            if y_actual[i] == positive_label:
                if y_pred[i] == positive_label:
                    true_positive += 1
                else:
                    false_negative += 1

        recall = true_positive / (true_positive + false_negative)
        # tp + fn 为真值的总数
        return recall

    def calculate_precision(self, y_actual, y_pred, positive_label=1):
        true_positive = 0
        false_positive = 0

        for i in range(len(y_actual)):
            if y_pred[i] == positive_label:
                if y_actual[i] == positive_label:
                    true_positive += 1
                else:
                    false_positive += 1

        if true_positive + false_positive == 0:
            return 0
        precision = true_positive / (true_positive + false_positive)
        # 在预测为正的情况下，有多少是真的正值
        # 如果模型过多地判定正值，那么recall会很高，但是precision会很低
        # 如果模型过多地判定负值，那么recall会很低，但是在判定为真值的样本里真值往往更多。（过于保守）
        
        return precision

    def calculate_f1_score(self, y_actual, y_pred, positive_label=1):
        precision = self.calculate_precision(y_actual, y_pred, positive_label)
        recall = self.calculate_recall(y_actual, y_pred, positive_label)
        if precision + recall == 0:
            return 0
        f1_score = 2 * precision * recall / (precision + recall)
        return f1_score
    
    def evaluate(self, y_actual, y_pred, positive_label=1):
        accuracy = self.calculate_accuracy(y_actual, y_pred)
        recall = self.calculate_recall(y_actual, y_pred, positive_label)
        precision = self.calculate_precision(y_actual, y_pred, positive_label)
        f1_score = self.calculate_f1_score(y_actual, y_pred, positive_label)
        
        print(f'Accuracy: {accuracy}')
        print(f'Recall: {recall}')
        print(f'Precision: {precision}')
        print(f'F1 Score: {f1_score}')

        return
            

