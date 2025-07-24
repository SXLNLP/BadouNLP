from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Evaluate:
     def __init__(self):
         pass
     def evaluate(self, y_true, y_pred):
         acc = accuracy_score(y_true, y_pred)
         pre = precision_score(y_true, y_pred)
         rec = recall_score(y_true, y_pred)
         f1 = f1_score(y_true, y_pred)

         return {
             'acc': acc,
             'pre': pre,
             'rec': rec,
             'f1_score': f1
         }

     def print_metrics(self, metrics):
         print(f"Accuracy: {metrics['acc']:.4f}")
         print(f"Precision: {metrics['pre']:.4f}")
         print(f"Recall: {metrics['rec']:.4f}")
         print(f"F1 Score: {metrics['f1_score']:.4f}")
