from sklearn.metrics import confusion_matrix
import numpy as np
from maxcorr import indicator

class FairnessMetric:
    def __init__(self, sensitive_attribute):
        self.sensitive_attribute = sensitive_attribute
    
    def __call__(self, y_true, y_pred):
        pass

class EqualizedOdds(FairnessMetric):
    def __init__(self, sensitive_attribute):
        super().__init__(sensitive_attribute)
    
    def __call__(self, y_true, y_pred):
        groups = np.unique(self.sensitive_attribute)
        tpr_fpr_by_group = {}

        for group in groups:
            group_indices = self.sensitive_attribute.ravel() == group
            group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_fpr_by_group[group] = {'tpr': tpr, 'fpr': fpr}

        tprs = [tpr_fpr_by_group[group]['tpr'] for group in groups]
        fprs = [tpr_fpr_by_group[group]['fpr'] for group in groups]

        disparities = {
            'tpr_diff': max(tprs) - min(tprs),
            'fpr_diff': max(fprs) - min(fprs),
        }

        return disparities['tpr_diff']


class DemographicParity(FairnessMetric):
    def __init__(self, sensitive_attribute):
        super().__init__(sensitive_attribute)

    def __call__(self, y_true, y_pred):
        groups = np.unique(self.sensitive_attribute)
        acceptance_rates = {}

        for group in groups:
            group_indices = self.sensitive_attribute.ravel() == group
            group_y_pred = y_pred[group_indices]
            group_y_true = y_true[group_indices]
            tn, fp, fn, tp = confusion_matrix(group_y_true, group_y_pred).ravel()
            acceptance_rate = (tp + fp)/(tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            acceptance_rates[group] = acceptance_rate

        disparities = np.abs(acceptance_rates[0] - acceptance_rates[1])

        return disparities
    
class HgrMetric(FairnessMetric):
    def __init__(self, sensitive_attribute):
        super().__init__(sensitive_attribute)
    
    def __call__(self, y_true, y_pred):
        ind = indicator('hgr', 'dk', 'numpy')
        return ind.compute(y_pred, self.sensitive_attribute)


