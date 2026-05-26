from GREEN.green_score import GREEN
from scipy.stats import bootstrap
import numpy as np

class Green:
    def __init__(self, model_name= "StanfordAIMI/GREEN-radllama2-7b", 
                 output_dir=".", use_aggregator=True):
        self.model_name = model_name
        self.output_dir = output_dir
        self.scorer = GREEN(model_name=self.model_name, output_dir=self.output_dir)
        self.use_aggregator = use_aggregator

    def compute(self, predictions, references, results_file):
        mean, std, green_score_list, summary, result_df = self.scorer(references, predictions)
        bs= None
        if self.use_aggregator:
            bs = self.bootstrap_confidence_interval(green_score_list)
        result = {
            'mean': mean,
            'std': std,
            'summary': summary,
            'result_df': result_df,
            'bs': bs,
            'greenscore':green_score_list
        }
        return result
    
    @staticmethod
    def compute_statistic(reward_list):
        def _compute(indices):
            r = [reward_list[i] for i in indices]
            return np.mean(r, axis=0)

        return _compute
 
    @staticmethod
    def bootstrap_confidence_interval(green_score_list, n_samples: int = 500, method: str = "percentile"):
        bs = bootstrap(
            data=(list(range(len(green_score_list))),),
            statistic=Green.compute_statistic(green_score_list),
            method=method,
            paired=False,
            vectorized=False,
            n_resamples=n_samples,
            confidence_level=0.95,
            random_state=3
        )
        return bs

if __name__ == '__main__':
    test_hyps = ["Interstitial opacities at bases without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Endotracheal and nasogastric tubes have been removed. Changes of median sternotomy, with continued leftward displacement of the fourth inferiomost sternal wire. There is continued moderate-to-severe enlargement of the cardiac silhouette. Pulmonary aeration is slightly improved, with residual left lower lobe atelectasis. Stable central venous congestion and interstitial pulmonary edema. Small bilateral pleural effusions are unchanged.",
    "PA and lateral views of the chest.  The lungs are clear.  There is no\n effusion or pneumothorax.  The cardiomediastinal silhouette is normal.  No\n acute osseous abnormality is identified.",
    "In comparison with the study of ___, there is little change in the\n appearance of the heart and lungs.  Again there is a large area of opacification\n in the right mid and lower zones consistent with pleural fluid and underlying\n consolidation.  The left lung is essentially clear.",
    "In comparison with the study of ___, there is little change and no\n evidence of acute cardiopulmonary disease.  No pneumonia, vascular congestion,\n or pleural effusion.  Surgical clips are seen in the right neck."
    ]
    test_refs = ["Interstitial opacities without changes.",
    "Interval development of segmental heterogeneous airspace opacities throughout the lungs . No significant pneumothorax or pleural effusion . Bilateral calcified pleural plaques are scattered throughout the lungs . The heart is not significantly enlarged .",
    "Lung volumes are low, causing bronchovascular crowding. The cardiomediastinal silhouette is unremarkable. No focal consolidation, pleural effusion, or pneumothorax detected. Within the limitations of chest radiography, osseous structures are unremarkable.",
    "A right upper extremity PICC has been\n removed in the interim.  \n \n There is obscuration of the left heart border, likely scarring from prior\n infection.  There is no pleural effusion or pneumothorax.  The heart size is\n normal.  The mediastinal and hilar structures are unremarkable.",
    "Frontal and lateral views of the chest.  When compared to previous exams,\n there has been no significant interval change.  Right-sided chest tube remains\n in place.  Loculated fluid seen laterally similar to prior CT as well as\n within the major fissure where the chest tube is located.  Underlying\n parenchymal opacity again noted and based on scout film from prior CT has not\n significantly changed.  There is no left-sided pleural effusion.  Focal left\n midlung opacity is unchanged from prior.  Cardiomediastinal silhouette is\n difficult to adequately assess given obscuration of the right heart border. \n No acute osseous abnormalities detected.",
    "In comparison with the study of ___, the right lower lobe\n consolidation has cleared.  No evidence of acute focal pneumonia, vascular\n congestion, or pleural effusion.\n \n Vascular shunts are again seen, as are the multiple rounded calcifications\n projecting over the spleen."]
    # test_refs = test_hyps
    green = Green(output_dir="/home/csgrad/mbhosale/phd/MrFair/LLaVA-Rad/results/small_tmp_llavarad")
    res = green.compute(test_hyps, test_refs, results_file=None)
    print(res)