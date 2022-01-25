# BPER (Bayesian Personalized Explanation Ranking) & BPER-J (Joint-ranking)

## Papers
- Lei Li, Yongfeng Zhang, Li Chen. [On the Relationship between Explanation and Recommendation: Learning to Rank Explanations for Improved Performance](https://arxiv.org/abs/2102.00627). 2021.
- Lei Li, Yongfeng Zhang, Li Chen. [EXTRA: Explanation Ranking Datasets for Explainable Recommendation](https://lileipisces.github.io/files/SIGIR21-EXTRA-paper.pdf). SIGIR'21 Resource.

## Datasets to [download](https://drive.google.com/drive/folders/1Kb4pOCUja1EgDlhP-YQI8AxofHBkioT5?usp=sharing)
- Amazon Movies & TV
- TripAdvisor Hong Kong
- Yelp 2019

If you are interested in how to create the datasets, please refer to [EXTRA](https://github.com/lileipisces/EXTRA).

## Usage
Below are examples of how to run BPER and BPER-J.
```
python -u run_bper.py \
--data_path ../Amazon/IDs.pickle \
--index_dir ../Amazon/2/ \
--mu_on_user 0.7 >> bper.log

python -u run_bperj.py \
--data_path ../Amazon/IDs.pickle \
--index_dir ../Amazon/1/ \
--alpha 0.6 \
--mu_on_user -1 >> bperj.log
```

## Friendly reminders
- If you want to do follow-up works on our BPER/BPER-J, please modify the code of BPER+, as it is more efficient.
- If you do so, please set the maximum iteration number to a relatively large value, e.g., ```--epochs 50```.

## Code dependency
- Python 3.6

## Citations
```
@article{2021-BPER,
	title={On the Relationship between Explanation and Recommendation: Learning to Rank Explanations for Improved Performance},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	journal={arXiv preprint arXiv:2102.00627},
	year={2021}
}
@inproceedings{SIGIR21-EXTRA,
	title={EXTRA: Explanation Ranking Datasets for Explainable Recommendation},
	author={Li, Lei and Zhang, Yongfeng and Chen, Li},
	booktitle={SIGIR},
	year={2021}
}
```
