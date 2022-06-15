# BPER+ (Bayesian Personalized Explanation Ranking enhanced by BERT)

## Papers
- Lei Li, Yongfeng Zhang, Li Chen. [On the Relationship between Explanation and Recommendation: Learning to Rank Explanations for Improved Performance](https://arxiv.org/abs/2102.00627). 2021.
- Lei Li, Yongfeng Zhang, Li Chen. [EXTRA: Explanation Ranking Datasets for Explainable Recommendation](https://lileipisces.github.io/files/SIGIR21-EXTRA-paper.pdf). SIGIR'21 Resource.

## Datasets to [download](https://lifehkbueduhk-my.sharepoint.com/:f:/g/personal/16484134_life_hkbu_edu_hk/EosVj7lRYKhBgpfdXRUDm84Ba4y0Kuueye3e1m0e3dQYEQ?e=4PUnLC)
- Amazon Movies & TV
- TripAdvisor Hong Kong
- Yelp 2019

If you are interested in how to create the datasets, please refer to [EXTRA](https://github.com/lileipisces/EXTRA).

## Usage
Below is an example of how to run BPER+.
```
python -u run_bperp.py \
--cuda \
--data_dir ../Amazon/ \
--index_dir ../Amazon/2/ \
--lr 0.0001 >> bperp.log
```

## Use pre-downloaded BERT
* Download the three files and put them in a folder, e.g., ./bert-base-uncased/
    * [vocab.txt](https://huggingface.co/bert-base-uncased/blob/main/vocab.txt)
    * [config.json](https://huggingface.co/bert-base-uncased/blob/main/config.json)
    * [pytorch_model.bin](https://huggingface.co/bert-base-uncased/blob/main/pytorch_model.bin)
* Run the program
```
python -u run_bperp.py \
--cuda \
--data_dir ../Amazon/ \
--index_dir ../Amazon/2/ \
--model_name ./bert-base-uncased/ \
--lr 0.0001 >> bperp.log
```

## Friendly reminders
- If you want to do follow-up works on our BPER/BPER-J, please modify the code of BPER+, as it is more efficient.
- If you do so, please set the maximum iteration number to a relatively large value, e.g., ```--epochs 50```.

## Code dependencies
- Python 3.6
- PyTorch 1.6

## Code reference
- [Tutorial: Fine tuning BERT for Sentiment Analysis](https://skimai.com/fine-tuning-bert-for-sentiment-analysis/#D---Fine-tuning-BERT)

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
